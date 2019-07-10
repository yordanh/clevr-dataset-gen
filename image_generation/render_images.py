# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

import get_b_box

import numpy as np
import os.path as osp

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.5, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.5, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=20, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_dir', default='../output',
    help="The directory where output data will be stored. It will be " +
         "created if it does not exist.")


parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  arr_template = '%s%%0%dd.npz' % (prefix, num_digits)
  
  img_template = osp.join(args.output_dir, 'train/images', img_template)
  scene_template = osp.join(args.output_dir, 'train/scenes', scene_template)
  blend_template = osp.join(args.output_dir, 'train/blend', blend_template)
  arr_template = osp.join(args.output_dir, 'train/arr', arr_template)

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  if not os.path.isdir(osp.join(args.output_dir, 'train/images')):
    os.makedirs(osp.join(args.output_dir, 'train/images'))

  if not os.path.isdir(osp.join(args.output_dir, 'train/scenes')):
    os.makedirs(osp.join(args.output_dir, 'train/scenes'))

  if not os.path.isdir(osp.join(args.output_dir, 'train/arr')):
    os.makedirs(osp.join(args.output_dir, 'train/arr'))

  if args.save_blendfiles == 1 and not os.path.isdir(osp(args.output_dir, 'train/blend')):
    os.makedirs(osp(args.output_dir, 'train/blend'))
  
  all_scene_paths = []
  for i in range(args.num_images):

    print("GENERATING FRAME {0}/{1}".format(i, args.num_images))

    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    arr_path = arr_template % (i + args.start_idx)

    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)

    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_arr=arr_path,
      output_blendfile=blend_path,
      scene_idx=i
    )

  exit()



def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_arr='arr',
    output_blendfile=None,
    scene_idx=0
  ):
  
  for object_name in bpy.data.objects.keys():
    if 'Sphere' in object_name or\
       'Cylinder' in object_name or\
       'Cube' in object_name or\
       'Duck' in object_name or\
       'Peg' in object_name or\
       'Disk' in object_name or\
       'Bowl' in object_name or\
       'Tray' in object_name:
       utils.delete_object_by_name(object_name)

  # Load the main blendfile
  # bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  scene_struct = {}
  render_args = bpy.context.scene.render
  render_args.filepath = output_image

  if scene_idx == 0:
    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
      # Blender changed the API for enabling CUDA at some point
      if bpy.app.version < (2, 78, 0):
        bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        bpy.context.user_preferences.system.compute_device = 'CUDA_0'
      else:
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
      bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)



  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, scene_idx)

  ### get b_box
  box_dict = get_b_box.main(bpy.context, blender_objects)
  for _id in box_dict:
    objects[_id]['bbox'] = box_dict[_id]

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)


  tree = bpy.context.scene.node_tree
  links = tree.links
  rl = tree.nodes['Render Layers']
  v = tree.nodes['Viewer']

  links.new(rl.outputs[0], v.inputs[0])
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)
  links.remove(links[0])

  # get viewer pixels
  rgb_pixels = bpy.data.images['Viewer Node'].pixels
  rgb_pixels = np.array(rgb_pixels[:])
  rgb_pixels = np.power(rgb_pixels, 1/2.2)
  rgb_pixels[rgb_pixels > 1] = 1
  rgb_pixels = rgb_pixels.reshape(args.height, args.width, 4)[...,:3]

  links.new(rl.outputs[2], v.inputs[0])
  render_shadeless(blender_objects, lights_off=False)
  links.remove(links[0])

  # get viewer pixels
  depth_pixels = bpy.data.images['Viewer Node'].pixels
  depth_pixels = np.array(depth_pixels[:])
  depth_pixels = depth_pixels.reshape(args.height, args.width, 4)[...,0, None]


  links.new(rl.outputs[0], v.inputs[0])
  render_shadeless(blender_objects)
  links.remove(links[0])

  # get viewer pixels
  mask_pixels = bpy.data.images['Viewer Node'].pixels
  mask_pixels = np.array(mask_pixels[:])
  mask_pixels = mask_pixels.reshape(args.height, args.width, 4)[...,:3]

  pixels = np.concatenate((rgb_pixels, depth_pixels, mask_pixels), axis=2)
  pixels = np.flipud(pixels)

  utils.save_arr(pixels, output_arr)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

def add_random_objects(scene_struct, num_objects, args, camera, scene_idx=0):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []

  last_obj_params = []

  for i in range(num_objects):

    # Choose random color and shape
    obj_name, obj_name_out = random.choice(object_mapping)
    
    while i != 0 and obj_name_out == "tray":
      obj_name, obj_name_out = random.choice(object_mapping)

    if i == 0:
      if np.random.rand() > 0.5:
        obj_name, obj_name_out = ("Tray", "tray")


    if obj_name_out == "tray":
      color_name, rgba = "gray", [0.66, 0.66, 0.66, 1]
    else:
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))

    # Choose a random size 
    if obj_name_out == "tray":
      size_name, r = "large", 0.6
    else:
      size_name, r = random.choice(size_mapping)
  
    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
      
      if np.random.rand() > 0.5 and \
         last_obj_params != [] and \
         ("Cube" in last_obj_params[4] or "Cylinder" in last_obj_params[4]) and \
         last_obj_params[2] < 1:
        
        x = last_obj_params[0] + random.uniform(-0.3, 0.3)
        y = last_obj_params[1] + random.uniform(-0.3, 0.3)
        z = last_obj_params[2] + (2*last_obj_params[3])

        min_dist = 0
        directions_to_check = []

      elif np.random.rand() > 0 and \
         last_obj_params != [] and \
         ("Tray" in last_obj_params[4]):
        
        x = last_obj_params[0] + random.uniform(-0.3, 0.3)
        y = last_obj_params[1] + random.uniform(-0.3, 0.3)
        z = last_obj_params[2]

        min_dist = -10
        directions_to_check = []

      else:
        x = random.choice(np.linspace(-2, 2, 25, endpoint=True))
        y = random.choice(np.linspace(-2, 2, 25, endpoint=True))
        z = random.choice(np.linspace(0, 2, 25, endpoint=True))

        if obj_name_out == "tray":
          z = random.choice(np.linspace(0, 1, 25, endpoint=True))

        # x = 0
        # y = 0
        # z = 0

        min_dist = args.min_dist
        directions_to_check = [x for x in scene_struct['directions'].keys()]
      

      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, zz, rr) in positions:
        dx, dy, dz = x - xx, y - yy, z - zz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist - r - rr < min_dist:
          # print('BROKEN DISTANCE!', dist - r - rr, min_dist)
          dists_good = False
          break
        for direction_name in directions_to_check:
          direction_vec = scene_struct['directions'][direction_name]
          margin = dx * direction_vec[0] + dy * direction_vec[1] + dz * direction_vec[2]
          if 0 < margin < args.margin:
            # print(margin, args.margin, direction_name)
            # print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    # theta = 360.0 * random.random()
    theta = 0

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y, z), theta=theta)
    last_obj_params = [x, y, z, r, obj_name]

    utils.add_material(mat_name, Color=rgba)

    obj = bpy.context.object
    blender_objects.append(obj)

    # if obj_name_out != "tray":
    positions.append((x, y, z, r))

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'r' : r,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

    if obj_name_out == "tray":
      objects[-1]["mask_color"] = [0.66, 0.66, 0.66]

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded or too small; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.5):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  close_far_thresh = 2
  above_below_xy_thresh = 0.3
  
  for name, direction_vec in scene_struct['directions'].items():
    # if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))

  all_relationships['on'] = []
  all_relationships['off'] = []
  all_relationships['close'] = []
  all_relationships['far'] = []
  all_relationships['in'] = []
  all_relationships['out'] = []

  for i, obj1 in enumerate(scene_struct['objects']):
    coords1 = obj1['3d_coords']
    r1 = obj1['r']

    related_on = set()
    related_off = set()
    related_close = set()
    related_far = set()
    related_in = set()
    related_out = set()

    for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        r2 = obj2['r']

        dx = coords1[0] - coords2[0]
        dy = coords1[1] - coords2[1]
        dz = coords1[2] - coords2[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # ON
        if abs(coords1[0] - coords2[0]) < r1 and\
           abs(coords1[1] - coords2[1]) < r1 and\
           np.round(coords2[2], 2) == np.round(r1 + r2 + coords1[2], 2) and\
           obj1['shape'] != "tray":
           related_on.add(j)
        # OFF
        elif (abs(coords1[0] - coords2[0]) > r1 or\
           abs(coords1[1] - coords2[1]) > r1 or \
           np.round(coords2[2], 2) != np.round(r1 + r2 + coords1[2], 2)) and\
           obj1['shape'] != "tray":
           related_off.add(j)

        # IN
        if abs(coords1[0] - coords2[0]) < eps and\
           abs(coords1[1] - coords2[1]) < eps and\
           np.round(coords2[2], 2) == np.round(r2 + coords1[2], 2) and\
           obj1['shape'] == "tray":
           related_in.add(j)
           print("IN")
        # OUT
        elif (abs(coords1[0] - coords2[0]) > eps or\
           abs(coords1[1] - coords2[1]) > eps or \
           np.round(coords2[2], 2) > np.round(r2 + coords1[2] + 0.3, 2)) and\
           obj1['shape'] == "tray":
           related_out.add(j)
           print("OUT")

        # CLOSE
        if np.round(dist, 2) < close_far_thresh :
          related_close.add(j)
        # FAR
        elif np.round(dist, 2) > close_far_thresh :
          related_far.add(j)
    
    all_relationships['on'].append(sorted(list(related_on)))
    all_relationships['off'].append(sorted(list(related_off)))
    all_relationships['close'].append(sorted(list(related_close)))
    all_relationships['far'].append(sorted(list(related_far)))
    all_relationships['in'].append(sorted(list(related_in)))
    all_relationships['out'].append(sorted(list(related_out)))

  return all_relationships


def check_visibility(blender_objects, objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  tree = bpy.context.scene.node_tree
  links = tree.links
  rl = tree.nodes['Render Layers']
  v = tree.nodes['Viewer']
  links.new(rl.outputs[0], v.inputs[0])

  object_colors = render_shadeless(blender_objects)
  links.remove(links[0])

  # get viewer pixels
  mask_pixels = bpy.data.images['Viewer Node'].pixels
  p = list(mask_pixels)

  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3]) for i in range(0, len(p), 4))

  # Some objects are fully occluded
  if len(color_count) != len(blender_objects) + 1:
    return False

  # Some objects are partially occluded or too small
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False

  obj_pixel_vals = []
  mask_pixels = np.array(mask_pixels[:])
  mask_pixels = mask_pixels.reshape(args.height, args.width, 4)
  mask_pixels = np.flipud(mask_pixels)
  
  for i, obj in enumerate(objects):
    pixel_coords = obj['pixel_coords'][:2]
    pixel_coords = [127 if x > 127 else x for x in pixel_coords]
    pixel_coords = [0 if x < 0 else x for x in pixel_coords]
    obj_pixel_val = mask_pixels[pixel_coords[1], pixel_coords[0]]
    obj_pixel_val = [float('%.2f' % round(x, 2)) for x in obj_pixel_val]

    if obj_pixel_val == [0.66, 0.66, 0.66] and objects[i].shape != "tray":
      return False

    obj_pixel_vals.append(list(obj_pixel_val))

  # Some objects are partially occluded
  # Checks whether the mask for each object is 
  # extractable later on
  if len(set(tuple(x) for x in obj_pixel_vals)) != len(objects):
    return False
    
  return True


def render_shadeless(blender_objects, path='flat.png', lights_off=True):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  if lights_off:
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    print(blender_objects[i].name)
    if blender_objects[i].name == "Tray_0":
      (r, g, b) = (0.66, 0.66, 0.66)
    else:
      while True:
        r, g, b = [random.random() for _ in range(3)]
        if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=False)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  if lights_off:
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

