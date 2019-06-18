import cv2
import numpy as np
import os
import os.path as osp
import json
import copy

DIR = "../outputs/clevr_data_128_3_obj_10/train"
# DIR = "../outputs_test/left-right_no_no/0/train"
ARR_DIR_READ = osp.join(DIR, "arr")
SCENE_DIR_READ = osp.join(DIR, "scenes")
DEPTH_DIR_WRITE = osp.join(DIR, "depth_image")
BGR_DIR_WRITE = osp.join(DIR, "bgr_image")
file_list = os.listdir(ARR_DIR_READ)

for file in file_list:
	file = "CLEVR_new_000057.npz"
	print(file)

	data = np.load(osp.join(ARR_DIR_READ, file))['arr_0']
	scene_file = open(osp.join(SCENE_DIR_READ, file.replace('.npz', '.json')), "r")
	json_data = json.load(scene_file)
	scene_objs = json_data['objects']
	rels = json_data['relationships']
	
	# print(data.shape)

	# print(np.max(data[...,:3]))
	# print(np.mean(data[...,:3]))

	bgr = (data[...,:3] * 255).astype(np.uint8)
	bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
	cv2.imshow("BGR", bgr)

	# bgr_saved = cv2.imread(osp.join('../output/images', file.replace('.npz', '.png')))
	# cv2.imshow("BGR Saved", bgr_saved)

	# print(bgr[100, 100])
	# print(bgr_saved[100, 100])

	depth = data[...,3]
	depth = depth / np.max(depth)
	depth = (depth * 255).astype(np.uint8)
	cv2.imshow("Depth", depth)

	mask = data[...,4:]
	mask = (mask * 255).astype(np.uint8)
	cv2.imshow("mask", mask)

	object_masks = {}

	for i, obj in enumerate(scene_objs):
		mask_tmp = mask.copy()
		pixel_coords = obj['pixel_coords'][:2]
		obj_pixel_val = mask_tmp[pixel_coords[1], pixel_coords[0]]

		object_masks[i] = (mask_tmp == obj_pixel_val).all(axis=2).astype(np.uint8)

		cv2.imshow("mask " + str(i), object_masks[i] * 255)
		# cv2.imwrite(osp.join(BGR_DIR_WRITE, file.replace('.npz', '.png')), mask_tmp * 255)
	cv2.waitKey()

	for rel_name, obj_list in rels.items():
		for ref_idx, target_list in enumerate(obj_list):
			for target_idx in target_list:
				mask_tmp = object_masks[ref_idx].copy() * 127
				mask_tmp += object_masks[target_idx].copy() * 255

				cv2.imshow(rel_name, mask_tmp)

	cv2.waitKey()
	
	# cv2.imwrite(osp.join(BGR_DIR_WRITE, file.replace('.npz', '.png')), bgr)
	# cv2.imwrite(osp.join(DEPTH_DIR_WRITE, file.replace('.npz', '.png')), depth)
