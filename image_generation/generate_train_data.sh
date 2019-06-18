#!/bin/bash

BLENDER="/home/yordan/blender-2.79b-linux-glibc219-x86_64/blender"

NUM_FOLDERS=1
NUM_IMAGES_PER_FOLDER=10
NUM_OBJECTS=3
IMAGE_SIZE=128
START_I=145

for (( i=$START_I; i<(START_I + $NUM_FOLDERS); i++))
do
	$BLENDER data/base_scene.blend --python render_images.py -- --use_gpu 1 --output_dir "../outputs/clevr_data_"$IMAGE_SIZE"_"$NUM_OBJECTS"_obj_"$i --num_images $NUM_IMAGES_PER_FOLDER --width $IMAGE_SIZE --height $IMAGE_SIZE --min_objects $NUM_OBJECTS --max_objects $NUM_OBJECTS --camera_jitter 0

done
