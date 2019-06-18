#!/bin/bash

BLENDER="/home/yordan/blender-2.79b-linux-glibc219-x86_64/blender"

NUM_FOLDERS=5
NUM_IMAGES_PER_FOLDER=25
NUM_OBJECTS=2
IMAGE_SIZE=128
START_I=0
BASE_DIR="../outputs_test/no_no_no_no_off-on"

for (( i=$START_I; i<(START_I + $NUM_FOLDERS); i++))
do
	$BLENDER data/base_scene.blend --python render_images_test.py -- --use_gpu 1 --output_dir $BASE_DIR"/"$i --num_images $NUM_IMAGES_PER_FOLDER --width $IMAGE_SIZE --height $IMAGE_SIZE --min_objects $NUM_OBJECTS --max_objects $NUM_OBJECTS --camera_jitter 0

done
