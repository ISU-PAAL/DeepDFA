#!/bin/bash
set -e

CONTAINER_ID="benjijang/deepdfa:latest"

docker pull $CONTAINER_ID
mkdir -p DDFA/storage LineVul/linevul/data
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --name deepdfa \
    -v "$PWD/DDFA/storage:/DeepDFA/DDFA/storage" -v "$PWD/LineVul/linevul/data:/DeepDFA/LineVul/linevul/data" \
    $CONTAINER_ID
