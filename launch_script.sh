#!/usr/bin/env bash
export CVAT_HOST=$(hostname -i)
docker-compose -f docker-compose.yml -f docker-compose.override.yml -f components/serverless/docker-compose.serverless.yml up -d
cd serverless
./deploy_gpu.sh pytorch/facebookresearch/fcos/nuclio/
./deploy_gpu.sh pytorch/saic-vul/hrnet/nuclio/
./deploy_gpu.sh pytorch/ultralytics/yolov5_maize_onnx/
