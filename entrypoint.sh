#!/bin/bash
set -e

MODEL_PATH="/models/depth_anything/1"
ONNX_FILE="$MODEL_PATH/depth_anything_v2_vits.onnx"
PLAN_FILE="$MODEL_PATH/model.plan"

# Convert ONNX to TensorRT if model.plan doesn't exist
if [ ! -f "$PLAN_FILE" ]; then
    echo "Converting ONNX model to TensorRT..."
    /usr/src/tensorrt/bin/trtexec --onnx="$ONNX_FILE" --saveEngine="$PLAN_FILE" --fp16 --memPoolSize=workspace:2048M
    echo "Conversion complete!"
else
    echo "TensorRT model already exists at $PLAN_FILE"
fi

# Start Triton server
exec tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1
