trtexec \
    --onnx=yolo11n.onnx \
    --saveEngine=yolo11n_fp16.engine \
    --fp16 \
    --builderOptimizationLevel=5