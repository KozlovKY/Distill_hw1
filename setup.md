# to setup

```bash
git clone 
docker build -t yolo11-tensorrt-profile .
docker run --gpus all -it --rm -v $(pwd):/workspace yolo11-tensorrt-profile
```
Inside container:
```bash
python run.py --iterations 1000
```

To convert model to trt:
```bash
bash convert.sh
```
