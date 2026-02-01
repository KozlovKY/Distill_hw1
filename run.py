import argparse
from advanced_profiler import YOLOProfiler
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt', help='Путь к модели')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--batch-size', type=int, default=1, help='Размер батча')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup итераций')
    parser.add_argument('--iterations', type=int, default=100, help='Итераций профилирования')
    
    args = parser.parse_args()
    
    profiler = YOLOProfiler(args.model)

    results = profiler.profile_operations(
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        fp16=True
    )
    # model = YOLO("yolo11n.pt")  
    # model.info(detailed=True)
    
if __name__ == "__main__":
    main()
    