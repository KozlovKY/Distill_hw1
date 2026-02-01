import torch
import time
from collections import defaultdict
from typing import Dict, Optional, List
from pathlib import Path
from ultralytics import YOLO
        

class YOLOProfiler:
    def __init__(self, model_path: str):
        
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.model.eval()
        self.model.fuse()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _prepare_input(self, imgsz: int, batch_size: int, fp16: bool) -> torch.Tensor:
        dtype = torch.float16 if (fp16 and self.device.type == 'cuda') else torch.float32
        return torch.randn(batch_size, 3, imgsz, imgsz, dtype=dtype, device=self.device)
    
    def profile_operations(
        self,
        imgsz: int = 640,
        batch_size: int = 1,
        warmup: int = 10,
        iterations: int = 50,
        fp16: bool = True,
    ) -> Dict:
        """Profile using torch.profiler with correct attribute access."""
        print("PROFILING torch.profiler")
        print(f"Device: {self.device} | FP16: {fp16} | Input: {batch_size}x3x{imgsz}x{imgsz}")
        
        if fp16:
            self.model = self.model.half()
        
        dummy_input = self._prepare_input(imgsz, batch_size, fp16)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16 and self.device.type == 'cuda'):
            for _ in range(warmup):
                _ = self.model(dummy_input)
        torch.cuda.synchronize()
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.no_grad(), torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False, 
            with_flops=False   
        ) as prof:
            for _ in range(iterations):
                with torch.cuda.amp.autocast(enabled=fp16 and self.device.type == 'cuda'):
                    _ = self.model(dummy_input)
        
        torch.cuda.synchronize()
        
        print("\nTop 15 Time-Consuming Operations:")
        print(prof.key_averages().table(sort_by="cuda_time", row_limit=15))

        category_time = defaultdict(float)
        categories = {
            'conv': 'Convolutions',
            'gemm|matmul|linear': 'Matrix Multiplications',
            'batch_norm': 'BatchNorm',
            'relu|silu|sigmoid|hardswish': 'Activations',
            'max_pool|avg_pool': 'Pooling',
            'add|addmm': 'Additions',
            'mul': 'Multiplications',
            'cat|concat': 'Concatenations'
        }
        
        total_cuda_time = 0.0
        for event in prof.key_averages():
            cuda_time =  event.device_time
            total_cuda_time +=  cuda_time
            
            op_name = event.key.lower()
            matched = False
            for pattern, category in categories.items():
                if any(p in op_name for p in pattern.split('|')):
                    category_time[category] += cuda_time
                    matched = True
                    break
            if not matched:
                category_time['Other'] += cuda_time
        
        print("\nOperation Category Breakdown:")
        total_time = total_cuda_time
        for category, time_val in sorted(category_time.items(), key=lambda x: x[1], reverse=True):
            if time_val > 0:
                pct = (time_val / total_time) * 100
                time_ms = time_val / 1000  
                print(f"{category:<30} {time_ms:>8.2f} ms ({pct:>5.1f}%)")
        
        avg_time_ms = total_cuda_time / iterations / 1000
        return {
            'profiler': prof,
            'avg_time_ms': avg_time_ms,
            'fps': 1000 / avg_time_ms,
            'total_cuda_time_us': total_cuda_time,
            'category_time_us': dict(category_time),
        }
