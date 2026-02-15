import tensorrt as trt
import cv2
import numpy as np
import os
from random import shuffle
import argparse
import PIL.Image
from PIL import ImageDraw
import json
from typing import Tuple, List, Optional, Union

import pycuda.driver as cuda
import pycuda.autoinit

import glob

trt.init_libnvinfer_plugins(None, '')

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_data, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calib_data = calib_data
        self.device_input = cuda.mem_alloc(self.calib_data.calibration_data.nbytes)
        self.cache_file = cache_file
        
    def get_batch_size(self):
        return self.calib_data.calibration_batch
    
    def get_batch(self, names):
        try:
            batch = next(self.calib_data)
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            
class Dataloader:
    def __init__(self, images_path, calibration_batch, calibration_data_part, model_shape, means, stds):
        self.images_path: str = images_path
        self.images: list = [image for image in glob.iglob(self.images_path+'/**/*', recursive=True) if os.path.isfile(image) and image.split('.')[-1] in ['jpg', 'jpeg', 'png']]
        shuffle(self.images)
        self.images = self.images[:int(len(self.images)*calibration_data_part)]
        self.calibration_batch: int = calibration_batch
        self.model_shape: tuple = (calibration_batch, *model_shape)
        self.batch_counter = 0
        self.batches: int = len(self.images)//calibration_batch
        self.pretrained_means: list = means
        self.pretrained_stds: list = stds
        self.calibration_data = np.zeros(self.model_shape, dtype=np.float32)
        self.div = len(self.images)//10//calibration_batch
        self.target_size = model_shape[1:]
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def reset(self):
        self.batch_counter = 0
    
    def next(self) -> np.ndarray:
        if self.batch_counter < self.batches:
            batch = list()
            for i in range(self.calibration_batch):
                image: np.ndarray = cv2.imread(self.images[i+self.batch_counter*self.calibration_batch])
                image = self.__PreprocessImage(image)
                batch.append(image)
            batch = np.array(batch).astype(np.float32)
            if self.batch_counter % self.div == 0:
                print('Progress: {0}/{1}'.format(self.batch_counter, self.batches))
            self.batch_counter += 1
            return batch
        raise StopIteration()

    def letterbox(self, image):
        h, w = image.shape[:2]
        new_w = int(w * min(self.target_size[0]/w, self.target_size[1]/h))
        new_h = int(h * min(self.target_size[0]/w, self.target_size[1]/h))
        resized = cv2.resize(image, (new_w, new_h))
        new_image = np.full((self.target_size[1], self.target_size[0], 3), 128, dtype=np.uint8)
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return new_image
    
    def __PreprocessImage(self, image: np.ndarray) -> np.ndarray:
        image = self.letterbox(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255
        image = image.transpose(2,0,1)
        return image
    
class BuildEngine:
    
    def __init__(self, onnx_model_path: str = "", 
                 batch_sizes: list = [1,4,16], 
                 explicit: bool = True,  
                 d_type: str = "fp32", 
                 calibration_dataloader = None, 
                 calibration_data_path: str = None,
                 calibration_data_part: float = 1.0,
                 calibration_batch: int = 4,
                 calib_file_path: str = 'cal.bin',
                 trt_logger: trt.tensorrt.Logger.Severity=trt.Logger.WARNING, 
                 workspace: int = 1<<31, 
                 model_input_shape: Union[tuple, list] = None, 
                 use_dla: bool = False, 
                 means: list = [0,0,0], 
                 stds: list = [1,1,1],
                 metadata: dict = None):
        
        self.onnx_model_path = onnx_model_path
        self.batch_sizes = batch_sizes
        self.explicit = explicit
        self.d_type = d_type
        self.calibration_dataloader = calibration_dataloader
        self.metadata = metadata if metadata is not None else {}
        self.logger = trt.Logger(trt_logger)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(flags = int(self.explicit))
        self.onnx_parser = trt.OnnxParser(self.network, self.logger)
        
        with open(self.onnx_model_path, 'rb') as model:
            self.onnx_parser.parse(model.read())
            assert self.network.num_layers > 0, 'Failed to parse ONNX model.'
            
        if not model_input_shape:
            self.model_input_shape = self.network.get_input(0).shape[-3::]
            assert -1 not in self.model_input_shape[-2:], 'User must specify model shapes.'
        else:
            self.model_input_shape = model_input_shape
        
        self.config = self.builder.create_builder_config()
        self.config.set_tactic_sources(1 << int(trt.TacticSource.CUDNN))
        # self.config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # self.config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS_LT))
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if use_dla is True:
            self.config.default_device_type = trt.DeviceType.DLA
            self.config.DLA_core = 0
            
        if self.explicit is True:
            self.profile = self.builder.create_optimization_profile()
            self.profile.set_shape(self.network.get_input(0).name, 
                                   (self.batch_sizes[0], *self.model_input_shape),
                                   (self.batch_sizes[0], *self.model_input_shape), 
                                   (self.batch_sizes[0], *self.model_input_shape))
            success = self.config.add_optimization_profile(self.profile)
            assert success != -1, f'Opt profile error, returns {success}'
            
            if self.d_type == 'int8':
                success = self.config.set_calibration_profile(self.profile)
                assert success is True, f'Opt cal profile error, returns {success}'
        
        if self.d_type == 'fp16':
            self.config.set_flag(trt.BuilderFlag.FP16)
        elif self.d_type == 'int8':
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.set_flag(trt.BuilderFlag.DEBUG)
            
            if not self.calibration_dataloader:
                print('Custom dataloader was not provided, using default.')
                assert calibration_data_path, 'User must specify path to images(dataset) for calibration.'
                self.calibration_dataloader = Dataloader(calibration_data_path, 
                                                    calibration_batch, 
                                                    calibration_data_part, 
                                                    self.model_input_shape, 
                                                    means, 
                                                    stds)
            self.calibrator = Int8Calibrator(self.calibration_dataloader, calib_file_path)
            self.config.int8_calibrator = self.calibrator
        
        self.engine = self.builder.build_serialized_network(self.network, self.config)
        
    def SaveEngine(self, filename):
        with open(filename, 'wb') as engine_file:
            meta = json.dumps(self.metadata)
            engine_file.write(len(meta).to_bytes(4, byteorder='little', signed=True))
            engine_file.write(meta.encode())
            engine_file.write(self.engine)
            
parser = argparse.ArgumentParser('Convert to int8')
parser.add_argument('-m', '--model', type=str, required=True, help='onnx model path')
parser.add_argument('-d', '--data_path', type=str, required=True, help='calibration data path, image dataset path')
parser.add_argument('-t', '--d_type', type=str, required=False, default='int8', help='type model int8 or fp16')
parser.add_argument('-s', '--model_input_shape', type=tuple, required=False, help='model_input_shape')
parser.add_argument('-o', '--output_calib_file', type=str, required=False, help='output calib file path')

args = parser.parse_args()
print(''.join(args.model_input_shape).split(','))
if args.model_input_shape:
    args.model_input_shape = tuple(map(int,''.join(args.model_input_shape).split(',')))
    
metadata = None
engine = BuildEngine(onnx_model_path=args.model, d_type=args.d_type, calibration_data_path=args.data_path, model_input_shape=args.model_input_shape, calib_file_path=args.output_calib_file,  metadata=metadata)
engine.SaveEngine(args.output_calib_file.replace('bin', 'engine'))
