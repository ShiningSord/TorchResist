import os
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt

from simulator import AbbeSim, AbbeGradient
from utils import BBox, Point



def load_images_in_batches(folder_path, batch_size):
    # 获取文件夹中所有符合命名规则的图片文件名
    image_files = [f for f in os.listdir(folder_path) if f.startswith("mask_") and f.endswith(".png")]
    image_files.sort()  # 确保按顺序读取图片
    
    # 批量读取图片
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            # 读取灰度图像并转化为numpy array
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert('L')  # 转换为灰度图
            image_np = np.array(image, dtype=np.float32)  # 转换为numpy array
            
            # 添加通道维度并转换为tensor，格式为 (C, H, W)，C=1
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # unsqueeze to add the channel dimension
            
            batch_images.append(image_tensor)
        
        # 将单个图像tensor组合为 (B, C, H, W)
        batch_tensor = torch.stack(batch_images)  # 合并batch
        
        yield batch_tensor  # 返回batch tensor

# 示例用法
folder_path = 'path_to_your_folder'  # 替换为你的文件夹路径
batch_size = 8  # 你需要的batch大小
sigma = 0.05
pixelsize = 4
canvas = 1024
size = round(canvas/pixelsize)
device = torch.device("cuda")

sim1 = AbbeSim(None, pixelsize, sigma, defocus=None, batch=True, par=False)
grad1 = AbbeGradient(None, pixelsize, sigma, defocus=None, batch=True, par=False)
intensity_np_all = []

for idx, batch in enumerate(load_images_in_batches(folder_path, batch_size)):
    intensity = sim1(batch)
    intensity_np = intensity.cpu().numpy()
    intensity_np_all.append(intensity_np)
    for i in range(intensity_np.shape[0]):
    # 取出第 i 张图片，并去掉通道维度
        img_array = intensity_np[i, 0, :, :]  # 形状为 (H, W)
        
        # 将数组转换为 PIL Image 对象
        img = Image.fromarray(img_array, mode='L')  # 'L' 表示灰度图
        
        # 保存图像
        img.save(f'abbe_intensity_{i+idx*batch_size}.png')
intensity_np_all = np.concatenate(intensity_np_all,0)
np.save("abbe_intensity.npy",intensity_np_all)
