# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import torch

from simulator import AbbeSim, AbbeGradient
from tqdm import tqdm


def load_images_in_batches(folder_path, batch_size, device):
 
    image_files = [f for f in os.listdir(folder_path) if f.startswith("cell") and f.endswith(".png")]
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
            
            # 添加通道维度并转换为tensor，格式为 (C, H, W)
            
            image_tensor = torch.from_numpy(image_np).to(device) / 255.0  # unsqueeze to add the channel dimension
           
            batch_images.append(image_tensor)
        
        # 将单个图像tensor组合为 (B, C, H, W)
        batch_tensor = torch.stack(batch_images)  # 合并batch
        
        yield (batch_files,batch_tensor)  # 返回batch tensor

# 示例用法
folder_path = '/research/d5/gds/zxwang22/storage/resist/cells/png/1nm'  # 替换为你的文件夹路径
batch_size = 4  # 你需要的batch大小
sigma = 0.05
pixelsize = 1

device = torch.device("cuda")

sim1 = AbbeSim(None, pixelsize, sigma, defocus=None, batch=True, par=False)
grad1 = AbbeGradient(None, pixelsize, sigma, defocus=None, batch=True, par=False)
intensity_np_all = []

for idx, batch in tqdm(enumerate(load_images_in_batches(folder_path, batch_size, device))):
    
    paths, masks = batch
    intensity = sim1(masks)
    intensity_np = intensity.cpu().numpy()
    intensity_np_all.append(intensity_np)
    for i in range(intensity_np.shape[0]):
    # 取出第 i 张图片，并去掉通道维度
        img_array = intensity_np[i] # 形状为 (H, W)
        img_array = ((img_array - img_array.min())/(img_array.max() - img_array.min())*255.0).astype(np.uint8)
        # 将数组转换为 PIL Image 对象
        img = Image.fromarray(img_array, mode='L')  # 'L' 表示灰度图
        
        img.save(os.path.join("/research/d5/gds/zxwang22/storage/resist/fuilt/lithoresult/1nm",paths[i]))
intensity_np_all = np.concatenate(intensity_np_all,0)
np.save(os.path.join("/research/d5/gds/zxwang22/storage/resist/fuilt/lithoresult/1nm","abbe_intensity.npy"),intensity_np_all)
