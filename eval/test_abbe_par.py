import time
import torch
import numpy as np
from matplotlib import pyplot as plt

from simulator import AbbeSim, AbbeGradient
from utils import BBox, Point

sigma = 0.05
pixelsize = 8
canvas = 512 * 4
size = round(canvas/pixelsize)
device = torch.device("cuda")

sim1 = AbbeSim(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=False, par=False)
sim2 = AbbeSim(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=False, par=True)
grad1 = AbbeGradient(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=False, par=False)
grad2 = AbbeGradient(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=False, par=True)

nb = np.zeros([size, size])
nb[31:33, 16:200] = 1
nb[44:46, 16:200] = 1
b = torch.tensor(nb)

timeSim1 = time.time()
image1 = sim1(b)
timeSim1 = time.time() - timeSim1
timeSim2 = time.time()
image2 = sim2(b)
timeSim2 = time.time() - timeSim2
print(f"Time 1: {timeSim1:.3e}s -> {timeSim2:.3e}s, speedup={timeSim1/timeSim2:.3f}x")
print(f" -> Difference 1: {torch.nn.functional.mse_loss(image1, image2)}")
timeSim1 = time.time()
image3 = grad1(b)
timeSim1 = time.time() - timeSim1
timeSim2 = time.time()
image4 = grad2(b)
timeSim2 = time.time() - timeSim2
print(f"Time 2: {timeSim1:.3e}s -> {timeSim2:.3e}s, speedup={timeSim1/timeSim2:.3f}x")
print(f" -> Difference 2: {torch.nn.functional.mse_loss(image3, image4)}")

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(image1.numpy())
plt.subplot(2, 2, 2)
plt.imshow(image2.numpy())
plt.subplot(2, 2, 3)
plt.imshow(image3.numpy())
plt.subplot(2, 2, 4)
plt.imshow(image4.numpy())
plt.savefig("test_abbe_1.png")


sim1 = AbbeSim(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=True, par=False)
sim2 = AbbeSim(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=True, par=True)
grad1 = AbbeGradient(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=True, par=False)
grad2 = AbbeGradient(BBox(Point(0, 0), Point(512, 512)), pixelsize, sigma, defocus=None, batch=True, par=True)

batch = 10
nb = np.zeros([batch, size, size])
nb[:, 16:48, 16:48] = 1
b = torch.tensor(nb)

timeSim1 = time.time()
image1 = sim1(b)
timeSim1 = time.time() - timeSim1
timeSim2 = time.time()
image2 = sim2(b)
timeSim2 = time.time() - timeSim2
print(f"Time 3: {timeSim1:.3e}s -> {timeSim2:.3e}s, speedup={timeSim1/timeSim2:.3f}x")
print(f" -> Difference 3: {torch.nn.functional.mse_loss(image1, image2)}")
timeSim1 = time.time()
image3 = grad1(b)
timeSim1 = time.time() - timeSim1
timeSim2 = time.time()
image4 = grad2(b)
timeSim2 = time.time() - timeSim2
print(f"Time 4: {timeSim1:.3e}s -> {timeSim2:.3e}s, speedup={timeSim1/timeSim2:.3f}x")
print(f" -> Difference 4: {torch.nn.functional.mse_loss(image3, image4)}")

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(image1[0].numpy())
plt.subplot(2, 2, 2)
plt.imshow(image2[1].numpy())
plt.subplot(2, 2, 3)
plt.imshow(image3[-1].numpy())
plt.subplot(2, 2, 4)
plt.imshow(image4[-2].numpy())

plt.savefig("test_abbe_2.png")


'''
NOTE: 

1. Set par=True to enable source points parallelization. 

2. Parallelizing source points seems not so effective. It only works in batch=False mode. 

3. Although the construction of pupil is not parallelized, it is not the bottleneck. Constructing the pupil consumes less than 15% running time. 
'''
