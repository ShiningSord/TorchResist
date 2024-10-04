import time
import torch
import numpy as np
from matplotlib import pyplot as plt

from simulator import AbbeSim, AbbeGradient
from utils import BBox, Point

sigma = 0.05
pixelsize = 14
canvas = 256*14
size = round(canvas/pixelsize)
device = torch.device("cuda")

sim1 = AbbeSim(None, pixelsize, sigma, defocus=None, batch=True, par=False)
grad1 = AbbeGradient(None, pixelsize, sigma, defocus=None, batch=True, par=False)

batch = 10
nb = np.zeros([batch, size, size])
nb[:, 16:48, 16:48] = 1
b = torch.tensor(nb)


image1 = sim1(b)
image3 = grad1(b)

