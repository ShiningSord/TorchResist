import torch
from torch import fft
import pickle

def getMaskFFT(mask):
    return fft.fftshift(fft.fft2(mask))

def getK(wavelength):
    return 2 * torch.pi / wavelength

def readTccParaFromDisc(path : str):
    with open(path, "rb") as fin:
        phis, weights = pickle.load(fin)
    return phis, weights