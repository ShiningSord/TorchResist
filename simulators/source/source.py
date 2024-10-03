import torch
from fuILT.utils import BBox
from torch import fft
from typing import List
import numpy as np


def getGridSize(bbox : BBox, pixel : int):
    check = lambda x, y : x % y == 0
    width, height = bbox.getWidth(), bbox.getHeight()
    assert check(width, pixel) and check(height, pixel), f"{width} {height} {pixel}"
    return [width / pixel, height/ pixel]

def getFreqSupport(grid : List[int], pixel : int, device):
    assert grid[0] == grid[1]
    freq_x = fft.fftshift(fft.fftfreq(int(grid[0]), d=pixel, device=device))
    freq_y = fft.fftshift(fft.fftfreq(int(grid[1]), d=pixel, device=device))
    freq_x = freq_x.reshape(-1, 1).expand(int(grid[0]), int(grid[1]))
    freq_x = freq_x * torch.conj(freq_x)
    freq_y = freq_y.reshape(1, -1).expand(int(grid[0]), int(grid[1]))
    freq_y = freq_y * torch.conj(freq_y)
    return torch.sqrt(freq_x + freq_y)

def getFreqCut(sigma, NA, wavelength):
    return sigma * NA / wavelength

# @torch.compile
def getSourcePoints(freq : torch.Tensor, freqshift):
    zeros = torch.zeros_like(freq, device=freq.device)
    mask = torch.ones_like(freq, device=freq.device)
    source = torch.where(freq <= abs(freqshift), freq, zeros)
    mask = torch.where(freq <= abs(freqshift), mask, zeros).to(torch.bool)
    return source.reshape(-1)[mask.reshape(-1)]  

def getSourcePointsMask(freq : torch.Tensor, freqshift):
    zeros = torch.zeros_like(freq, device=freq.device)
    ones = torch.ones_like(freq, device=freq.device)
    return torch.where(freq < torch.abs(freqshift), ones, zeros)

def getDeltaFreq(NA, wavelength):
    return NA / wavelength

def getDefocus(pupil, freqs, wavelength, defocus): 
    if defocus is None: 
        return pupil
    mask = pupil > 0
    tmp = wavelength**2 * (freqs*mask)**2
    opd = defocus * (1.44 - torch.sqrt(1.44**2 - wavelength**2 * (freqs*mask)**2))
    shift = torch.exp(1j * (2 * torch.pi / wavelength) * opd)
    return pupil * shift
