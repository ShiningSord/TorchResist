import pickle
import torch
import torch.nn.functional as F
import numpy as np
from fuILT.utils import BBox, Point
from ..utils import getMaskFFT, readTccParaFromDisc
import matplotlib.pyplot as plt
from typing import List, Union

'''
Note: you may use simulator/hopkins/tcc.py to generate the TCC file
'''

class HopkinsSim:
    def __init__(self, 
                 tcc : List[torch.Tensor] = None,
                 defocus : bool = False, 
                 device : torch.device = None,
                 filename : str = None) -> None:
        assert device is not None
        self.device = device
        self.defocus = defocus
        
        if tcc is not None:
            self.phis, self.weights = tcc
        else:
            assert filename is not None
            phis, weights = readTccParaFromDisc(filename)
            self.phis = [torch.tensor(phi, dtype=torch.complex64).to(device) for phi in phis]
            self.weights = [torch.tensor(weight, dtype=torch.float32).to(device) for weight in weights]
        
    
    def __compute_hopkins(self, maskFFT, phis, weights): # already batched
        aerial = torch.zeros(maskFFT.shape, dtype=torch.float32, device=maskFFT.device)
        for idx, phi in enumerate(phis): 
            weight = weights[idx]
            # phi = torch.tensor(phi, dtype=torch.complex64).to(maskFFT.device)
            real = F.interpolate(phi.real[None, None, :, :], size=maskFFT.shape[-2:], mode="bilinear")[0, 0]
            imag = F.interpolate(phi.imag[None, None, :, :], size=maskFFT.shape[-2:], mode="bilinear")[0, 0]
            phi = torch.zeros(maskFFT.shape[-2:], dtype=torch.complex64).to(maskFFT.device)
            phi.real = real
            phi.imag = imag
            phiFFT = torch.fft.fft2(phi)
            if len(phiFFT.shape) < len(maskFFT.shape): 
                phiFFT = phiFFT[None, :, :]
            conved = torch.fft.ifft2(maskFFT * phiFFT)
            conved = torch.fft.fftshift(conved, dim=[-2, -1]) / np.prod(conved.shape)
            aerial += (weight * conved * conved.conj()).real
        return aerial   
    
    def __call__(self, mask) -> torch.Tensor:
        assert mask.device == self.device, f"{mask.device} != {self.device}"
        mask_fft = torch.fft.fft2(mask)
        if self.defocus: 
            assert len(self.phis) == len(self.weights)
            results = []
            for idx in range(len(self.phis)): 
                phis = self.phis[idx]
                weights = self.weights[idx]
                result = self.__compute_hopkins(mask_fft, phis, weights)
                if len(mask.shape) == 2: 
                    result = result[None, ...]
                if len(mask.shape) == 3: 
                    result = result[:, None, ...]
                results.append(result)
            if len(mask.shape) == 2: 
                results = torch.cat(results, dim=0)
            elif len(mask.shape) >= 3: 
                results = torch.cat(results, dim=1)
            return results
        else: 
            return self.__compute_hopkins(mask_fft, self.phis, self.weights)
    
    