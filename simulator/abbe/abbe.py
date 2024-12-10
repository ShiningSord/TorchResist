from typing import Any
import torch
from torch import fft
from ..source.source import getSourcePoints
from ..source.source import getFreqCut
from ..source.source import getFreqSupport
from ..source.source import getGridSize
from ..source.source import getDeltaFreq
from ..source.source import getDefocus
from ..utils import getMaskFFT
from utils import BBox, Point


class AbbeSim:
    def __init__(self, 
                 bbox : BBox, 
                 pixel : int, 
                 sigma=0.05, 
                 NA=1.35, 
                 wavelength=193,
                 batch=False, 
                 defocus=0, 
                 par=False
                 ) -> None:
        self.sigma = sigma
        self.NA = NA
        self.wavelength = wavelength
        self.pixel = pixel
        
        self.batch = batch

        self.defocus = defocus
        
        self.par = par
    
    def __compute_source(self):
        # Define the effective source points of the circular source
        return getSourcePoints(self.freq, getFreqCut(self.sigma, self.NA, self.wavelength))
        
    def __compute_freq(self, bbox, device):
        return getFreqSupport(getGridSize(bbox, pixel=self.pixel), self.pixel, device)
    
    def __compute_abbe(self, mask_fft):
        bbox = BBox(Point(0, 0), Point(mask_fft.shape[0] * self.pixel, mask_fft.shape[1] * self.pixel))
        self.freq = self.__compute_freq(bbox, mask_fft.device)
        self.source_points = self.__compute_source()
        
        # Initialisation of the aerial image as single precision
        aerial_image = torch.zeros_like(mask_fft, dtype=torch.float)
        # Compute all source points contribution to the aerial image
        if self.par: 
            pupils = []
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = self.freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                pupils.append(pupil_shifted[None, ...])
            pupils = torch.cat(pupils, dim=0)
            mask_lpfs = mask_fft[None, ...] * pupils
            aerials = torch.pow(torch.abs(fft.ifft2(fft.ifftshift(mask_lpfs))), 2)
            aerial_image = aerials.mean(dim=0)
        else: 
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = self.freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                # The shifted transfer function of the projection lens is applied to the mask spectrum
                
                mask_lpf = torch.multiply(mask_fft, pupil_shifted)
            
                # Add the contribution of the current src point to the overall aerial image
                aerial_image += torch.pow(torch.abs(fft.ifft2(fft.ifftshift(mask_lpf))), 2)
            # Normalisation with number of source points
            aerial_image /= self.source_points.shape[0]
        return aerial_image
    
    def __compute_abbe_batch(self, mask_fft):
        batch_size = mask_fft.shape[0]
        mask_shape = [mask_fft.shape[1], mask_fft.shape[2]]
        bbox = BBox(Point(0, 0), Point(mask_shape[0] * self.pixel, mask_shape[1] * self.pixel))
        self.freq = self.__compute_freq(bbox, mask_fft.device)
        self.source_points = self.__compute_source()
        
        # Initialisation of the aerial image as single precision
        aerial_image = torch.zeros_like(mask_fft, dtype=torch.float)
        # Compute all source points contribution to the aerial image
        freq = self.freq.expand(batch_size, *mask_shape)

        # Compute all source points contribution to the aerial image
        if self.par: 
            pupils = []
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                pupils.append(pupil_shifted[:, None, ...])
            pupils = torch.cat(pupils, dim=1)
            mask_lpfs = mask_fft[:, None, ...] * pupils
            aerials = torch.pow(torch.abs(fft.ifft2(fft.ifftshift(mask_lpfs))), 2)
            aerial_image = aerials.mean(dim=1)

        else: 
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                # The shifted transfer function of the projection lens is applied to the mask spectrum
                
                mask_lpf = torch.multiply(mask_fft, pupil_shifted)
            
                # Add the contribution of the current src point to the overall aerial image
                aerial_image += torch.pow(torch.abs(fft.ifft2(fft.ifftshift(mask_lpf))), 2)
            # Normalisation with number of source points
            aerial_image /= self.source_points.shape[0]
        return aerial_image
    
    def __call__(self, mask) -> torch.Tensor:
        mask_fft = getMaskFFT(mask)
        if not self.batch:
            return self.__compute_abbe(mask_fft)
        return self.__compute_abbe_batch(mask_fft)
    

class AbbeGradient:
    def __init__(self, 
                 bbox : BBox, 
                 pixel : int, 
                 sigma=0.05, 
                 NA=1.35, 
                 wavelength=193,
                 batch=False, 
                 defocus=0, 
                 par=False
                 ) -> None:
        
        self.sigma = sigma
        self.NA = NA
        self.wavelength = wavelength
        self.pixel = pixel
        
        self.batch = batch

        self.defocus = defocus

        self.par = par
    
    def __compute_source(self):
        # Define the effective source points of the circular source
        return getSourcePoints(self.freq, getFreqCut(self.sigma, self.NA, self.wavelength))
        
    def __compute_freq(self, bbox : BBox, device):
        return getFreqSupport(getGridSize(bbox, pixel=self.pixel), self.pixel, device) 
            
    def __compute_gradient(self, mask_fft):
        
        self.freq = self.__compute_freq(BBox(Point(0, 0), Point(mask_fft.shape[0] * self.pixel, mask_fft.shape[1] * self.pixel)), mask_fft.device)
        self.source_points = self.__compute_source()
        
        gradient = torch.zeros_like(mask_fft, dtype=torch.float)

        if self.par: 
            pupils = []
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = self.freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                pupils.append(pupil_shifted[None, ...])
            pupils = torch.cat(pupils, dim=0)
            mask_lpfs = mask_fft[None, ...] * pupils
            
            ta = fft.ifft2(fft.ifftshift(mask_lpfs))
            tmpgradient = 2*ta
            tmpgradient = fft.fftshift(fft.fft2(tmpgradient))
            tmpgradient = tmpgradient*torch.conj(pupils)
            tmpgradient = fft.ifft2(fft.ifftshift(tmpgradient))
            
            gradient = tmpgradient.real.mean(dim=0)

        else: 
            for freq_src in self.source_points:
                tmpgradient = torch.zeros_like(mask_fft, dtype=mask_fft.dtype)

                freq_msk_shft = self.freq - freq_src
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                mask_lpf = torch.multiply(mask_fft, pupil_shifted)
                ta = fft.ifft2(fft.ifftshift(mask_lpf))
                tmpgradient = 2*ta
                tmpgradient = fft.fftshift(fft.fft2(tmpgradient))
                tmpgradient = tmpgradient*torch.conj(pupil_shifted)
                tmpgradient = fft.ifft2(fft.ifftshift(tmpgradient))
                
                gradient += tmpgradient.real
        
            gradient = gradient / self.source_points.shape[0]
        return gradient
    
    
    def __compute_gradient_batch(self, mask_fft):
        batch_size = mask_fft.shape[0]
        mask_shape = [mask_fft.shape[1], mask_fft.shape[2]]
        
        self.freq = self.__compute_freq(BBox(Point(0, 0), Point(mask_shape[0] * self.pixel, mask_shape[1] * self.pixel)), mask_fft.device)
        self.source_points = self.__compute_source()
        
        gradient = torch.zeros_like(mask_fft, dtype=torch.float)
        
        freq = self.freq.expand(batch_size, *mask_shape)

        if self.par: 
            pupils = []
            for freq_src in self.source_points:
                # Shift of the frequency support relative to the current source point frequency
                freq_msk_shft = freq - freq_src
                # Shifted transfer function of the projection lens.
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                pupils.append(pupil_shifted[:, None, ...])
            pupils = torch.cat(pupils, dim=1)
            mask_lpfs = mask_fft[:, None, ...] * pupils
            
            ta = fft.ifft2(fft.ifftshift(mask_lpfs))
            tmpgradient = 2*ta
            tmpgradient = fft.fftshift(fft.fft2(tmpgradient))
            tmpgradient = tmpgradient*torch.conj(pupils)
            tmpgradient = fft.ifft2(fft.ifftshift(tmpgradient))
            
            gradient = tmpgradient.real.mean(dim=1)
            
        else: 
            for freq_src in self.source_points:
                tmpgradient = torch.zeros_like(mask_fft, dtype=mask_fft.dtype)

                freq_msk_shft = freq - freq_src
                pupil_shifted = torch.where(freq_msk_shft < getDeltaFreq(self.NA, self.wavelength), 
                                            torch.ones_like(freq_msk_shft, device=freq_msk_shft.device), 
                                            torch.zeros_like(freq_msk_shft, device=freq_msk_shft.device))
                pupil_shifted = getDefocus(pupil_shifted, freq_msk_shft, self.wavelength, self.defocus)
                mask_lpf = torch.multiply(mask_fft, pupil_shifted)
                ta = fft.ifft2(fft.ifftshift(mask_lpf))
                tmpgradient = 2*ta
                tmpgradient = fft.fftshift(fft.fft2(tmpgradient))
                tmpgradient = tmpgradient*torch.conj(pupil_shifted)
                tmpgradient = fft.ifft2(fft.ifftshift(tmpgradient))
                
                gradient += tmpgradient.real
            
            gradient = gradient / self.source_points.shape[0]
        return gradient
    
    
    def __call__(self, mask) -> torch.Tensor:
        mask_fft = getMaskFFT(mask)
        if not self.batch:
            return self.__compute_gradient(mask_fft)
        return self.__compute_gradient_batch(mask_fft)
            
            
            
