from typing import Any
from .abbe import AbbeGradient, AbbeSim
import torch
from torch.autograd import Function
from fuILT.utils import BBox

class AbbeFunc(Function):
    @staticmethod
    def forward(ctx,
                mask,
                bbox : BBox, 
                pixel : int, 
                defocus=0, 
                sigma=0.05, 
                NA=1.35, 
                wavelength=193,
                par=False, 
                batch=False) -> Any:
        if defocus == "default": 
            defocus = [0, 30, 60]
        ctx.save_for_backward(mask)
        ctx.pixel, ctx.sigma, ctx.NA, ctx.wavelength, ctx.batch, ctx.defocus, ctx.par = pixel, sigma, NA, wavelength, batch, defocus, par
        if isinstance(defocus, tuple) or isinstance(defocus, list): 
            if not batch and len(mask.shape) == 3 and mask.shape[0] == 1: 
                mask = mask[0, :, :]
            elif batch and len(mask.shape) == 4 and mask.shape[1] == 1: 
                mask = mask[:, 0, :, :]
            sims = []
            for value in defocus: 
                sim = AbbeSim(bbox=None, pixel=pixel, sigma=sigma, NA=NA, wavelength=wavelength, defocus=value, batch=batch, par=par)
                sims.append(sim)
            results = []
            for sim in sims: 
                simmed = sim(mask)
                results.append(simmed[:, None, ...] if batch else simmed[None, ...])
            return torch.cat(results, dim=1) if batch else torch.cat(results, dim=0)
        else: 
            sim = AbbeSim(bbox=None, pixel=pixel, sigma=sigma, NA=NA, wavelength=wavelength, defocus=defocus, batch=batch, par=par)
            return sim(mask)

    
    @staticmethod
    def vjp(ctx, grad_outputs : torch.Tensor) -> Any:
        pixel, sigma, NA, wavelength, batch, defocus, par = ctx.pixel, ctx.sigma, ctx.NA, ctx.wavelength, ctx.batch, ctx.defocus, ctx.par
        mask, = ctx.saved_tensors
        if isinstance(defocus, tuple) or isinstance(defocus, list): 
            origin = mask
            if not batch and len(mask.shape) == 3 and mask.shape[0] == 1: 
                mask = mask[0, :, :]
            elif batch and len(mask.shape) == 4 and mask.shape[1] == 1: 
                mask = mask[:, 0, :, :]
            
            grads = []
            for value in defocus: 
                grad = AbbeGradient(bbox=None, pixel=pixel, sigma=sigma, NA=NA, wavelength=wavelength, defocus=value, batch=batch, par=par)
                grads.append(grad)
            results = []
            for grad in grads: 
                graded = grad(mask)
                results.append(graded[:, None, ...] if batch else graded[None, ...])
            concated = torch.cat(results, dim=1) if batch else torch.cat(results, dim=0)
            gradient = concated * grad_outputs
            if batch: 
                gradient = gradient.mean(dim=1)
            else: 
                gradient = gradient.mean(dim=0)
            if not batch and len(origin.shape) == 3 and origin.shape[0] == 1: 
                gradient = gradient[None, :, :]
            elif batch and len(origin.shape) == 4 and origin.shape[1] == 1: 
                gradient = gradient[:, None, :, :]
            
            return gradient, None, None, None, None, None, None, None, None
        else: 
            grad = AbbeGradient(bbox=None, pixel=pixel, sigma=sigma, NA=NA, wavelength=wavelength, defocus=defocus, batch=batch, par=par)
    
        return grad(mask) * grad_outputs, None, None, None, None, None, None, None, None