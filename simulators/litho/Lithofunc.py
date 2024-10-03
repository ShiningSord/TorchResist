
from typing import Any
import torch
from .functions.Calculate2DAerialImage import *
import torch.nn.functional as F

class LithoFunc:        
    @staticmethod
    def apply(mask, defocus=[-60, 0, 60], batch=False) -> Any:
        sr = Source(device=mask.device)
        po = ProjectionObjective()
        numerics = Numerics()
        aerail_litho_image = ImageData()

        defocuses = [0, -defocus, defocus]
        X, Y = mask.shape[-2], mask.shape[-1]
        numerics.SampleNumber_Mask_X, numerics.SampleNumber_Mask_Y = X, Y
        numerics.SampleNumber_Wafer_X, numerics.SampleNumber_Wafer_Y = X, Y

        if not batch: 
            # mask = F.interpolate(mask[None, :, :, :], size=[81, 81], mode='bilinear')[0]
            results = torch.zeros([len(defocuses)] + list(mask.shape[-2:]), 

                                  dtype=mask.dtype, device=mask.device)
            mk = Mask(device=mask.device)
            mk.Feature = mask[0] if len(mask.shape) == 3 else mask
            mk.MaskType = "2dpixel"
            for jdx, defocus in enumerate(defocus): 
                rp = Receipe()
                rp.FocusRange = torch.tensor([defocus])
                result = Calculate2DAerialImage(sr, mk, po, aerail_litho_image, rp, numerics)
                results[jdx] = result.Intensity[0].T
            # results = F.interpolate(results[None, :, :, :], size=[X, Y], mode='bilinear')[0]
            return results
        else: # Note: It seems that Calculate2DAerialImage does not support batch simulation, due to the design of the Mask
            reshaped = mask[:, 0] if len(mask.shape) == 4 else mask
            results = torch.zeros(list(mask.shape[:1]) + [len(defocus)] + list(mask.shape[-2:]), 
                                  dtype=mask.dtype, device=mask.device)
            for idx in range(mask.shape[0]): 
                mk = Mask(device=mask.device)
                mk.Feature = reshaped[idx]
                mk.MaskType = "2dpixel"
                for jdx, defocus in enumerate(defocus): 
                    rp = Receipe()
                    rp.FocusRange = torch.tensor([defocus])
                    result = Calculate2DAerialImage(sr, mk, po, aerail_litho_image, rp, numerics)
                    results[idx, jdx] = result.Intensity[0].T
            return results
