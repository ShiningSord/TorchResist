import torch
import torch.nn as nn
import fuILT
from fuILT.simulator.contest.utils import *
from torch.nn.functional import interpolate
from fuILT.algorithm.base import Sigmoid
class Kernel:
    def __init__(self, defocus=False, conjuncture=False, combo=False, device=DEVICE):
        self._basedir = f"{fuILT.__path__[0]}/simulator/contest/kernel"
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self): 
        return self._kernels
        
    @property
    def scales(self): 
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus" + filename
        else:
            filename = "focus" + filename
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        filename = self._basedir + "/kernels/" + filename + ".pt"
        return filename

    def _scale_file(self):
        filename = self._basedir + "/scales/"
        if self._combo:
            return filename + "combo.pt"
        else:
            if self._defocus:
                return filename + "defocus.pt"
            else:
                return filename + "focus.pt"

def _maskFloat(mask, dose):
    return (dose * mask).to(COMPLEXTYPE)

def _kernel_resize(kernel, maskFFT):
    result = torch.zeros_like(maskFFT, device=maskFFT.device, dtype=maskFFT.dtype)
    kernel_ifft = torch.fft.ifft2(_shift(kernel), norm="forward")
    start_x, start_y = (result.shape[0] - kernel_ifft.shape[0]) / 2, (result.shape[1] - kernel_ifft.shape[1]) / 2
    result[start_x:start_x + kernel_ifft.shape[0], start_y:start_y + kernel_ifft.shape[1]] += kernel_ifft
    result_fft = torch.fft.fft2(_shift(result), norm="forward")
    return result_fft

def _kernelMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device)
    if len(maskFFT.shape) == 3: 
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else: 
        assert len(maskFFT.shape) == 4, f"[_kernelMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:]  = maskFFT[:, :, :knxh+1, -knyh:]  * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1]  = maskFFT[:, :, -knxh:, :knyh+1]  * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:]   = maskFFT[:, :, -knxh:, -knyh:]   * kernel[None, :kernelNum, :knxh, :knyh]
    return output

def _shift(cmask):
    shifted = torch.zeros_like(cmask)
    if len(shifted.shape) == 3: 
        shifted[:, :cmask.shape[-2]//2, :cmask.shape[-1]//2] = cmask[:, cmask.shape[-2]//2:, cmask.shape[-1]//2:]  # 1 = 4
        shifted[:, :cmask.shape[-2]//2, cmask.shape[-1]//2:] = cmask[:, cmask.shape[-2]//2:, :cmask.shape[-1]//2]  # 2 = 3
        shifted[:, cmask.shape[-2]//2:, :cmask.shape[-1]//2] = cmask[:, :cmask.shape[-2]//2, cmask.shape[-1]//2:]  # 3 = 2
        shifted[:, cmask.shape[-2]//2:, cmask.shape[-1]//2:] = cmask[:, :cmask.shape[-2]//2, :cmask.shape[-1]//2]  # 4 = 1
    else: 
        assert len(shifted.shape) == 4
        shifted[:, :, :cmask.shape[-2]//2, :cmask.shape[-1]//2] = cmask[:, :, cmask.shape[-2]//2:, cmask.shape[-1]//2:]  # 1 = 4
        shifted[:, :, :cmask.shape[-2]//2, cmask.shape[-1]//2:] = cmask[:, :, cmask.shape[-2]//2:, :cmask.shape[-1]//2]  # 2 = 3
        shifted[:, :, cmask.shape[-2]//2:, :cmask.shape[-1]//2] = cmask[:, :, :cmask.shape[-2]//2, cmask.shape[-1]//2:]  # 3 = 2
        shifted[:, :, cmask.shape[-2]//2:, cmask.shape[-1]//2:] = cmask[:, :, :cmask.shape[-2]//2, :cmask.shape[-1]//2]  # 4 = 1
    return shifted

def _centerMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    imx, imy = maskFFT.shape[-2:]
    knx, kny = kernel.shape[-2:]
    imxh, imyh = imx // 2, imy // 2
    knxh, knyh = knx // 2, kny // 2
    xstart = imxh - knx // 2
    ystart = imyh - kny // 2
    xend =  xstart + knx
    yend = ystart + kny
    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device)
    if len(maskFFT.shape) == 3: 
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, xstart:xend, ystart:yend] = maskFFT[:, xstart:xend, ystart:yend] * kernel[:kernelNum, :, :]
    else: 
        assert len(maskFFT.shape) == 4, f"[_centerMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, xstart:xend, ystart:yend] = maskFFT[:, :, xstart:xend, ystart:yend] * kernel[None, :kernelNum, :, :]
    return output

def _computeImageMatrixLegacy(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    assert len(cmask.shape) in [3, 4], f"[_computeImageMatrixLegacy]: Invalid shape: {cmask.shape}"
    cmask_fft = _shift(torch.fft.fft2(_shift(cmask), norm="forward"))
    tmp = _centerMult(kernel, cmask_fft, kernelNum)
    tmp = _shift(torch.fft.ifft2(_shift(tmp), norm="forward"))
    return tmp

def _computeImageMaskLegacy(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    cmask_fft = _shift(torch.fft.fft2(_shift(cmask), norm="forward"))
    tmp = _centerMult(kernel, cmask_fft, kernelNum)
    tmp = _shift(torch.fft.ifft2(_shift(tmp), norm="forward"))
    return tmp

def _computeImageMatrix(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    assert len(cmask.shape) in [3, 4], f"[_computeImageMask]: Invalid shape: {cmask.shape}"
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp

def _computeImageMask(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp


def _convMatrix(cmask, dose, kernel, scale, kernelNum): 
    image = _computeImageMatrix(cmask, kernel, scale, kernelNum)
    return image

def _convMask(mask, dose, kernel, scale, kernelNum): 
    cmask = _maskFloat(mask, dose)
    image = _computeImageMask(cmask, kernel, scale, kernelNum)
    return image

class _LithoSim(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad): 
        ctx.saved = (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad)
        tmp = _convMask(mask, dose, kernel, scale, kernelNum)
        if len(mask.shape) == 2: 
            scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
            return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
        else: 
            assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
            scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)
    @staticmethod
    def backward(ctx, grad): 
        (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad) = ctx.saved
        cpx0 = torch.mul(_convMask(mask, dose, kernelGradCT, scaleGradCT, kernelNumGradCT), grad.unsqueeze(len(grad.shape) - 2))
        cpx1 = _convMatrix(cpx0, dose, kernelGrad, scaleGrad, kernelNumGrad)
        cpx2 = torch.mul(_convMask(mask, dose, kernelGrad, scaleGrad, kernelNumGrad), grad.unsqueeze(len(grad.shape) - 2))
        cpx3 = _convMatrix(cpx2, dose, kernelGradCT, scaleGradCT, kernelNumGradCT)
        cpx4 = cpx1 + cpx3
        if len(mask.shape) == 2: 
            scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
            cpx4 = torch.sum(scale * cpx4, dim=0)
        else: 
            assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
            scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            cpx4 = torch.sum(scale * cpx4, dim=1)

        return cpx4.real, None, None, None, None, None, None, None, None, None, None
class LithoSim(nn.Module): # Mask -> Aerial -> Printed
    inter_mode = ["nearest", "bilinear", "area"]
    def __init__(self, config = None): 
        super(LithoSim, self).__init__()
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        else: #default 
            self._config = parseConfig(f"{fuILT.__path__[0]}/simulator/contest/config/lithosimple.txt")
        required = ["KernelNum", "TargetDensity", "PrintThresh", 
                    "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in required: 
            assert key in self._config, f"[LithoSim]: Cannot find the config {key}."
        intfields = ["KernelNum", ]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
        # Read the kernels
        self._kernels = {"focus": Kernel(), 
                         "defocus": Kernel(defocus=True),
                         "CT focus": Kernel(conjuncture=True),
                         "CT defocus": Kernel(defocus=True, conjuncture=True),
                         "combo focus": Kernel(combo=True),
                         "combo defocus": Kernel(defocus=True, combo=True),
                         "combo CT focus": Kernel(conjuncture=True, combo=True),
                         "combo CT defocus": Kernel(defocus=True, conjuncture=True, combo=True)}
        
        self.use_interpolate = False
        self.interpolate = self.inter_mode[1]
        
    def setPixel(self, pixel):
        self.pixel = pixel
        return self
    
    def getPixel(self):
        pixel = getattr(self, "pixel", None)
        assert pixel != None
        return pixel
    
    def fullChipSimulation(self):
        self.use_interpolate = True
        return self
    
    def setInterolateMode(self, mode):
        self.interpolate = mode
        return self
        
    def __interpolate(self, image : torch.Tensor):
        target_shape = torch.Size([image.shape[0] * self.getPixel(), image.shape[1] * self.getPixel()])
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        return interpolate(image, size=target_shape, mode=self.interpolate).reshape(target_shape[0], target_shape[1])

    def forward(self, mask): 
        aerialNom = _LithoSim.apply(mask, self._config["DoseNom"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"], 
                                    self._kernels["CT focus"].kernels, self._kernels["CT focus"].scales, self._config["KernelNum"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"])
        
        if self.use_interpolate:
            aerialNom = self.__interpolate(aerialNom)
        printedNom = Sigmoid.apply(aerialNom, self._config["PrintSteepness"], self._config["TargetDensity"])
        
        aerialMax = _LithoSim.apply(mask, self._config["DoseMax"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"], 
                                    self._kernels["CT focus"].kernels, self._kernels["CT focus"].scales, self._config["KernelNum"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"])
        if self.use_interpolate:
            aerialMax = self.__interpolate(aerialMax)
        printedMax = Sigmoid.apply(aerialMax, self._config["PrintSteepness"], self._config["TargetDensity"])
        
        aerialMin = _LithoSim.apply(mask, self._config["DoseMin"], 
                                    self._kernels["defocus"].kernels, self._kernels["defocus"].scales, self._config["KernelNum"], 
                                    self._kernels["CT defocus"].kernels, self._kernels["CT defocus"].scales, self._config["KernelNum"], 
                                    self._kernels["defocus"].kernels, self._kernels["defocus"].scales, self._config["KernelNum"])
        if self.use_interpolate:
            aerialMin = self.__interpolate(aerialMin)
       
        printedMin = Sigmoid.apply(aerialMin, self._config["PrintSteepness"], self._config["TargetDensity"])
        
        # OpenILT implementation
        # printedNom = torch.sigmoid(self._config["PrintSteepness"] * (aerialNom - self._config["TargetDensity"]))
        # printedMax = torch.sigmoid(self._config["PrintSteepness"] * (aerialMax - self._config["TargetDensity"]))
        # printedMin = torch.sigmoid(self._config["PrintSteepness"] * (aerialMin - self._config["TargetDensity"]))
        
        return printedNom, printedMax, printedMin
    