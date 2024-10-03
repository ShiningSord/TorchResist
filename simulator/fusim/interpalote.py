from torch.nn.functional import interpolate
import torch

def getInterpolateAerialImage(image : torch.Tensor, pixel, mode="bilinear"):
        target_shape = torch.Size([image.shape[0] * pixel, image.shape[1] * pixel])
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        return interpolate(image, size=target_shape, mode=mode).reshape(target_shape[0], target_shape[1])


def getInterpolateAerialImageBatch(image : torch.Tensor, pixel, mode="bilinear"):
        batch = image.shape[0]
        target_shape = torch.Size([image.shape[1] * pixel, image.shape[2] * pixel])
        image = image.reshape(batch, 1, image.shape[1], image.shape[2])
        return interpolate(image, size=target_shape, mode=mode).reshape(batch, target_shape[0], target_shape[1])

def getInterpolateAerialImageNCHW(image : torch.Tensor, pixel, mode="bilinear"):
        target_shape = torch.Size([image.shape[-2] * pixel, image.shape[-1] * pixel])
        return interpolate(image, size=target_shape, mode=mode)

def getInterpolateAerialImageCHW(image : torch.Tensor, pixel, mode="bilinear"):
        target_shape = torch.Size([image.shape[-2] * pixel, image.shape[-1] * pixel])
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        return interpolate(image, size=target_shape, mode=mode)