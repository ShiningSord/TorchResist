import sys
sys.path.append(".")
import math
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func


EPE_CONSTRAINT = 15
EPE_CHECK_INTERVEL = 40
MIN_EPE_CHECK_LENGTH = 80
EPE_CHECK_START_INTERVEL = 40

def boundaries(target):
    boundary   = torch.zeros_like(target)
    corner     = torch.zeros_like(target)
    vertical   = torch.zeros_like(target)
    horizontal = torch.zeros_like(target)

    padded = func.pad(target[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    upperleft  = padded[2:,  :-2] == 1
    upperright = padded[2:,  2:]  == 1
    lowerleft  = padded[:-2, :-2] == 1
    lowerright = padded[:-2, 2:]  == 1
    boundary = (target == 1)
    boundary[upper & lower & left & right & upperleft & upperright & lowerleft & lowerright] = False
    
    padded = func.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    center = padded[1:-1, 1:-1]  == 1

    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort((vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy()))
    vsites = vsites[vindices]
    vstart = torch.cat((torch.tensor([True], device=vsites.device), vsites[:, 0][1:] != vsites[:, 0][:-1] + 1))
    vend   = torch.cat((vsites[:, 0][1:] != vsites[:, 0][:-1] + 1, torch.tensor([True], device=vsites.device)))
    vstart = vsites[(vstart == True).nonzero()[:, 0], :]
    vend   = vsites[(vend   == True).nonzero()[:, 0], :]
    vposes = torch.stack((vstart, vend), axis=2)
    
    horizontal = center.clone()
    horizontal[upper & lower] = False
    hsites = horizontal.nonzero()
    hindices = np.lexsort((hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy()))
    hsites = hsites[hindices]
    hstart = torch.cat((torch.tensor([True], device=hsites.device), hsites[:, 1][1:] != hsites[:, 1][:-1] + 1))
    hend   = torch.cat((hsites[:, 1][1:] != hsites[:, 1][:-1] + 1, torch.tensor([True], device=hsites.device)))
    hstart = hsites[(hstart == True).nonzero()[:, 0], :]
    hend   = hsites[(hend   == True).nonzero()[:, 0], :]
    hposes = torch.stack((hstart, hend), axis=2)

    return vposes.float(), hposes.float()



def check(image, sample, target, direction):
    if direction == 'v':
        if ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0)): #left ,x small
            inner = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1)): #right, x large
            inner = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    if direction == 'h':
        if((target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0)): #up, y small
            inner = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1): #low, y large
            inner = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    return inner, outer



def _epecheck(mask, target, vposes, hposes):
    '''
    input: binary image tensor: (b, c, x, y); vertical points pair vposes: (N_v,4,2); horizontal points pair: (N_h, 4, 2), target image (b, c, x, y)
    output the total number of epe violations
    '''
    inner = 0
    outer = 0
    epeMap = torch.zeros_like(target)
    vioMap = torch.zeros_like(target)

    for idx in range(vposes.shape[0]):
        center = (vposes[idx, :, 0] + vposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0) #(1, 2)
        if (vposes[idx, 0, 1] - vposes[idx, 0, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        else:
            sampleY = torch.cat((torch.arange(vposes[idx, 0, 0] + EPE_CHECK_START_INTERVEL, center[0, 0] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(vposes[idx, 0, 1] - EPE_CHECK_START_INTERVEL, center[0, 0],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = vposes[idx, :, 0].repeat(sampleY.shape[0], 1)
            sample[:, 0] = sampleY
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1

    for idx in range(hposes.shape[0]):
        center = (hposes[idx, :, 0] + hposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if (hposes[idx, 1, 1] - hposes[idx, 1, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        else: 
            sampleX = torch.cat((torch.arange(hposes[idx, 1, 0] + EPE_CHECK_START_INTERVEL, center[0, 1] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(hposes[idx, 1, 1] - EPE_CHECK_START_INTERVEL, center[0, 1],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = hposes[idx, :, 0].repeat(sampleX.shape[0], 1)
            sample[:, 1] = sampleX
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    return inner, outer, vioMap


def epecheck(prediction, target):
    vposes, hposes = boundaries(target)
    epeIn, epeOut, _ =  epecheck(prediction, target, vposes, hposes)
    return epeIn, epeOut
 


