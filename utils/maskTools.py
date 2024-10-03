import torch
from typing import List, Tuple, Union
from fuILT.utils import BBox, Point

def setBBoxRegionToConstant(image : torch.Tensor, bbox : List[int], c = 1):
    ones = torch.ones((bbox[2] - bbox[0], bbox[3] - bbox[1])) * c
    if image.is_cuda:
        ones.to(torch.device("cuda"))
    image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = ones
    
def getScaledBBoxTensor(total_bbox : torch.Tensor, pixel, padding, is_cuda):
    from fuILT._C import bbox_reduction
    if is_cuda:
        assert total_bbox.is_cuda == True
        bb : torch.Tensor = bbox_reduction.bbox_reduction(total_bbox).cpu()
    else:
        assert total_bbox.is_cuda == False
        bb : torch.Tensor = bbox_reduction.bbox_reduction_cpu(total_bbox)
    assert bb.shape[0] == 4 and len(bb.shape) == 1
    w, h = abs(bb[0].item() - bb[2].item()), abs(bb[1].item() - bb[3].item())
    target_len = (max(w // pixel, h // pixel) + padding) * pixel
    if (target_len / pixel) % 2 == 1:
        target_len = target_len + pixel
        
    ll_shift_x = (target_len - w) // 2
    ur_shift_x = (target_len - w) - ll_shift_x

    ll_shift_y = (target_len - h) // 2
    ur_shift_y = (target_len - h) - ll_shift_y

    bb[0] = bb[0] - ll_shift_x
    bb[1] = bb[1] - ll_shift_y
    bb[2] = bb[2] + ur_shift_x
    bb[3] = bb[3] + ur_shift_y

    return bb

def getScaledBBoxTensorWithLevel(total_bbox : torch.Tensor, pixel, padding, is_cuda, level=2):
    from fuILT._C import bbox_reduction
    if total_bbox.dim() == 1 and total_bbox.shape[0] == 4:
        bb = total_bbox
    else:
        if is_cuda:
            assert total_bbox.is_cuda == True
            bb : torch.Tensor = bbox_reduction.bbox_reduction(total_bbox).cpu()
        else:
            assert total_bbox.is_cuda == False
            bb : torch.Tensor = bbox_reduction.bbox_reduction_cpu(total_bbox)
    assert bb.shape[0] == 4 and len(bb.shape) == 1
    w, h = abs(bb[0].item() - bb[2].item()), abs(bb[1].item() - bb[3].item())
    target_len = (max(w // pixel, h // pixel) + padding) * pixel
    
    mod = (target_len / pixel) % (2**level)
    if mod != 0:
        mod = 2**level - mod
    
    target_len = target_len + mod * pixel     
    
    ll_shift_x = (((target_len - w) / pixel) // 2) * pixel
    ur_shift_x = (target_len - w) - ll_shift_x

    ll_shift_y = (((target_len - h) / pixel) // 2) * pixel
    ur_shift_y = (target_len - h) - ll_shift_y

    bb[0] = bb[0] - ll_shift_x
    bb[1] = bb[1] - ll_shift_y
    bb[2] = bb[2] + ur_shift_x
    bb[3] = bb[3] + ur_shift_y
    
    return bb

#TODO
def getScaledPaddingSize(size : torch.Size, # mask shape
                         pixel : int,
                         marco_size : List[int], #TODO
                         padding : Union[int, Tuple[int]]):
    bb = torch.tensor([0, 0, size[0], size[1]])
    assert len(list(size)) == 2 and bb.shape[0] == 4 and len(bb.shape) == 1
    mask_w, mask_h = size[0], size[1]
    w, h = mask_w * pixel, mask_h * pixel
    target_len = (max(w // pixel, h // pixel) + padding * 2) * pixel
    while (target_len / pixel) % (2 * marco_size[0]) != 0:
        target_len = target_len + pixel
        
    ll_shift_x = (target_len - w) // pixel // 2
    ur_shift_x = (target_len - w) // pixel - ll_shift_x

    ll_shift_y = (target_len - h) // pixel // 2
    ur_shift_y = (target_len - h) // pixel - ll_shift_y

    bb[0] = bb[0] - ll_shift_x
    bb[1] = bb[1] - ll_shift_y
    bb[2] = bb[2] + ur_shift_x
    bb[3] = bb[3] + ur_shift_y
    bb = BBox(Point(0, 0), Point((bb[2] - bb[0]).item(), (bb[3] - bb[1]).item()))
    assert ll_shift_y == ur_shift_x and  ur_shift_y == ll_shift_y and ur_shift_x == ur_shift_y

    return bb, (ll_shift_y, ur_shift_x, ur_shift_y, ll_shift_y), abs(ll_shift_x)