import numpy as np
from math import sqrt

try:
    import maskpy
except Exception as e:
    print("We do not find maskpy package.")
    pass

import torch
from typing import List
from fuILT.simulator.smit import SimulationConfigs
    
class RastedMask():
    unit_ = "nm"
    def __init__(self,
                 LAYERPATH : str = None, 
                 chipbox : List[int] = [], 
                 layer : int = 1, 
                 pixel : int = 28) -> None:
        self.config = SimulationConfigs(LAYERPATH, layer, pixel)
        self.mask = None
        self.__setChipbox(chipbox=chipbox)
        
    def __setChipbox(self, chipbox : List[int]):
        assert len(chipbox) == 4
        self.config.all_setting["chipbox"] = chipbox
        
    def __getShape(self, numpy = False):
        size_ = int(sqrt(self.target.shape[0]))
        if numpy:
            return (size_, size_)
        return torch.Size((size_, size_))

    def __getInitialOutput(self):
        self.output = maskpy.computemask(self.config.all_setting)  
        self.target = np.zeros(self.output[1].shape)
        self.target[self.output[1] > 0.5 ] = 1
        self.target[self.output[1] <= 0.5 ] = 0
        return self.target

    def getMask(self):
        if self.mask == None:
            print(f"Start to generate mask~~~~")
            self.mask = self.__getInitialOutput()
        return self.mask

    def __str__(self):
        s = f"[MASK SHAPE] : {self.__getShape()}, [MASK] : {self.mask is None}"
        return s