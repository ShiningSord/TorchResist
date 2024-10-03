import torch
import numpy as np
from fuILT.algorithm.base import SimConfig

import pickle
from typing import Dict, List

class HopkinsTCC:
    
    def __init__(self,
                 config : SimConfig,
                 defocus : int,
                 device_list : List[torch.device],
                 VERBOSE : bool = False,
                 path : str = None) -> None:
        
        self.config : SimConfig = config
        
        self.defocus = defocus
        self.pixel = config.pixel
        self.NA = config.NA
        
        self.wavelength = config.wavelength
        assert config.bbox.getHeight() == config.bbox.getWidth()
        self.size = config.bbox.getWidth()
        self.convas = self.pixel * self.size
        
        self.device_list : List[torch.device] = device_list
        
        self.VERBOSE = VERBOSE
        
        self.path = path
        
        self.REFRACT = 1.44
        
         # TCC parameters
        self.phis : List[np.ndarray] = None
        self.weights : List[np.ndarray] = None
        
        self.device_tcc_map : Dict[torch.device, List[torch.Tensor]] = dict()
        
        
    def __gen_tcc(self) -> None:
        if self.weights != None:
            return
        from .tcc import srcPoint, funcPupil, genTCC
        
        pupil = funcPupil(self.pixel, 
                          self.convas, 
                          self.NA, 
                          self.wavelength, 
                          defocus=self.defocus, 
                          refract=self.REFRACT)
        circ = srcPoint(self.pixel, self.convas)
        phis, weights = genTCC(circ, pupil, self.pixel, self.convas)
        
        if self.VERBOSE:    
            print(f"Get {len(weights)} TCC: {weights}")
        
        self.phis, self.weights = phis, weights
        assert self.phis != None and self.weights != None 
        
        
    def gen_tcc_map(self) -> None:
        if self.phis == None or self.weights == None:
            self.__gen_tcc()
        
        for device in self.device_list:
            phis = []
            for phi in self.phis:
                phis.append(torch.tensor(phi, dtype=torch.complex64).to(device))
            weights = []
            for weight in self.weights:
                weights.append(torch.tensor(weight, dtype=torch.complex64).to(device))
            self.device_tcc_map[device] = (phis, weights)
            
    def dump(self):
        import os
        
        #TODO 加一个判断path是否存在的api
        
        assert self.path != None
        assert os.path.exists(self.path)
        
        with open(self.path, "wb") as fout: 
            pickle.dump((self.phis, self.weights), fout)
        fout.close()
