from fuILT.simulator.smit import Tcc
from fuILT.simulator.smit import SimulationConfigs

import torch
import time
import numpy as np

try:
    import aerialpy
except Exception as e:
    print("We do not find aerialpy package.")
    pass

DEBUG = False

class Simulator:
    def __init__(self, 
                 tcc : str, 
                 source : str, 
                 pixel : int, 
                 layer_path : str = None, 
                 layer : int = 1,
                 dose : int = 0.03) -> None:
        self.tcc = tcc
        self.source = source
        self.pixel = pixel
        self.layer_path = layer_path
        self.layer = layer

        self.allsettinggrad={}
        self.allsettinggrad["computemode"]="fast" #normal or fast

        self.allsettinga={}
        
        self.allsettinga["pixelsize"]=self.pixel
        self.allsettinga["computemode"]="fast"

        self.dose = {
            "normal" : 1,
            "inner" : 1 - dose,
            "outer" : 1 + dose
        }
        self.counter = 0

    def initial(self, chip_box = None, need = True):
        TCCSAVEPATH=self.tcc
        SOURCEFILE=self.source
        self.tcc = {}
        for mode in ["normal", "inner", "outer"]:
            print(f"Start to generate {mode} tcc, pixel = {self.pixel}")
            tcc = Tcc(mode=mode, 
                      pixel=self.pixel, 
                      TCCSAVEPATH=TCCSAVEPATH, 
                      SOURCEFILE=SOURCEFILE)
            if need:
                tcc.run()
            self.tcc[mode] = {
                "TCCVALUEPATH" : f"{TCCSAVEPATH}/{mode}_tcc/f{tcc.focus}p0_value_T.tcc",
                "TCCVECTORPATH" : f"{TCCSAVEPATH}/{mode}_tcc/f{tcc.focus}p0_vector_T.tcc"
            }
            print(f"Generate {mode} tcc success!")
        simulator = SimulationConfigs(LAYERPATH=self.layer_path, 
                                      layer=self.layer, 
                                      pixel = self.pixel)
        if chip_box == None:
            raise ValueError("Please set the chip box")
        else:
            simulator.setChipbox(chip_box)
        self.mask = simulator.getInitialOutput()
        self.shape = simulator.getShape(numpy=True)

        self.output = simulator.output

    def getShape(self):
        return self.shape

    @torch.no_grad()
    def simulate(self, mask, mode = "normal"):
        if isinstance(mask.data, torch.Tensor):
            if mask.is_cuda:
                mask_numpy = mask.data.cpu().numpy()
            else:
                mask_numpy = mask.data.numpy()
        else:
            mask_numpy = mask
        shape = mask_numpy.shape
        mask_numpy = mask_numpy.reshape(-1)
        assert mode in ["normal", "inner", "outer"]
        TCCVALUEPATH=self.tcc[mode]["TCCVALUEPATH"]
        TCCVECTORPATH=self.tcc[mode]["TCCVECTORPATH"]
        self.allsettinga["valuepath"]=TCCVALUEPATH
        self.allsettinga["vectorpath"]=TCCVECTORPATH

        # print("Start to compute aerial image.")
        # writeMaskPNG("/data/syin/yinshuo/projects/dac/figures/debug", mask.reshape(*shape))
        # np.savetxt("/data/syin/yinshuo/projects/dac/tests/grad.txt", mask_numpy)
        # print(f"[MASK] : {shape}")
        start_time = time.time()
        outai = aerialpy.computeaerial(self.allsettinga, mask_numpy * self.output[2] + self.output[3])
        end_time = time.time()
        if DEBUG:
            print(f"[FORWARD TIME] : {end_time - start_time}s")
        npai = np.array(outai) * self.dose[mode]
        return npai.reshape(shape)

    @torch.no_grad()
    def gradient(self, gradin):
        if isinstance(gradin, torch.Tensor):
            if gradin.is_cuda:
                gradin = gradin.cpu().numpy()
            else:
                gradin = gradin.numpy()
        shape = gradin.shape
        start_time = time.time()
        outgrad = aerialpy.computegradientai2mi(self.allsettinggrad, gradin, self.output[2]) # d aerialimage / d bitmap, list
        end_time = time.time()
        if DEBUG:
            print(f"[BACKWARD TIME] : {end_time - start_time}s")
        npgrad = np.array(outgrad)# 1-dim
        if DEBUG:
            if (True in np.isnan(npgrad)):
                raise ValueError("Sorry, simulator don't support such pixel size or Layout size")
        return npgrad.reshape(shape)
