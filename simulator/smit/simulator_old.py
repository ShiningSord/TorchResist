from ast import arg
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import math
import os
try:
    import tccpy
except Exception as e:
    print("We do not find tccpy package.")
    pass

try:
    import aerialpy
except Exception as e:
    print("We do not find aerialpy package.")
    pass

try:
    import maskpy
except Exception as e:
    print("We do not find maskpy package.")
    pass

try:
    import sourcepy
except Exception as e:
    print("We do not find sourcepy package.")
    pass

import torch
from math import ceil
from typing import List
from imageTools import writeMaskPNG, writePVBandPNG, writeMaskPNGNoT, writeMaskSVG
from configs import ConfigLitho, ConfigILT
import argparse
from torch.nn.functional import interpolate
import time
import pickle

DEBUG = False


        # sleep(3)

class Simulator:
    def __init__(self, tcc : str, source : str, pixel : int, layer_path : str = None, layer : int = 1) -> None:
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
            "inner" : 0.97,
            "outer" : 1.03
        }
        self.counter = 0

    def initial(self, chip_box = None, need = True):
        TCCSAVEPATH=self.tcc
        SOURCEFILE=self.source
        self.tcc = {}
        for mode in ["normal", "inner", "outer"]:
            print(f"Start to generate {mode} tcc, pixel = {self.pixel}")
            tcc = TccConfig(mode=mode, pixel=self.pixel, TCCSAVEPATH=TCCSAVEPATH, SOURCEFILE=SOURCEFILE)
            if need:
                tcc.run()
            self.tcc[mode] = {
                "TCCVALUEPATH" : f"{TCCSAVEPATH}/{mode}_tcc/f{tcc.focus}p0_value_T.tcc",
                "TCCVECTORPATH" : f"{TCCSAVEPATH}/{mode}_tcc/f{tcc.focus}p0_vector_T.tcc"
            }
            print(f"Generate {mode} tcc success!")
        simulator = SimulationConfigs(LAYERPATH=self.layer_path, layer=self.layer, pixel = self.pixel)
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


def sigmoid(input : torch.Tensor, steepness, target_intensity):
    return 1 / (1 + torch.exp(-steepness * (input - target_intensity)))

def computeSize(chip_box):
    h, w = chip_box[2] - chip_box[0], chip_box[3] - chip_box[1]
    return max(h, w)

def findTileShape(size, initial = 4):
    while True:
        if size % initial == 0:
            return initial
        else:
            initial += 1

def getSize(x1, y1, x2, y2, shape):
    def bound(x):
        if x < 0:
            x = 0
        elif x > shape:
            x = shape
        else:
            x = x
        return x
            
    x1, y1, x2, y2 = x1-1, y1-1, x2+1, y2+1
    return [bound(x1), bound(y1), bound(x2), bound(y2)]
    
MODE = ["normal", "inner", "outer"]
inter_mode = ["nearest", "bilinear", "area", "bicubic"]
RESULT_NORMAL_PATH = "/data/syin/yinshuo/projects/dac/results/normal"
RESULT_INNER_PATH = "/data/syin/yinshuo/projects/dac/results/inner"
RESULT_OUTTER_PATH = "/data/syin/yinshuo/projects/dac/results/outter"
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Levelset based opc simulator.")
    parser.add_argument("--gds", nargs='?', type=str, default="AND2_X1.gds", const="AND2_X1.gds", 
    help="Gds file name (Don't forget .gds suffix), default = \'AND2_X1\'")
    parser.add_argument("--pixel", "-p", type=int, nargs='?', default=28, const=28, 
    help="pixel setting, default = 7")
    parser.add_argument("--layer", "-l", nargs='?', type=int, default=1, const=1,
    help="layer setting, the layer number in the origin gds, \
        you can choose the metal layer or via layer, default = 1")
    parser.add_argument("--tcc", action="store_true", help="need regenerate tcc, default = False")
    parser.add_argument("--test", action="store_true", help="store the image in the test files, for debug mode, default = False")
    parser.add_argument("--box", "-b",  nargs=4, help="chip box setting, please input four integers, \
    [x1, y1, x2, y2], (x1, y1) represent low left point, (x2, y2) represent up right point", required=True)
    parser.add_argument("--mask", "-m",  action="store_true", help="use the input mask, if false, we will generate mask using smit simulator, default = False")
    parser.add_argument("--size", "-s",  nargs=1, default=None, const=None, help="mask size, np.Size(size, size)")
    parser.add_argument("--mode", '-i', nargs="?", type=int, default=1, const=1, help="\
    interpolate mode, can be \{0, 1, 2 ,3\}, default = 1")

    parser.add_argument("--pvband", nargs="?", type=int, default=0, const=0, help="\
    simulation corner, [0, 1, 2], default = 0")
    args = parser.parse_args()

    print(args)

    try:
        assert Target.pixel_ == 1
    except Exception as e:
        raise ValueError(f"Sorry, pixel size need to be 1nm, however we got {Target.pixel_}nm")

    gds = args.gds
    layer = args.layer
    pixel = args.pixel
    need_tcc = args.tcc
    mode = args.mode
    chip_box = list(map(lambda x : int(x), args.box))
    test = args.test
    if args.size != None:
        shape = int(args.size[0])
    else:
        shape = None
    use_mask = args.mask
    assert int(args.pvband) in [0, 1, 2]
    pvband = MODE[int(args.pvband)]
    
    TCCSAVEPATH=f"/data/syin/yinshuo/projects/dac/configs/tcc/tcc_{pixel}"
    SOURCEFILE="/data/syin/yinshuo/projects/dac/configs"
    simulator = Simulator(TCCSAVEPATH, SOURCEFILE, pixel, layer_path = f"/data/syin/yinshuo/projects/dac/GDSfile/{gds}", layer = layer)
    simulator.initial(chip_box=chip_box, need=need_tcc)

    configILT = ConfigILT(chip_box[2] - chip_box[0], chip_box[3] - chip_box[1])

    configLitho = ConfigLitho()

    if use_mask:
        
        with open(f"/data/syin/yinshuo/projects/dac/mask/final_mask.pkl", 'rb') as file:
            mask = pickle.load(file)
        file.close()
    else:
        mask = simulator.output[1]

    writeMaskPNG(f"/data/syin/yinshuo/projects/dac/results/final_mask", mask)

    if use_mask:
        print(f"[MASK SHAPE] : {mask.shape}, [SIMULATOR REGION] : {simulator.getShape()}, [CORNER] : {pvband}")
        image = simulator.simulate(mask = mask, mode = pvband)
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        # target = np.loadtxt("/data/syin/yinshuo/projects/dac/target/target.txt")
        # targetShape = torch.Size(list(target.shape))
        image = image[1:image.shape[0] - 1, 1:image.shape[1] - 1]
        targetShape = torch.Size([computeSize(chip_box), computeSize(chip_box)])
        assert computeSize(chip_box) == image.shape[0] * pixel
        image = image.reshape(1, 1, image.shape[0], image.shape[1])

        image = interpolate(image, size=targetShape, mode=inter_mode[mode]).reshape(targetShape)
        image = sigmoid(image, configILT.resistSigmoid, configLitho.targetIntensity)
        image = image.numpy()
        # gradient = simulator.gradient(np.ones_like(mask))
    else:
        raise ValueError(f"We must need mask now")
        image = simulator.simulate(mask = mask.reshape(shape, shape), mode = "normal")
        image = sigmoid(image, configILT.resistSigmoid, configLitho.targetIntensity)
        # gradient = simulator.gradient(np.ones_like(mask.reshape(shape, shape)))

    print(f"[IMAGE SHAPE] : {image.shape}")
    target_num = 8
    if not test:
        assert (image.shape[0]) % target_num == 0
        size_xy = (image.shape[0]) // target_num
        for i in range(target_num):
            start_x = i * size_xy
            for j in range(target_num):
                start_y = j * size_xy
                clip = image[start_x:start_x + size_xy, start_y:start_y+size_xy]
                clip = np.pad(clip, ((1,1),(1,1)))
                print(f"[INFO] : Start to write clip [{i + 1}][{j + 1}]")
                start_time = time.time()
                if pvband == 'normal':

                    with open(f"{RESULT_NORMAL_PATH}/normal_{i}_{j}.pkl", 'wb') as file:
                        pickle.dump(clip, file)
                    file.close()

                    writeMaskPNG(f"/data/syin/yinshuo/projects/dac/results/final_image_{i}_{j}", clip)
                elif pvband == 'inner':
                    with open(f"{RESULT_INNER_PATH}/inner_{i}_{j}.pkl", 'wb') as file:
                        pickle.dump(clip, file)
                    file.close()
                else:
                    with open(f"{RESULT_OUTTER_PATH}/outter_{i}_{j}.pkl", 'wb') as file:
                        pickle.dump(clip, file)
                    file.close()
                
                end_time = time.time()
                print(f"[INFO] : write clip [{i + 1}][{j + 1}] finish, [TIME] : {end_time - start_time}")
    else:
        writeMaskPNG(f"/data/syin/yinshuo/projects/dac/tests/final_image", image)
    print("[FINISH] : simulation finish")
    
