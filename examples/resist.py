from simulator import get_default_simulator
import numpy as np
import torch
from PIL import Image



simulator = get_default_simulator()
    
aerial_image = Image.open('examples/aerial.png')
aerial_image = torch.from_numpy(np.array(aerial_image))/255.0
# import pdb; pdb.set_trace()
res = simulator.forward(aerial_image, dx=7.0)