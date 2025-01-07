import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from tqdm import tqdm
from scipy.integrate import cumtrapz
torch.autograd.set_detect_anomaly(True)

np.set_printoptions(threshold=np.inf)


class ResistSimulator(torch.nn.Module):
    def __init__(self, alpha= 0.0005, dill_a=0.00075, dill_b=0.00005, dill_c=0.0025, lamp_power=30000, dose=2000, n_steps=50, m_th=0.01, r_min=0.8, r_max=10, developed_time=15, thickness=1000, nz=50, threshold=300, save_dir = "output") -> None:
        super().__init__()
        self.alpha= torch.nn.parameter.Parameter(torch.tensor(alpha),requires_grad=False)
        self.dill_a= torch.nn.parameter.Parameter(torch.tensor(dill_a),requires_grad=False)
        self.dill_b= torch.nn.parameter.Parameter(torch.tensor(dill_b),requires_grad=False)
        self.dill_c= torch.nn.parameter.Parameter(torch.tensor(dill_c),requires_grad=False)
        # Typical lamp power in W/m²
        self.lamp_power= torch.nn.parameter.Parameter(torch.tensor(lamp_power),requires_grad=False)
        # Dose in J/m²
        self.dose=torch.nn.parameter.Parameter(torch.tensor(dose),requires_grad=False)
        self.n_steps=n_steps
        
        # parameter for Mack_Developement_Rate 
        self.m_th=torch.nn.parameter.Parameter(torch.tensor(m_th),requires_grad=False)
        self.r_min=torch.nn.parameter.Parameter(torch.tensor(r_min),requires_grad=False)
        self.r_max=torch.nn.parameter.Parameter(torch.tensor(r_max),requires_grad=False)
        
        
        self.developed_time = torch.nn.parameter.Parameter(torch.tensor(developed_time),requires_grad=False)
        self.threshold = torch.nn.parameter.Parameter(torch.tensor(threshold),requires_grad=False)
        self.thickness = thickness
        self.nz = nz
        
        self.device='cuda'if torch.cuda.is_available() else "cpu"
        
        self.save_dir = save_dir
        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def forward(self, aerial_image, dx = 1.0):
        assert len(aerial_image.shape) == 3, f"we only accept batch input, current input has a shape of {aerial_image.shape}"
        # Create a meshgrid corresponding to the resist coordinates in x,y and z direction
        
        x_support = torch.tensor(list(range(aerial_image.shape[1])), device=self.device) * dx # dx in nm/pixel
        y_support = torch.tensor(list(range(aerial_image.shape[2])), device=self.device) * dx # dx in nm/pixel
        dz=self.thickness/self.nz
        z=torch.from_numpy(np.linspace(0,self.thickness,self.nz)).float().to(self.device)
    
        X,Y,Z = torch.meshgrid((x_support, y_support, z), indexing="ij")
        # Instanciate bulk image, the aerial image is stacked with itself self.nz times.
        aerial_image=torch.stack([aerial_image for _ in range(self.nz)],-1)
        bulk_ini=aerial_image.clone().detach()
        # Apply beer Lambert absorption
        bulk_img=bulk_ini*torch.exp(-self.alpha*Z[None])
    
        
        # Initialise latent image
        lat_img=torch.ones_like(bulk_img)
        # Exposure time in s
        t_tot=self.dose/self.lamp_power
        time_step=t_tot/self.n_steps
        
        # Loop to compute exposition
        for _ in range(self.n_steps):
            # Latent image update
            lat_img*=torch.exp(-self.dill_c*bulk_img*time_step*self.lamp_power)
            # Absorption coefficient update
            alpha=self.dill_a*lat_img+self.dill_b
            # Bulk image update
            bulk_img=bulk_ini*torch.exp(-alpha*Z[None])

        # Computation of the development rate with typical parameters
        cur_dev_rate=self.Mack_Developement_Rate(latent_img=lat_img, m_th=self.m_th, r_min=self.r_min, r_max=self.r_max, n=5)
        # cur_dev_rate.shape = H, W, D
        # inverse
        # time_resist_z = cumtrapz(1. / cur_dev_rate.clone().detach().numpy(), dx=dz, axis=2,initial=0)
        # resist_result_1 = self.get_max_depth_below_threshold(time_resist_z, threshold=self.developed_time.clone().detach().numpy()) * dz
        # resist_result_numpy = resist_result.numpy()
        # normal 
        resist_result = torch.zeros_like(cur_dev_rate[:,:,:,0])
        resist_time = torch.ones_like(resist_result) * self.developed_time
        
        for depth in range(cur_dev_rate.shape[-1]):
            cur_depth = cur_dev_rate[:,:,:,depth] * resist_time
            mask_less_eq_dz = (cur_depth <= dz)
            mask_greater_dz = (cur_depth > dz)
            
            # 防止就地修改
            resist_time = resist_time.clone()
            resist_result = resist_result.clone()
            
            resist_time[mask_less_eq_dz] = 0
            resist_result = resist_result + cur_depth * mask_less_eq_dz.float()
            
            resist_time[mask_greater_dz] = resist_time[mask_greater_dz] - (dz / cur_dev_rate[:,:,:,depth])[mask_greater_dz]
            resist_result = resist_result + dz * mask_greater_dz.float()
            
            if (resist_time <= 0).all():
                break
        
      
        return resist_result
        
    def get_max_depth_below_threshold(self, A, threshold):
        # 获取A数组的形状
        X, Y, Z = A.shape
        
        # 初始化结果B为一个大小为(X, Y)的二维数组，初始值为-1（表示没有找到小于等于阈值的深度）
        B = torch.zeros((X, Y), dtype=int)
        
        # 遍历Z方向，找出每个XY平面坐标上的最大深度
        for z in range(Z):
            # 对于A[:,:,z]平面，找到满足条件的位置并记录对应的深度z
            mask = A[:,:,z] <= threshold
            B[mask] = z+1
        
        return B

        
    # This function computes the developement rate according to the 4 parameters model from Mack.
    def Mack_Developement_Rate(self, latent_img, m_th, r_min, r_max, n):
        a_mack = (1 - m_th) ** n
        a_mack *= (n + 1) / (n - 1)
        dev_rate = (a_mack + 1) * (1 - latent_img) ** n
        dev_rate /= a_mack + (1 - latent_img) ** n
        dev_rate *= r_max
        dev_rate += r_min
        dev_rate = torch.clip(dev_rate, r_min, r_max)
        return dev_rate


def get_default_simulator():
    alpha = 0.006186  # 6.186/um
    dill_a = 0
    dill_b = alpha - dill_a
    dill_c = 0.001  # need further calibration
    lamp_power = 30000.0 # need further calibration
    dose = 2000.0 # need further calibration
    n_steps = 50
    m_th = 0.61 # need further calibration
    r_min = 0.1 # need further calibration
    r_max = 15.0 # need further calibration
    developed_time = 10.0 # need further calibration
    threshold = 25.0 # need further calibration
    thickness = 75 
    nz = 75
    
    params = locals()  # 获取所有局部变量的字典
    simulator =  ResistSimulator(**params)
    simulator.dill_c.requires_grad_(True)
    simulator.lamp_power.requires_grad_(True)
    simulator.dose.requires_grad_(True)
    simulator.m_th.requires_grad_(True)
    simulator.r_min.requires_grad_(True)
    simulator.r_max.requires_grad_(True)
    simulator.developed_time.requires_grad_(True)
    simulator.threshold.requires_grad_(True)
    return simulator


def get_fuilt_simulator():
    alpha = 0.006186  # 6.186/um
    dill_a = 0
    dill_b = alpha - dill_a
    dill_c = 0.001  
    lamp_power = 30000.0 
    dose = 2003.27124
    n_steps = 50
    m_th = 0.668257773 
    r_min = 0.73013 
    r_max = 9.759724 
    developed_time = 7.68653 
    threshold = 8.55224
    thickness = 75
    nz = 75
    params = locals() 
    simulator =  ResistSimulator(**params)
    return simulator

def get_iccad13_simulator():
    alpha = 0.006186  # 6.186/um
    dill_a = 0
    dill_b = alpha - dill_a
    dill_c = 0.001  
    lamp_power = 30000.0 
    dose = 1999.04821777
    n_steps = 50
    m_th = 0.32625094
    r_min = 0.0
    r_max = 17.271150588989258
    developed_time = 12.224553108215332
    threshold = 32.845272064208984
    thickness = 75
    nz = 75
    params = locals() 
    simulator =  ResistSimulator(**params)
    return simulator

def get_hg_simulator():
    alpha = 0.006186  # 6.186/um
    dill_a = 0
    dill_b = alpha - dill_a
    dill_c = 0.001  
    lamp_power = 30000.0 
    dose = 1999.963989
    n_steps = 50
    m_th = 0.73410082
    r_min = -0.00081108051
    r_max = 16.35210609436
    developed_time = 11.245305061340332 
    threshold = 27.381071090698242
    thickness = 75
    nz = 75
    params = locals() 
    simulator =  ResistSimulator(**params)
    return simulator




if __name__ == "__main__":
    simulator = get_default_simulator().cuda()
    aerial_image = Image.open('/research/d5/gds/zxwang22/code/TorchLitho/aerial.png')
    aerial_image = torch.from_numpy(np.array(aerial_image))/255.0
    aerial_image = torch.cat([aerial_image.T[None],aerial_image[None]],0).to(simulator.device)
    # import pdb; pdb.set_trace()
    res = simulator.forward(aerial_image, dx=1.0)
        
        
