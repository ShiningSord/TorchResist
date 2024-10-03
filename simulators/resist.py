import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from tqdm import tqdm
from scipy.integrate import cumtrapz

np.set_printoptions(threshold=np.inf)


class ResistSimulator(torch.nn.Module):
    def __init__(self, alpha= 0.0005, dill_a=0.00075, dill_b=0.00005, dill_c=0.0025, lamp_power=30000, dose=2000, n_steps=50, m_th=0.01, r_min=0.8, r_max=10, developed_time=15, thickness=1000, nz=50, save_dir = "output") -> None:
        super().__init__()
        self.alpha= alpha
        self.dill_a=dill_a
        self.dill_b=dill_b
        self.dill_c=dill_c
        # Typical lamp power in W/m²
        self.lamp_power=lamp_power
        # Dose in J/m²
        self.dose=dose
        self.n_steps=n_steps
        
        # parameter for Mack_Developement_Rate 
        self.m_th=m_th
        self.r_min=r_min
        self.r_max=r_max
        
        
        self.developed_time = developed_time
        self.thickness = thickness
        self.nz = nz
        
        self.save_dir = save_dir
        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def forward(self, aerial_image, dx = 1.0):
        # Create a meshgrid corresponding to the resist coordinates in x,y and z direction
        x_support = torch.tensor(list(range(aerial_image.shape[0]))) * dx # dx in nm/pixel
        y_support = torch.tensor(list(range(aerial_image.shape[1]))) * dx # dx in nm/pixel
        dz=self.thickness/self.nz
        z=torch.from_numpy(np.linspace(0,self.thickness,self.nz)).float()
    
        X,Y,Z = torch.meshgrid((x_support, y_support, z), indexing="xy")
        
        # Instanciate bulk image, the aerial image is stacked with itself self.nz times.
        aerial_image=torch.stack([aerial_image for _ in range(self.nz)],-1)
        bulk_ini=aerial_image.clone().detach()
        # Apply beer Lambert absorption
        bulk_img=bulk_ini*torch.exp(-self.alpha*Z)
    
        # Plotting section
        if False:
            for i in range(self.nz):
                # Extract the i-th layer of the bulk image
                bulk_slice = bulk_img[:,:,i].numpy()
                
                # Plot and save the bulk image for the current depth
                plt.figure(figsize=(6, 6))
                plt.imshow(bulk_slice, cmap='gray', extent=[0, x_support[-1], 0, y_support[-1]])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.title(f'Bulk Image at Depth {i}, Z = {z[i]:.1f} nm')
                save_path = os.path.join(self.save_dir, f"initail_bulk_image_depth_{i}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        
        # Initialise latent image
        lat_img=torch.ones_like(bulk_img)
        # Exposure time in s
        t_tot=self.dose/self.lamp_power
        time_step=t_tot/self.n_steps
        
        # Loop to compute exposition
        for _ in tqdm(range(self.n_steps)):
            # Latent image update
            lat_img*=torch.exp(-self.dill_c*bulk_img*time_step*self.lamp_power)
            # Absorption coefficient update
            alpha=self.dill_a*lat_img+self.dill_b
            # Bulk image update
            bulk_img=bulk_ini*torch.exp(-alpha*Z)
            
        # Plotting section
        if False:
            for i in range(self.nz):
                # Extract the i-th layer of the bulk image
                bulk_slice = bulk_img[:,:,i].numpy()
                
                # Plot and save the bulk image for the current depth
                plt.figure(figsize=(6, 6))
                plt.imshow(bulk_slice, cmap='gray', extent=[0, x_support[-1], 0, y_support[-1]])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.title(f'Bulk Image at Depth {i}, Z = {z[i]:.1f} nm')
                save_path = os.path.join(self.save_dir, f"exposition_bulk_image_depth_{i}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        # Computation of the development rate with typical parameters
        cur_dev_rate=self.Mack_Developement_Rate(latent_img=lat_img, m_th=self.m_th, r_min=self.r_min, r_max=self.r_max, n=2)
        # cur_dev_rate.shape = H, W, D
        time_resist_z = cumtrapz(1. / cur_dev_rate, dx=dz, axis=2,initial=0)
        np.set_printoptions(precision=3, threshold=1000, linewidth=150)
        # import pdb; pdb.set_trace()
        for th in range(60):
        
            resist_result = self.get_max_depth_below_threshold(time_resist_z, threshold=th)
            
            # Plotting section
            resist_result_numpy = resist_result.numpy()
  
            # Plot and save the bulk image for the current depth
            plt.figure(figsize=(6, 6))
            plt.imshow(resist_result_numpy, cmap='gray', extent=[0, x_support[-1], 0, y_support[-1]])
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.title(f'Resist Image Dev Time {th}')
            save_path = os.path.join(self.save_dir, f"resist_t_dev{th}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        
        return resist_result
        
    def get_max_depth_below_threshold(self, A, threshold):
        # 获取A数组的形状
        X, Y, Z = A.shape
        
        # 初始化结果B为一个大小为(X, Y)的二维数组，初始值为-1（表示没有找到小于等于阈值的深度）
        B = -torch.ones((X, Y), dtype=int)
        
        # 遍历Z方向，找出每个XY平面坐标上的最大深度
        for z in range(Z):
            # 对于A[:,:,z]平面，找到满足条件的位置并记录对应的深度z
            mask = A[:,:,z] <= threshold
            B[mask] = z
        
        return B

        
    # This function computes the developement rate according to the 4 parameters model from Mack.
    def Mack_Developement_Rate(sefl, latent_img, m_th, r_min, r_max, n):
        a_mack = (1 - m_th) ** n
        a_mack *= (n + 1) / (n - 1)
        dev_rate = (a_mack + 1) * (1 - latent_img) ** n
        dev_rate /= a_mack + (1 - latent_img) ** n
        dev_rate *= r_max
        dev_rate += r_min
        dev_rate = np.clip(dev_rate, r_min, r_max)
        return dev_rate


def get_default_simulator():
    alpha = 0.006186  # 6.186/um
    dill_a = 0
    dill_b = alpha - dill_a
    dill_c = 0.001  # need further calibration
    lamp_power = 30000 # need further calibration
    dose = 2000 # need further calibration
    n_steps = 50
    m_th = 0.01 # need further calibration
    r_min = 0.8 # need further calibration
    r_max = 10 # need further calibration
    developed_time = 15 # need further calibration
    thickness = 75
    nz = 75
    
    params = locals()  # 获取所有局部变量的字典
    return ResistSimulator(**params)


if __name__ == "__main__":
    simulator = get_default_simulator()
    
    aerial_image = Image.open('/research/d5/gds/zxwang22/code/TorchLitho/aerial.png')
    aerial_image = torch.from_numpy(np.array(aerial_image))/255.0
    import pdb; pdb.set_trace()
    res = simulator.forward(aerial_image, dx=1.0)
        
        

