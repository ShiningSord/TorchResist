import torch
from torch import cuda

threshold = 8 * 1024

def _getMemoryMB(memory):
    return memory // 1024 // 1024

class DeviceHandler:
    def __init__(self, 
                 is_master_a_worker : bool=False,
                 num_worker : int = None) -> None:
        self.has_cuda = cuda.is_available()
        if not self.has_cuda:
            return 
        
        self.is_master_a_worker = is_master_a_worker
        self.device_num = cuda.device_count()
        
        import os 
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.device_num)])
        
        self.device_pool = [torch.device(f"cuda:{i}") for i in range(self.device_num)]
        self.properties = []
        self.__check_memory()
        
        self.master_index = self.__find_master()
        self.workes_index_list = self.__find_workers()
        
        self.num_worker : int = num_worker
        
    def __check_memory(self):
        self.mask = [True for _ in range(self.device_num)]
        for idx, device in enumerate(self.device_pool):
            property = cuda.get_device_properties(device)
            self.properties.append(property)
            total_memory = _getMemoryMB(property.total_memory)
            if total_memory < threshold:
                self.mask[idx] = False
    
    def __find_master(self):
        max_idx, max_memory = -1, 0
        for idx, (m, p) in enumerate(list(zip(self.mask, self.properties))):
            if m:
                if p.total_memory > max_memory:
                    max_idx = idx

        return max_idx
                
    def __find_workers(self):
        workers = []
        for idx, mask in enumerate(self.mask):
            if mask:
                if not self.is_master_a_worker:
                    if idx != self.master_index:
                        workers.append(idx)
                else:
                    workers.append(idx)
        return workers
    
    def getNumWorkers(self):
        if self.num_worker is not None:
            return self.num_worker
        return len(self.workes_index_list)
    
    def getMaster(self):
        return self.device_pool[self.master_index]
    
    def getMasterRank(self):
        return self.master_index
    
    def getWorker(self, idx):
        assert idx >= 0 and idx < self.getNumWorkers()
        return self.device_pool[self.workes_index_list[idx]]
    
    def getWorkerRank(self, idx):
        assert idx >= 0 and idx < self.getNumWorkers()
        return self.workes_index_list[idx]

    def hasCUDA(self):
        return self.has_cuda
    
    def getCPUDevice(self):
        return torch.device("cpu")
    

    def getSize(self):
        count = 0
        for item in self.mask:
            if item:
                count += 1
        return count
            
            
        
            
        
        