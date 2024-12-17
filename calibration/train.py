import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F

from simulator import get_default_simulator



target_threshold = 0.5

def prepare_data(input_data, target_data, test_size=0.2, batch_size=32):
    """
    将数据集划分为训练集和测试集，并返回DataLoader
    :param input_data: 输入数据，numpy数组 [N, H, W]
    :param target_data: 目标数据，numpy数组 [N, H, W]
    :param test_size: 测试集比例
    :param batch_size: 每个batch的大小
    :return: 训练和测试的DataLoader
    """
    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=test_size, random_state=42)

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 创建Dataset和DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    进行一个epoch的训练
    :param model: 要训练的模型
    :param dataloader: 训练数据的DataLoader
    :param optimizer: 优化器
    :param criterion: 损失函数（L2 Loss）
    :param device: 设备 (CPU 或 GPU)
    :return: 平均训练损失
    """
    scale = 6.0
    model.train()
    total_loss = 0.0
    total_iter = len(dataloader)
    for iter, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs, dx = 7.0)
        outputs = (outputs - model.threshold)/model.thickness
        scale_outputs = torch.zeros_like(outputs)
        mask = outputs < 0
        scale_outputs[mask] =  outputs[mask] * scale * model.thickness / model.threshold
        scale_outputs[~mask] = outputs[~mask] * scale * model.thickness /(model.thickness- model.threshold)

        loss = criterion(scale_outputs, (targets > target_threshold).float())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        model.r_min.data = torch.clamp(model.r_min.data, 0.0)
        if iter % 10 == 0:
            print(f"{iter}/{total_iter}, Loss:{loss.item():.2f}")
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    评估模型在测试集上的表现
    :param model: 训练好的模型
    :param dataloader: 测试数据的DataLoader
    :param criterion: 损失函数
    :param device: 设备 (CPU 或 GPU)
    :return: 平均测试损失
    """
    model.eval()
    scale = 6.0
    total_diff = 0.0
    total_loss = 0.0
    
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            H, W = targets.shape[-2:]
            upsampled_targets = F.interpolate(targets.unsqueeze(0).clone().detach(), size=(int(7*H), int(7*W)), mode='bilinear', align_corners=False).squeeze(0)

            upsampled_targets =  (upsampled_targets > target_threshold).float()
            # 前向传播   
            outputs = model(inputs, dx = 7.0)
            outputs_clone = outputs.clone().detach()
            scale_outputs = torch.zeros_like(outputs)
            outputs = (outputs - model.threshold)/model.thickness
            mask = outputs < 0
            scale_outputs[mask] =  outputs[mask] * scale * model.thickness / model.threshold
            scale_outputs[~mask] = outputs[~mask] * scale * model.thickness /(model.thickness- model.threshold)

            loss = criterion(scale_outputs, (targets > target_threshold).float())

            total_loss += loss.item()
            
            
            
            upsampled_outputs = F.interpolate(outputs_clone.unsqueeze(0), size=(int(7*H), int(7*W)), mode='bilinear', align_corners=False).squeeze(0)
            upsampled_outputs = (upsampled_outputs > model.threshold).float()
            diff = torch.abs(upsampled_targets - upsampled_outputs).mean()
            total_diff += diff.item()
    avg_diff = total_diff / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    return avg_diff,avg_loss


import torch

def train_model(model, train_loader, test_loader, num_epochs, initial_lr, print_every, device):
    """
    主训练循环，进行多个epoch的训练和评估
    :param model: 要训练的模型
    :param train_loader: 训练集的DataLoader
    :param test_loader: 测试集的DataLoader
    :param num_epochs: 训练的epoch数量
    :param initial_lr: 初始学习率
    :param print_every: 打印损失的间隔
    :param device: 设备 (CPU 或 GPU)
    """
    # 定义损失函数和Adam优化器
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # 设置学习率调度器，step_size表示每隔多少个epoch调整一次学习率，gamma是学习率的调整因子
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    
    # 将模型放到指定设备上
    model.to(device)
    
    # 初始化用于保存最优模型的变量
    best_test_diff, best_test_loss = evaluate(model, test_loader, criterion, device)
    print(f"best_test_diff {best_test_diff}, best_test_loss {best_test_loss}")
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 每隔print_every个epoch打印一次损失
        if (epoch + 1) % print_every == 0:
            model.eval()  # 设置模型为评估模式
            test_diff, test_loss = evaluate(model, test_loader, criterion, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {train_loss:.8f}, '
                  f'Test Loss: {test_loss:.8f}, '
                  f'Test Diff: {test_diff:.8f}, '
                  f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            # 保存测试损失最小的模型
            if test_diff < best_test_diff:
                best_test_diff = test_diff
                best_model_state = model.state_dict()
                print(f'New best model found at epoch {epoch+1}, saving model...')
                
    
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data}")
        
        # 学习率调度器步进
        scheduler.step()
    
    # 打印最终的模型参数
    
    
    # 如果找到最优模型，保存它
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')
        print('Best model saved with test diff:', best_test_diff)




def main(input_data, target_data, simulator, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 数据集划分
    train_loader, test_loader = prepare_data(input_data, target_data, test_size=0.2, batch_size=16)
    
    # 开始训练
    num_epochs = 9
    learning_rate = 1e-2
    print_every = 1 # 每10个epoch打印一次损失
    
    train_model(simulator, train_loader, test_loader, num_epochs, learning_rate, print_every, device)

# 你可以在主程序中调用main()，传入数据集和模型


if __name__ == "__main__":
    inputs = np.random.random([1000,100,100])
    inputs = inputs / (inputs.max() + 1e-6)
    targets = np.random.random([1000,100,100]).astype(np.float)
    simulator = get_default_simulator()
    simulator.dill_c.requires_grad_(False)
    main(inputs, targets, simulator)