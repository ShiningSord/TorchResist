import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from simulator import get_default_simulator



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
    model.train()
    total_loss = 0.0
    total_iter = len(dataloader)
    for iter, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        outputs = (outputs - model.threshold)/model.thickness
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
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
    total_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            outputs = (outputs - model.threshold)/model.thickness
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, print_every, device):
    """
    主训练循环，进行多个epoch的训练和评估
    :param model: 要训练的模型
    :param train_loader: 训练集的DataLoader
    :param test_loader: 测试集的DataLoader
    :param num_epochs: 训练的epoch数量
    :param learning_rate: 学习率
    :param print_every: 打印损失的间隔
    :param device: 设备 (CPU 或 GPU)
    """
    # 定义损失函数和优化器
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 将模型放到指定设备上
    model.to(device)
    
    for epoch in range(num_epochs):
        # 训练模型
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 每隔print_every个epoch打印一次损失
        if (epoch + 1) % print_every == 0:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}')


def main(input_data, target_data, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 数据集划分
    train_loader, test_loader = prepare_data(input_data, target_data, test_size=0.2, batch_size=4)
    
    # 开始训练
    num_epochs = 2
    learning_rate = 1e-4
    print_every = 1 # 每10个epoch打印一次损失
    
    train_model(model, train_loader, test_loader, num_epochs, learning_rate, print_every, device)

# 你可以在主程序中调用main()，传入数据集和模型


if __name__ == "__main__":
    inputs = np.random.random([1000,100,100])
    targets = (np.random.random([1000,100,100])>0.5).astype(np.float)
    simulator = get_default_simulator()
    simulator.dill_c.requires_grad_(False)
    main(inputs, targets, simulator)