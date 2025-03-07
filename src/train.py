import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from PIL import Image
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 数据预处理（保持不变）
train_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomCrop(288),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.7, scale=(0.03, 0.15), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 模型定义（保持不变）
class AgePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        for name, param in base_model.named_parameters():
            if 'features.5' in name or 'features.6' in name or 'features.7' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 768),
            torch.nn.Sigmoid()
        )
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 1)
        )
        
        self.shortcut = torch.nn.Linear(768, 1)
        self.base_model = base_model

    def forward(self, x):
        features = self.base_model.features(x)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)
        
        att = self.attention(features)
        features = features * att
        
        return 0.7*self.regressor(features) + 0.3*self.shortcut(features)

# 组合损失函数（保持不变）
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, target):
        return (1-self.alpha)*self.l1(pred, target) + self.alpha*self.mse(pred, target)

# 数据集类（保持不变）
class AgeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        age = torch.tensor(row['age'], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
            
        return img, age

# 评估函数（保持不变）
def evaluate(model, loader):
    model.eval()
    errors, all_targets, all_preds = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets_cpu = targets.cpu().numpy()
            preds = np.atleast_1d(model(inputs).squeeze().cpu().numpy())
            errors.extend(np.abs(preds - targets_cpu))
            all_targets.extend(targets_cpu)
            all_preds.extend(preds)
    print(f"误差90%分位数: {np.quantile(errors, 0.9):.2f}")
    return np.array(errors), np.array(all_targets), np.array(all_preds)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载
    BATCH_SIZE = 64 if torch.cuda.is_available() else 32
    train_dataset = AgeDataset("train.csv", train_transform)
    val_dataset = AgeDataset("val.csv", val_transform)
    
    # 样本加权
    age_bins = pd.cut(train_dataset.df['age'], bins=[0,12,18,30,50,100])
    class_weights = 1.0 / age_bins.value_counts().sort_index().values
    sample_weights = class_weights[age_bins.cat.codes]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 模型初始化
    model = AgePredictor().to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.features[5].parameters(), 'lr': 1e-5},
        {'params': model.base_model.features[6].parameters(), 'lr': 1e-5},
        {'params': model.base_model.features[7].parameters(), 'lr': 1e-5},
        {'params': model.attention.parameters(), 'lr': 3e-4},
        {'params': model.regressor.parameters(), 'lr': 1e-3},
        {'params': model.shortcut.parameters(), 'lr': 1e-4}
    ], weight_decay=0.005)
    
    criterion = CombinedLoss(alpha=0.3)
    
    # 学习率调度器
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=20,
        eta_min=1e-6
    )
    
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练指标记录
    best_mae = float('inf')
    early_stop_counter = 0
    val_mae_history = []  # 新增：记录每个 epoch 的验证 MAE

    # 训练循环
    for epoch in range(30):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 混合精度训练
            with torch.amp.autocast('cuda'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=0.1 * (1 + epoch/100)
            )
            
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix(loss=loss.item())

        # 验证阶段
        val_errors, val_targets, val_preds = evaluate(model, val_loader)
        val_mae = np.mean(val_errors)
        val_mae_history.append(val_mae)  # 记录当前 epoch 的 MAE
        
        # 学习率调度
        if epoch < 10:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(val_mae)
        
        # 早停机制
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'best_con_tiny_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 7:
                print("早停触发")
                break

        print(f"Epoch {epoch+1} | Val MAE: {val_mae:.2f} | Best MAE: {best_mae:.2f}")

    # ==================== 新增：MAE 趋势图 ====================
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(val_mae_history) + 1)
    
    # 绘制验证 MAE 曲线
    plt.plot(epochs, val_mae_history, 'b-o', linewidth=2, markersize=8, label='Validation MAE')
    
    # 标注最佳值
    best_epoch = np.argmin(val_mae_history) + 1
    best_mae_value = np.min(val_mae_history)
    plt.scatter(best_epoch, best_mae_value, s=200, c='red', 
                edgecolors='black', zorder=5, 
                label=f'Best MAE: {best_mae_value:.2f} (Epoch {best_epoch})')
    
    # 图表装饰
    plt.title('MAE Trend during Training', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.xticks(epochs, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('mae_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 原有可视化 ====================
    plt.figure(figsize=(12, 5))
    
    # 误差分布图
    plt.subplot(1, 2, 1)
    sns.histplot(val_errors, kde=True, bins=30, color='royalblue')
    plt.title("Error Distribution", fontsize=12)
    plt.xlabel('Absolute Error (years)')
    
    # 预测值 vs 真实值
    plt.subplot(1, 2, 2)
    plt.scatter(val_targets, val_preds, alpha=0.5, color='darkorange')
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2)
    plt.title("Predictions vs Ground Truth", fontsize=12)
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()