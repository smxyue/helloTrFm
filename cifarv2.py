# 1. 环境准备
import pickle
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import torchvision.transforms as T

from mylib import load_images, resize_and_center, select_test_file

CHECKPOINT_PATH = 'cifarv2.pth'
cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std  = [0.2023, 0.1994, 0.2010]
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),            # CIFAR 常用
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    # 保守起见暂时移除 RandomPerspective/GaussianBlur/RandomAffine（可做实验开启）
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(5, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=cifar_mean, std=cifar_std),
    T.RandomErasing(p=0.2)
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=cifar_mean, std=cifar_std),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
#train_dataset = Subset(train_dataset, range(100))
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# 3. Vision Transformer模型构建

class PatchEmbedding(nn.Module):
    """将图像分割成patches并嵌入到向量空间"""
    def __init__(self, img_size=64, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层高效地分割和嵌入patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 64, 64)
        x = self.proj(x)  # (batch_size, embed_dim, 8, 8)
        x = x.flatten(2)  # (batch_size, embed_dim, 64_patches)
        x = x.transpose(1, 2)  # (batch_size, 64_patches, embed_dim)
        return x

class Attention(nn.Module):
    """自注意力机制的核心"""
    def __init__(self, embed_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Q, K, V线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 加权求和
        out = (attn_probs @ v).transpose(1, 2)  # (batch_size, n_patches, n_heads, head_dim)
        out = out.reshape(batch_size, n_patches, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """完整的Transformer编码器块"""
    def __init__(self, embed_dim, n_heads, mlp_ratio=3.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP部分
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    """完整的Vision Transformer模型"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, n_classes=10, 
                 embed_dim=128, depth=6, n_heads=4):
        super().__init__()
        
        # Patch嵌入层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 分类token（CLS token）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Transformer编码器堆叠
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # (batch_size, 64_patches, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 65_tokens, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 提取CLS token用于分类
        x = self.norm(x[:, 0])  # 只取第一个token (CLS token)
        return self.head(x)
    def netsize(self):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层用 xavier，避免输出爆炸
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm 的 gamma 初始为 1, beta 为 0
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # Patch embedding 用 kaiming
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# 4. 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False) 
    print('Loaded model from checkpoint.')
else:
    model.init_weights()
    print('Initialized model weights.')
# 使用交叉熵损失和AdamW优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 5. 训练循环
def train_epoch(model, loader, criterion, optimizer, device):
    
    total = 0
    correct = 0
    total_loss = 0

    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪，防止不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
        train_acc = correct / total
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}','Train Acc':f'{train_acc:.2%}'})
    
    return total_loss / len(loader),correct / total
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
def maintrain():
    # 6. 主训练流程
    epoches=50
    print(f"Using device: {device}")
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epoches):  # 仅训练10个epochs作为演示
        train_loss,train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate(model, test_loader, device)
        #print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        print(f"Epoch {epoch+1}/{epoches} | Train Loss: {train_loss:.4f} Train Accuracy:{train_accuracy:.4f} Test accuracy: {test_accuracy:.4f}")
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Training complete!state_dict saved to {CHECKPOINT_PATH}.")
    
    epochs_range = range(1, epoches + 1)
    
    # Plot accuracies together
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accuracies, 'g-', marker='o', label='Training Accuracy')
    plt.plot(epochs_range, test_accuracies, 'r-', marker='s', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # If you still want to see the loss plot separately
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, 'b-', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def predict():
    model = VisionTransformer().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print(f"Model weights not found at {CHECKPOINT_PATH}!")
        return
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    c=0
    while c<9:
        # Create a simple GUI for file selection
        try:
            import tkinter as tk
            from tkinter import filedialog
        
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Make dialog appear on top
        
            # Open file dialog to select JPG file
            image_path = filedialog.askopenfilename(
                title="Select a JPG Image",
                filetypes=[
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("PNG files", "*.png"),
                    ("All image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )
        
            # Destroy the root window
            root.destroy()
        
            # Check if user cancelled the dialog
            if not image_path:
                print("No file selected.")
                return
        
        except ImportError:
            print("tkinter not available. Falling back to manual input...")
            image_path = input("Enter the path to your JPG image: ").strip()
    
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found!")
            return
    
        # Process image
        try:
            from PIL import Image
            import torchvision.transforms as T
        
            # Define preprocessing transforms (same as used during training)
            preprocess = T.Compose([
                T.Resize((32, 32)),  # ViT requires 224x224 images
                T.ToTensor(),          # Convert to tensor and scale to [0,1]
                T.Normalize(mean=cifar_mean, std=cifar_std),
                #.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        
            # Load and process image
            image = Image.open(image_path).convert('RGB')  # Ensure RGB format
            image = resize_and_center(image, (32, 32))

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
                    
            # Make prediction
            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            # Get category name if available
            predicted_category = f"{train_dataset.classes[predicted_class]}"    
            print(f"Predicted class index: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Top 5 predictions:")
        
            # Show top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for j in range(top5_prob.size(0)):
                print(f"{j+1}. Class {top5_catid[j].item()}: {top5_prob[j].item():.4f}")
            
            # Display image
 
            axes[c].imshow(image)
            axes[c].set_title(f"{predicted_category}:({confidence:.2f})")
            axes[c].axis('off')
            c+=1
        
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    fig.tight_layout()
    plt.show()

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverse normalization for proper visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def testmnist_dataset():
    startIndex = torch.randint(0, len(train_dataset) - 9, (1,)).item()
    img,lable = train_dataset[startIndex]
    
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {lable}')
    plt.show()

def show_dataset(datasetname):
    startIndex = torch.randint(0, len(datasetname) - 100, (1,)).item()
    fig,axes = plt.subplots(10, 10, figsize=(12, 12))
    axes = axes.ravel()
    for i in range(100):
        image, lable = datasetname[startIndex + i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'{lable}')
        axes[i].axis('off')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #model = VisionTransformer().to(device)
    #print(model.netsize())
    #maintrain()
    predict()
    #showsample()
    #show_image_data()
    #testmnist_dataset()
    #show_dataset(train_dataset)
    pass