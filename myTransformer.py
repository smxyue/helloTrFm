# 1. 环境准备
import pickle
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CalTech101Dataset import CalTech101Dataset

CHECKPOINT_PATH = 'myTransformer_caltech101.pth'
train_dataset = CalTech101Dataset(data_dir="./processed")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Vision Transformer模型构建

class PatchEmbedding(nn.Module):
    """将图像分割成patches并嵌入到向量空间"""
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
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
    def __init__(self, embed_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Q, K, V线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = attn_scores.softmax(dim=-1)
        
        # 加权求和
        out = (attn_probs @ v).transpose(1, 2)  # (batch_size, n_patches, n_heads, head_dim)
        out = out.reshape(batch_size, n_patches, embed_dim)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """完整的Transformer编码器块"""
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
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
    def __init__(self, img_size=224, patch_size=8, in_channels=3, n_classes=101, 
                 embed_dim=128, depth=6, n_heads=8):
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

# 4. 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False) 
    print('Loaded model from checkpoint.')
# 使用交叉熵损失和AdamW优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# 5. 训练循环
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

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
    print(f"Using device: {device}")
    for epoch in range(30):  # 仅训练10个epochs作为演示
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        #test_acc = evaluate(model, test_loader, device)
        #print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} ")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("Training complete!state_dict saved to {CHECKPOINT_PATH}}.")
    # 预期输出（示例）：
    # Epoch 1/10 | Train Loss: 1.9876 | Test Accuracy: 0.3124
    # Epoch 10/10 | Train Loss: 0.4567 | Test Accuracy: 0.7245


def predict():
    model = VisionTransformer().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print(f"Model weights not found at {CHECKPOINT_PATH}!")
        return
    model.eval()
    dataset = CalTech101Dataset()
    categories = dataset.categories
    dataset.close()
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
            
            print(f"Selected file: {image_path}")
        
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
                T.Resize((224, 224)),  # ViT requires 224x224 images
                T.ToTensor(),          # Convert to tensor and scale to [0,1]
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        
            # Load and process image
            image = Image.open(image_path).convert('RGB')  # Ensure RGB format
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
            print(f"Processed image shape: {input_batch.shape}")
        
            # Make prediction
            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            # Get category name if available
            if categories and predicted_class < len(categories):
                predicted_category = categories[predicted_class]
            else:
                predicted_category = f"Class {predicted_class}"    
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
if __name__ == "__main__":
    #maintrain()
    predict()
    #showsample()
    #show_image_data()
    pass