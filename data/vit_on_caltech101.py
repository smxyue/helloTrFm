import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("D:\\yue\\helloTrFm\\data\\caltech101")
from CalTech101Dataset import CalTech101Dataset


model_path = "vit_b_16_caltech101.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
# ==================== 1. 数据加载与预处理 ====================

def get_data_loaders(batch_size=128):
    
    all_data = CalTech101Dataset()
    
    train_dataset = Subset(all_data, range(0, len(all_data) - 1000))  # First N-1000 samples
    test_dataset = Subset(all_data, range(len(all_data) - 1000, len(all_data)))  # Last 1000 samples
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

# ==================== 2. 模型定义 ====================

class ViTForMNIST(nn.Module):
    """使用torchvision预训练的ViT模型"""
    def __init__(self, model_name='vit_b_16', pretrained=True, num_classes=102):
        super().__init__()
        
        # 加载torchvision预定义模型
        if model_name == 'vit_b_16':
            self.vit = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            self.vit = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            self.vit = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_l_32':
            self.vit = models.vit_l_32(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 修改分类头（从1000类改为10类）
        # 注意：torchvision的ViT模型使用self.vit.heads而不是self.vit.head
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded pretrained weights from {model_path}")
        
    def forward(self, x):
        return self.vit(x)

# ==================== 3. 训练与评估 ====================

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(test_loader), correct / total

# ==================== 4. 主训练流程 ====================

def main():
    # 超参数配置
    config = {
        'batch_size': 64,
        'epochs': 10,
        'lr': 1e-4,
        'model_name': 'vit_b_16',  # 可选: 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'
        'pretrained': True,  # 使用ImageNet预训练权重
    }
    
    # 设备配置
    
    
    # 数据加载
    train_loader, test_loader = get_data_loaders(config['batch_size'])
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")   
    # 模型初始化
    model = ViTForMNIST(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        num_classes=102
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 如果使用预训练模型，建议使用较小的学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    print(f"\nModel: {config['model_name']} (pretrained: {config['pretrained']})")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练循环
    best_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({100.*train_acc:.2f}%)")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} ({100.*test_acc:.2f}%)")
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_{config["model_name"]}_mnist.pth')
            print(f"✓ New best model saved! Acc: {100.*best_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best Test Accuracy: {100.*best_acc:.2f}%")
    print(f"{'='*60}")
    torch.save(model.state_dict(), model_path)
# ==================== 5. 运行 ====================
def predict():
    model = ViTForMNIST(
        model_name="vit_b_16",
        pretrained=False,
        num_classes=102
    ).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model weights not found at {model_path}!")
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
def show_catalogries():
    dataset = CalTech101Dataset()
    categories = dataset.categories
    dataset.close()
    print(f"Categories: {categories}")
    print(f"Total number of images: {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape}")
if __name__ == '__main__':
    predict()
    #show_catalogries()
    pass