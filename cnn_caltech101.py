import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from CalTech101Dataset import CalTech101Dataset
from mylib import resize_and_center, select_test_file, test_by_file
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

CHECKPOINT_PATH = 'CNN_caltech101.pth'

train_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(5, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5)
])

# 2. 验证集弱增强（仅必要预处理）
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CalTech101Dataset(data_dir="./processed",train=True,transform=train_transform)
test_dataset =  CalTech101Dataset(data_dir="./processed",train=False,transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNN_Caltech101Net(nn.Module):
    def __init__(self):
        super(CNN_Caltech101Net, self).__init__()
        
        # 卷积块
        self.conv_blocks = nn.Sequential(
            # Block 1: 64 features
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 128 features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 256 features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 512 features
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 512 features
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 101)
        )

        if os.path.exists(CHECKPOINT_PATH):
            self.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            print('Loaded model weights from', CHECKPOINT_PATH)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
    def trainModel(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001,weight_decay=0.3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        epochs = 20
        
        for epoch in range(epochs):
            pbar=tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
            self.train()
            total_loss = 0
            train_correct = 0
            train_total=0
            for batch_idx, (data, target) in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
    
                train_acc = train_correct / train_total
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}','Train Acc':f'{train_acc:.2%}'})
    
            scheduler.step()
            self.eval()
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = self(data)
                    _, predicted = torch.max(outputs, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
                test_acc = test_correct / test_total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2%} Test Acc: {test_acc:.2%}')
        torch.save(self.state_dict(), CHECKPOINT_PATH)
        print('Model weights saved to', CHECKPOINT_PATH)
if __name__ == "__main__":
    model = CNN_Caltech101Net().to(device)

    model.trainModel()
    #test_by_file(3,model,train_dataset.categories)
