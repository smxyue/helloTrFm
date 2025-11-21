from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
# 必须添加 Resize，因为图片大小不一
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一分辨率
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]    # ImageNet统计值(用预训练模型时推荐)
    )
])

# 下载并加载
dataset = Caltech101(
    root='./data',
    download=True,
    transform=transform
)

print(f"总样本数: {len(dataset)}")  # ~9,144
print(f"类别数: {len(dataset.categories)}")  # 101
print(f"类别名称: {dataset.categories[:]}")  # 打印前5个类别