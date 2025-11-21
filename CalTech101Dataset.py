import os
import json
import gzip
from PIL import Image
import io

from matplotlib import pyplot as plt
import numpy as np
import torch

import sys

import torchvision
sys.path.append('D:\yue\helloTrFm')
from helloworld import VisionTransformer



class CalTech101Dataset:
    def __init__(self, data_dir="processed", target_size=(224, 224)):
        """
        Caltech101数据集类
        
        Args:
            data_dir: 处理后的数据目录
            target_size: 图像尺寸 (高, 宽)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.image_bytes_per_file = target_size[0] * target_size[1] * 3
        
        # 文件路径
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.bin_path = os.path.join(data_dir, "caltech101_data.bin")
        self.gz_path = self.bin_path + ".gz"
        
        # 检查并解压文件
        self._check_and_extract()
        
        # 加载元数据
        self._load_metadata()
        
        # 打开二进制文件
        self.bin_file = open(self.bin_path, 'rb')
        
        # 构建类别到索引的映射
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
    def _check_and_extract(self):
        """检查文件是否存在，如有需要则解压"""
        # 检查元数据文件
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"找不到元数据文件: {self.metadata_path}")
        
        # 检查二进制文件
        if not os.path.exists(self.bin_path):
            if os.path.exists(self.gz_path):
                print(f"发现压缩文件 {self.gz_path}，正在解压...")
                self._extract_gz()
            else:
                raise FileNotFoundError(f"找不到数据文件: {self.bin_path} 或 {self.gz_path}")
        else:
            print(f"使用已存在的二进制文件: {self.bin_path}")
    
    def _extract_gz(self):
        """解压gzip文件"""
        print(f"正在解压 {self.gz_path}...")
        with gzip.open(self.gz_path, 'rb') as f_in:
            with open(self.bin_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"解压完成: {self.bin_path}")
        
        # 可选择删除压缩文件以节省空间
        # os.remove(self.gz_path)
        # print(f"已删除压缩文件: {self.gz_path}")
    
    def _load_metadata(self):
        """加载元数据"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.metadata = metadata
        self.num_images = metadata["num_images"]
        self.image_size = metadata["image_size"]  # [H, W, C]
        self.categories = metadata["categories"]
        self.image_infos = metadata["images"]
        
        print(f"加载数据集: {self.num_images} 张图片, {len(self.categories)} 个类别")
        print(f"图像尺寸: {self.image_size}")
    
    def __len__(self):
        """返回数据集大小"""
        return self.num_images
    
    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签
        
        Args:
            idx: 索引
            
        Returns:
            dict: 包含 'image', 'category', 'category_idx', 'filename'
        """
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_images-1}]")
        
        # 获取图像信息
        img_info = self.image_infos[idx]
        
        # 从二进制文件读取图像数据
        self.bin_file.seek(img_info["offset"])
        img_bytes = self.bin_file.read(img_info["size"])
        
        # 转换为图像
        image = Image.frombytes(
            'RGB', 
            (self.image_size[1], self.image_size[0]), 
            img_bytes
        )
        image = torch.from_numpy(np.array(image)).permute(2,0,1).float()/255.0
        label = self.categories.index(img_info['category'])
        label = torch.tensor(label)
        return image,label
        #return {
        #    'image': image,
        #    'label': label,
        #    'category': img_info['category'],
        #    'filename': img_info['filename']
        #}
    
    def get_images_by_category(self, category):
        """
        获取指定类别的所有图像索引
        
        Args:
            category: 类别名称
            
        Returns:
            list: 图像索引列表
        """
        return [i for i, info in enumerate(self.image_infos) 
                if info['category'] == category]
    
    def close(self):
        """关闭文件"""
        if hasattr(self, 'bin_file') and not self.bin_file.closed:
            self.bin_file.close()
            print("已关闭二进制文件")
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()

    def resize_and_center(self, img, target_size):
        """
        按比例缩放图像并居中
    
        Args:
        img: PIL Image object
        target_size: (width, height) tuple for target size
    
        Returns:
        PIL Image resized and centered within target size
        """
        target_width, target_height = target_size
        original_width, original_height = img.size
    
        # 计算缩放比例，保持宽高比
        scale = min(target_width / original_width, target_height / original_height)
    
        # 计算新尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    
        # 缩放图像
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
        # 如果尺寸已经匹配，则直接返回
        if new_width == target_width and new_height == target_height:
            return img_resized
    
        # 创建新的图像（黑色背景）
        new_img = Image.new('RGB', (target_width, target_height), (255,255, 255))
    
        # 计算居中位置
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
    
        # 将缩放后的图像粘贴到中心位置
        new_img.paste(img_resized, (x_offset, y_offset))
    
        return new_img
def get_cifar10_classes():
    """获取CIFAR-10数据集的类别名称"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
def test_db():
    # 创建数据集实例
    dataset = CalTech101Dataset(data_dir="processed")
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数: {len(dataset.categories)}")
    print(f"类别示例: {dataset.categories[:5]}")
    
    fig,axes=plt.subplots(5,5, figsize=(10,10))
    axes=axes.ravel()
    for i in range(25):# 获取单个样本
        sample = dataset[torch.randint(0,len(dataset)-1, (1,)).item()]
        image, label = sample
        

        # 显示图像（需要matplotlib）
        try:
            axes[i].imshow(image.permute(1,2,0))  # Need to permute dimensions for visualization
            axes[i].set_title(f"{label.item()}: {dataset.categories[label.item()]}")
            axes[i].axis('off')
                
        except ImportError:
            print("未安装matplotlib，无法显示图像")
    fig.tight_layout()
    plt.show()
    # 清理资源
    dataset.close()
def test_cifar():
    dataset = CalTech101Dataset(data_dir="processed")
    cifar_class=get_cifar10_classes()
    filtered_indices = []

    for i in range(len(dataset)):
        image,label = dataset[i]
        category = dataset.categories[label]
        #'airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck'
        fit_class=['airplanes', 'wild_cat','dalmatian','ketch','pigeon']
        if category in fit_class:
            filtered_indices.append(i)
    fit_count=[]
    for cls in fit_class:
        fit_count.append([cls,0])
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_indices)}")
    if len(filtered_indices) < 10:
        print("Not enough samples to show")
        return
    
    fig,axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()
    i=0
    while i<16:
        lucker=torch.randint(0, len(filtered_indices), (1,)).item()
        image,lable = dataset[filtered_indices[lucker]]
        category = dataset.categories[lable]
        print(f"类别: {category}, 图像尺寸: {image.shape}")
  
        if category in fit_class and fit_count[fit_class.index(category)][1] < 4:
            # Convert tensor back to PIL Image for resizing
            # First convert tensor to numpy array
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Now resize using the method
            newimg = dataset.resize_and_center(pil_image, (64, 64))
            
            # Convert PIL Image back to tensor for model input
            newimg_array = np.array(newimg) / 255.0  # Normalize to [0,1]
            newimg_tensor = torch.tensor(newimg_array, dtype=torch.float32)
            # Change dimension order from HWC to CHW and add batch dimension
            newimg_tensor = newimg_tensor.permute(2, 0, 1).unsqueeze(0)
        
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = VisionTransformer().to(device)
            model.load_state_dict(torch.load('vit_cifar10.pth', map_location=device))
            model.eval()
        
            # Move tensor to device
            newimg_tensor = newimg_tensor.to(device)
        
            with torch.no_grad():
                output = model(newimg_tensor)    # 预测
                pred_label = output.argmax(dim=1).item()
                pred_label = cifar_class[pred_label]
                axes[i].set_title(f"S:{category} P:{pred_label}")
                axes[i].imshow(pil_image)
                axes[i].axis('off')
                
            filtered_indices.pop(lucker)
            i += 1
            fit_count[fit_class.index(category)][1] += 1
        elif len(filtered_indices) <= 0:
            break
    plt.tight_layout()
    plt.show()
# 使用示例
if __name__ == "__main__":
    pass
    #test_db()
    test_cifar()