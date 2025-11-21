import os
import json
import gzip
from PIL import Image
from tqdm import tqdm

class Caltech101Preprocessor:
    def __init__(self, data_dir=r"data\caltech101\101_ObjectCategories", 
                 output_dir="processed",
                 target_size=(224, 224)):
        """
        预处理Caltech101数据集
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            target_size: 目标图像尺寸 (高, 宽)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.image_bytes_per_file = target_size[0] * target_size[1] * 3
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def scan_files(self):
        """扫描所有jpg文件"""
        image_files = []
        categories = []
        
        for category in os.listdir(self.data_dir):
            category_path = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            categories.append(category)
            
            for file in os.listdir(category_path):
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(category_path, file)
                    image_files.append({
                        'category': category,
                        'filename': file,
                        'path': full_path
                    })
        
        print(f"找到 {len(categories)} 个类别, {len(image_files)} 张图片")
        return image_files, categories
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
    def process(self):
        """主处理函数"""
        # 扫描文件
        image_files, categories = self.scan_files()
        # 元数据
        metadata = {
            "version": "1.0",
            "image_size": [self.target_size[0], self.target_size[1], 3],  # H, W, C
            "num_images": len(image_files),
            "categories": categories,
            "images": []
        }
        
        # 创建二进制文件
        bin_path = os.path.join(self.output_dir, "caltech101_data.bin")
        with open(bin_path, 'wb') as f:
            offset = 0
            
            for img_info in tqdm(image_files, desc="Processing images"):
                try:
                    # 打开并处理图像
                    with Image.open(img_info['path']) as img:
                        # 转换为RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 调整大小（使用高质量插值）
                        img_resized = self.resize_and_center(img, (self.target_size[1], self.target_size[0]))
                        
                        # 获取字节数据
                        img_bytes = img_resized.tobytes()
                        
                        # 写入二进制文件
                        f.write(img_bytes)
                        
                        # 记录元数据
                        metadata["images"].append({
                            "category": img_info['category'],
                            "filename": img_info['filename'],
                            "offset": offset,
                            "size": len(img_bytes)
                        })
                        
                        offset += len(img_bytes)
                        
                except Exception as e:
                    print(f"处理 {img_info['path']} 时出错: {e}")
                    continue
        
        # 保存元数据
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"二进制文件保存至: {bin_path}")
        print(f"元数据保存至: {meta_path}")
        
        # 压缩二进制文件
        self.compress(bin_path)
        
        return bin_path, meta_path
    
    def compress(self, bin_path):
        """压缩二进制文件"""
        gz_path = bin_path + ".gz"
        
        print("正在压缩文件...")
        with open(bin_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb', compresslevel=6) as f_out:
                f_out.writelines(f_in)
        
        # 获取文件大小
        original_size = os.path.getsize(bin_path)
        compressed_size = os.path.getsize(gz_path)
        
        print(f"压缩完成: {gz_path}")
        print(f"原始大小: {original_size / 1024 / 1024:.2f} MB")
        print(f"压缩后大小: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"压缩率: {compressed_size / original_size:.2%}")
        
        # 删除原始文件（可选）
        os.remove(bin_path)
        print(f"已删除原始文件: {bin_path}")


if __name__ == "__main__":
    # 使用示例
    preprocessor = Caltech101Preprocessor()
    preprocessor.process()