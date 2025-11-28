import os
import urllib.request
import urllib.parse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import ssl

# 禁用SSL验证警告（用于测试环境）
ssl._create_default_https_context = ssl._create_unverified_context

class CalTech101Dataset:
    """CalTech101数据集类，提供categories属性"""
    
    def __init__(self):
        # CalTech101的101个类别名称（来自官方数据集）
        self.categories = [
            'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion',
            'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver',
            'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha',
            'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan',
            'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
            'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup',
            'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
            'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo',
            'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano',
            'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate',
            'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama',
            'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
            'minaret', 'minotaur', 'motorbikes', 'nautilus', 'octopus', 'okapi',
            'pagoda', 'panda', 'pigeon', 'pizza', 'plasma_tv', 'platypus',
            'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
            'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler',
            'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower',
            'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair',
            'wild_cat', 'windsor_chair', 'yinyang'
        ]

class BingImageDownloader:
    """Bing图片下载器，不使用第三方downloader库"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def search_images(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        搜索Bing图片
        
        Args:
            query: 搜索关键词
            count: 需要的图片数量
            
        Returns:
            图片信息列表，每个元素包含url、width、height等
        """
        print(f"正在搜索: {query}")
        images = []
        offset = 0
        
        while len(images) < count:
            encoded_query = urllib.parse.quote(query)
            url = f"https://cn.bing.com/images/async?q={encoded_query}&first={offset}&count=35&relp=35&lostate=r&mmasync=1"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                
                # 使用正则表达式从响应中提取图片信息
                # Bing图片搜索返回的HTML中包含mjson数据
                pattern = r'mediaurl=(.*?)&amp;.*?width=(\d*?)&amp;.*?height=(\d*?)&amp;'
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                
                if not matches:
                    print(f"  未找到更多图片，当前已获取 {len(images)} 张")
                    break
                
                for mediaurl, width, height in matches:
                    if len(images) >= count:
                        break
                    
                    try:
                        # 清理URL
                        image_url = urllib.parse.unquote(mediaurl)
                        width = int(width) if width else 0
                        height = int(height) if height else 0
                        
                        # 过滤符合条件的图片
                        if width > 224 and height > 224:
                            images.append({
                                'url': image_url,
                                'width': width,
                                'height': height,
                                'query': query
                            })
                            print(f"  找到图片: {image_url[:80]}... (尺寸: {width}x{height})")
                        else:
                            print(f"  跳过图片: 尺寸 {width}x{height} 太小")
                    except Exception as e:
                        print(f"  解析图片信息失败: {e}")
                        continue
                
                offset += 35
                time.sleep(1)  # 添加延迟，避免请求过快
                
            except requests.RequestException as e:
                print(f"  请求失败: {e}")
                break
            except Exception as e:
                print(f"  发生错误: {e}")
                break
        
        print(f"搜索完成: {query}，共获取 {len(images)} 张图片")
        return images[:count]
    
    def download_image(self, image_info: Dict[str, Any], save_path: Path) -> bool:
        """
        下载单张图片
        
        Args:
            image_info: 图片信息字典
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        url = image_info['url']
        query = image_info['query']
        
        for attempt in range(self.max_retries):
            try:
                print(f"  下载中 ({attempt+1}/{self.max_retries}): {url[:60]}...")
                
                # 下载图片数据
                response = requests.get(url, headers=self.headers, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '')
                if 'image' not in content_type:
                    # 尝试从URL判断
                    if '.jpg' in url.lower() or '.jpeg' in url.lower():
                        ext = '.jpg'
                    elif '.png' in url.lower():
                        ext = '.png'
                    else:
                        print(f"    跳过: 不是图片类型 - {content_type}")
                        return False
                else:
                    # 根据内容类型确定扩展名
                    if 'jpeg' in content_type:
                        ext = '.jpg'
                    elif 'png' in content_type:
                        ext = '.png'
                    else:
                        print(f"    跳过: 不支持的图片格式 - {content_type}")
                        return False
                
                # 验证图片并获取尺寸
                try:
                    image = Image.open(io.BytesIO(response.content))
                    width, height = image.size
                    
                    # 再次验证尺寸
                    if width <= 224 or height <= 224:
                        print(f"    跳过: 尺寸 {width}x{height} 太小")
                        return False
                    
                    # 确保保存路径有正确的扩展名
                    if not save_path.suffix:
                        save_path = save_path.with_suffix(ext)
                    
                    # 保存图片
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"    成功: 保存到 {save_path} (尺寸: {width}x{height})")
                    return True
                    
                except Exception as e:
                    print(f"    图片验证失败: {e}")
                    return False
                    
            except Exception as e:
                print(f"    下载失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2  ** attempt)  # 指数退避
                continue
        
        return False


def download_images_for_categories(
    categories: List[str],
    output_dir: str = "  testdata",
    num_images: int = 10,
    delay: float = 2.0
) -> None:
    """
    为CalTech101的所有类别下载图片
    
    Args:
        categories: 类别名称列表
        output_dir: 输出根目录
        num_images: 每个类别下载的图片数量
        delay: 每个类别之间的延迟（秒）
    """
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建下载器实例
    downloader = BingImageDownloader()
    
    # 统计信息
    total_downloaded = 0
    category_stats = {}
    
    print(f"开始下载 {len(categories)} 个类别的图片...")
    print(f"每个类别 {num_images} 张，目标尺寸 > 224x224")
    print("=" * 60)
    
    # 遍历每个类别
    for i, category in enumerate(categories):
        print(f"\n处理类别 {i+1}/{len(categories)}: {category}")
        
        # 创建类别目录
        category_dir = output_path / category
        category_dir.mkdir(exist_ok=True)
        
        # 搜索图片
        images = downloader.search_images(category, count=num_images * 2)  # 搜索更多，过滤后可能减少
        
        if not images:
            print(f"  警告: 未找到 {category} 的图片")
            category_stats[category] = 0
            continue
        
        # 下载图片
        downloaded = 0
        for j, image_info in enumerate(images):
            if downloaded >= num_images:
                break
            
            save_path = category_dir / f"{category}_{j+1:03d}"
            if downloader.download_image(image_info, save_path):
                downloaded += 1
                total_downloaded += 1
        
        category_stats[category] = downloaded
        print(f"  类别 {category} 下载完成: {downloaded}/{num_images} 张")
        
        # 添加延迟，避免请求过快
        if i < len(categories) - 1:
            print(f"  等待 {delay} 秒...")
            time.sleep(delay)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("下载完成统计:")
    print(f"总类别数: {len(categories)}")
    print(f"成功下载图片总数: {total_downloaded} 张")
    print(f"平均每个类别: {total_downloaded / len(categories):.1f} 张")
    
    # 显示每个类别的下载情况
    print("\n各类别下载详情:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if count >= num_images else "⚠" if count > 0 else "✗"
        print(f"  {status} {category}: {count} 张")


def verify_downloads(output_dir: str = "testdata") -> None:
    """
    验证下载的图片，删除损坏或尺寸不足的文件
    
    Args:
        output_dir: 输出根目录
    """
    print("\n验证下载的图片...")
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"目录 {output_dir} 不存在")
        return
    
    total_files = 0
    valid_files = 0
    deleted_files = 0
    
    # 遍历所有类别目录
    for category_dir in output_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        print(f"检查类别: {category_dir.name}")
        
        # 检查该目录下的所有图片
        for image_file in category_dir.glob("*"):
            if not image_file.is_file():
                continue
            
            total_files += 1
            
            try:
                # 验证图片
                with Image.open(image_file) as img:
                    width, height = img.size
                    if width > 224 and height > 224:
                        valid_files += 1
                    else:
                        print(f"  删除尺寸不足: {image_file.name} ({width}x{height})")
                        image_file.unlink()
                        deleted_files += 1
            except Exception as e:
                print(f"  删除损坏图片: {image_file.name} - {e}")
                image_file.unlink()
                deleted_files += 1
    
    print(f"\n验证完成:")
    print(f"总文件数: {total_files}")
    print(f"有效文件: {valid_files}")
    print(f"已删除: {deleted_files}")


def main():
    """主函数"""
    # 创建CalTech101数据集实例
    dataset = CalTech101Dataset()
    
    # 下载图片
    download_images_for_categories(
        categories=dataset.categories,
        output_dir="testdata",
        num_images=10,
        delay=3.0  # 每个类别之间延迟3秒
    )
    
    # 验证下载的图片
    verify_downloads("testdata")
    
    print("\n所有任务完成！")


if __name__ == "__main__":
    main()