import os
import shutil
from PIL import Image
import imagehash
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
def check_dup(folder_path):
    """
    Check all image files (jpg, png, webp) in a folder for duplicates and move them to a 'dup' subfolder.
    
    Args:
        folder_path (str): Path to the folder containing images to check
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # Create dup folder if it doesn't exist
    dup_folder = os.path.join(folder_path, 'dup')
    
    
    # Get all image files in the folder
    image_files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(file)
    
    # Dictionary to store hash values and corresponding files
    hashes = {}
    duplicates = []
    
    # Calculate perceptual hash for each image
    for image_file in image_files:
        try:
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path) as img:
                # Calculate perceptual hash
                hash_value = imagehash.phash(img)
                
                # Check if similar hash already exists (allowing for small differences)
                found_duplicate = False
                for existing_hash, existing_files in hashes.items():
                    # If hashes are very similar (difference <= 5), consider them duplicates
                    if hash_value - existing_hash <= 5:
                        existing_files.append(image_file)
                        duplicates.extend(existing_files)
                        found_duplicate = True
                        break
                
                # If no similar hash found, add as new entry
                if not found_duplicate:
                    hashes[hash_value] = [image_file]
                    
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Move duplicates to dup folder
    moved_files = set()
    for hash_value, files in hashes.items():
        if len(files) > 1:  # More than one file with similar hash
            # Move all but the first (keep one in original location)
            for file in files[1:]:
                if file not in moved_files:
                    src_path = os.path.join(folder_path, file)
                    dst_path = os.path.join(dup_folder, file)
                    try:
                        if not os.path.exists(dup_folder):
                            os.makedirs(dup_folder)
                        shutil.move(src_path, dst_path)
                        moved_files.add(file)
                        print(f"Moved duplicate: {file}")
                    except Exception as e:
                        print(f"Error moving {file}: {e}")

    return len(moved_files) 
def Caltech101_catetories():
    return ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'Faces_easy', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'Motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
def check_all_cls():
    duped = 0
    parentpath = "testdata\\"
    for cls in Caltech101_catetories():
        folder_path = os.path.join(parentpath, cls)
        print(f"Checking duplicates in category: {cls}")
        cnt = check_dup(folder_path)
        duped += cnt
        print(f"  Found {cnt} duplicates in category: {cls}")
    print(f"Total duplicates found across all categories: {duped}")



class MYCaltechDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
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
    def __init__(self, root_dir='testdata', transform=transform, train=True):
        self.root_dir = root_dir
        self.categories = Caltech101_catetories()
        self.transform = transform
        self.train = train
        
        # Create mapping from category name to index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.categories)}
        
        # Collect all image paths and their labels
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """
        Load all image paths and their corresponding labels.
        """
        for class_idx, class_name in enumerate(self.categories):
            class_path = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Category folder {class_path} does not exist")
                continue
            
            # Get all image files in the category folder
            image_files = []
            for file in os.listdir(class_path):
                if self._is_image_file(file):
                    image_files.append(file)
            
            # Sort files to ensure consistent ordering
            image_files.sort()
            
            # Select either training or test samples
            if self.train:
                # All but last 3 images for training
                selected_files = image_files[:-5] if len(image_files) > 5 else image_files
            else:
                # Last 3 images for testing
                selected_files = image_files[-5:] if len(image_files) >= 5 else image_files
            
            # Add samples to the list
            for file_name in selected_files:
                file_path = os.path.join(class_path, file_name)
                self.samples.append((file_path, class_idx))
    
    def _is_image_file(self, filename):
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        _, ext = os.path.splitext(filename)
        return ext.lower() in image_extensions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.resize_and_center(image, (224, 224))
            
            # Apply transforms if any
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and label in case of error
            return Image.new('RGB', (224, 224)), label
def test_dataset():
    

    # Create training dataset
    train_dataset = MYCaltechDataset(
        train=True
    )

    # Create test dataset
    test_dataset = MYCaltechDataset(
        train=False
    )

    # Use with DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(len(train_loader.dataset), len(test_loader.dataset))
    fig,axes = plt.subplots(6,6, figsize=(10,10))
    axes = axes.ravel()
    samples=[]
    for i in range(36):
        p=torch.randint(0,len(train_dataset)-1,(1,)).item()
        while p in samples:
            p=torch.randint(0,len(train_dataset)-1,(1,)).item()
        img,label = train_dataset[p]
        axes[i].imshow(img.permute(1,2,0))
        axes[i].set_title(f"{train_dataset.categories[label]}")
        axes[i].axis('off')
        samples.append(p)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
    #check_all_cls()
    test_dataset()