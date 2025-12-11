import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
def select_test_file(count):
    c=0
    images = []

    while c<count:
        # Create a simple GUI for file selection
        try:
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
            
            #print(f"Selected file: {image_path}")
        
        except ImportError:
            print("tkinter not available. Falling back to manual input...")
            image_path = input("Enter the path to your JPG image: ").strip()
    
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found!")
            return
    
        # Process image
        try:
            
             # Load and process image
            images.append(image_path)
            c+=1
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    return images
def test_by_file(nCount,model,categories):
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file = select_test_file(nCount*nCount)
    processed_images = []
    for f in test_file:
        img=Image.open(f).convert('RGB')
        img = resize_and_center(img, (224, 224))
        image_array=np.array(img)
        processed_img = torch.from_numpy(image_array).float()
        processed_img = processed_img.permute(2, 0, 1) / 255.0  # Convert to CxHxW and scale to [0,1]
        processed_images.append(processed_img)
        test_tensors = torch.stack(processed_images)  # Stack into batch tensor
    test_tensors=test_tensors.to(mydevice)
    model.to(mydevice)
    model.eval()
    with torch.no_grad():
        outputs = model(test_tensors)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)  # Get predicted class for each sample
         # Get category name if available
        #probs, predicted = torch.max(outputs.data, 1)
        fig,axes=plt.subplots(nCount,nCount, figsize=(15,15))
        axes=axes.flatten()
        num_images = len(test_file)
        for i in range(num_images):
            img = test_tensors[i].permute(1, 2, 0).cpu().numpy()  # Convert tensor to HWC format for plotting
            pred = predicted_classes[i].item()
            prob = probabilities[i][pred].item()  # Probability of the predicted class for this sample
            
            axes[i].imshow(img)
            # Make sure we don't access out-of-bounds category index
            if pred < len(categories):
                category_name = categories[pred]
            else:
                category_name = f"Class {pred}"
                
            axes[i].set_title(f'Pred: {category_name} ({prob:.2f})')
            axes[i].axis('off')
        fig.tight_layout()
        plt.show()
def resize_and_center(img, target_size):
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
def load_images(image_paths, target_size=(224, 224)):   
    processed_images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = resize_and_center(img, target_size)
        processed_images.append(img)
    return processed_images

def show_paremeters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
def test1():
    selected_files = select_test_file(9)
    print(f"Number of selected images: {len(selected_files)}")
    selected_images = load_images(selected_files)
    fig,axes = plt.subplots(1, len(selected_images), figsize=(15,5))
    axes = axes.ravel()
    for ax, img, path in zip(axes, selected_images, selected_files):
        ax.imshow(img)
        ax.set_title(os.path.basename(path))
        ax.axis('off')
    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    test1()