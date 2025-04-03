import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MobileNetV2_CA import MobileNetV2_CA
from datetime import datetime
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Load dependies 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torchvision import models
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import logging
import shutil
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

# MobileNetV2
from MobileNetV2 import MobileNetV2
from MobileNetV2_CA2 import MobileNetV2_CA
from MobileNetV2_SE import MobileNetV2_SE

# MobileNeXt
from MobileNeXt import MobileNeXt
from MobileNeXt_CA import MobileNeXt_CA
from MobileNeXt_SE import MobileNeXt_SE

# 加载模型

config = {
    "model_name": "MobileNeV2",
    "batch_size": 1,
    "dataset_path": "./data/tiny_imagenet",
    "train_transform": transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),  # 扩大裁剪范围
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 新增垂直翻转
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),  # 增加HSV扰动
        transforms.RandomGrayscale(p=0.1),  # 灰度化
        transforms.GaussianBlur(kernel_size=3),  # 高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2302, 0.2265, 0.2262])
    ]),
    "test_transform": transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2302, 0.2265, 0.2262])
    ]),
    "MobileNetV2_Standard_Path": "./logs/MobileNetV2_Standard_model/MobileNetV2_Standard_best_model.pth",
    "MobileNetV2_SE_Path": "./logs/MobileNetV2_SE_model/MobileNetV2_SE_best_model.pth",
    "MobileNetV2_CA_Path": "./logs/MobileNetV2_CA_model/MobileNetV2_CA_best_model.pth",
}

def setup_logging(model_variant):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{model_variant}_CAM_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def remove_hidden_dirs(root_dir):
    for subdir in os.listdir(root_dir):
        if subdir.startswith("."):
            path = os.path.join(root_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

def prepare_dataset():
    dataset_path = config["dataset_path"]
    extract_path = os.path.join(dataset_path, "tiny-imagenet-200")
    
    if not os.path.exists(extract_path):
        os.makedirs(dataset_path, exist_ok=True)
        zip_path = os.path.join(dataset_path, "tiny-imagenet-200.zip")
        
        if not os.path.exists(zip_path):
            logging.info("Downloading dataset...")
            os.system(f"wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P {dataset_path}")
        
        logging.info("Extracting dataset...")
        os.system(f"unzip -q {zip_path} -d {dataset_path}")
        
        val_dir = os.path.join(extract_path, "val")
        anno_file = os.path.join(val_dir, "val_annotations.txt")
        
        with open(anno_file) as f:
            for line in tqdm(f, desc="Organizing validation set"):
                if not line.strip():
                    continue
                img_name, class_name = line.strip().split()[:2]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_dir, "images", img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
        
        images_dir = os.path.join(val_dir, "images")
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

def build_dataloaders():
    train_transform = config["train_transform"]
    test_transform = config["test_transform"]
    
    base_path = os.path.join(config["dataset_path"], "tiny-imagenet-200")
    
    remove_hidden_dirs(os.path.join(base_path, "train"))
    remove_hidden_dirs(os.path.join(base_path, "val"))
    
    assert os.path.isdir(os.path.join(base_path, "train")), "训练集路径错误"
    assert os.path.isdir(os.path.join(base_path, "val")), "验证集路径错误"
    
    train_set = ImageFolder(os.path.join(base_path, "train"), train_transform)
    val_set = ImageFolder(os.path.join(base_path, "val"), test_transform)
    
    train_loader = DataLoader(train_set, config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def get_model():
    model_name = config["model_name"]
    if model_name == "MobileNetV2":
        return MobileNetV2(), MobileNetV2_SE(), MobileNetV2_CA()
    else:
        raise ValueError(f"未知的模型名称: {model_name}")


def visualize():
    prepare_dataset()
    train_loader, val_loader = build_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_Standard, model_SE, model_CA = get_model()
    model_variant = config["model_name"]
    if model_variant == "MobileNetV2":
        model_checkpoint = torch.load(config["MobileNetV2_Standard_Path"])
        model_Standard.load_state_dict(model_checkpoint['state_dict'])
        model_SE_checkpoint = torch.load(config["MobileNetV2_SE_Path"])
        model_SE.load_state_dict(model_SE_checkpoint['state_dict'])
        model_CA_checkpoint = torch.load(config["MobileNetV2_CA_Path"])
        model_CA.load_state_dict(model_CA_checkpoint['state_dict'])

    model_Standard = model_Standard.to(device)
    model_SE = model_SE.to(device)
    model_CA = model_CA.to(device)

    cam_extractor_Standard = SmoothGradCAMpp(model_Standard.eval())

    model_Standard.eval()
    cam_extractor_SE = SmoothGradCAMpp(model_SE.eval())

    model_SE.eval()
    cam_extractor_CA = SmoothGradCAMpp(model_CA.eval())
    model_CA.eval()
    

    log_dir = setup_logging(model_variant)

    with tqdm(val_loader, desc="Visualization") as pbar:
        index  = 0
        for image, label in pbar:
            if index == 100: break
            image, label = image.to(device), label.to(device)
            output_Standard = model_Standard(image)
            output_SE = model_SE(image)
            output_CA = model_CA(image)

            # 生成 Grad-CAM
            cam_Standard = cam_extractor_Standard(output_Standard.squeeze(0).argmax().item(), output_Standard)
            result_Standard = overlay_mask(to_pil_image(image.squeeze(0)), to_pil_image(cam_Standard[0], mode="F"), alpha=0.5)

            cam_SE = cam_extractor_SE(output_SE.squeeze(0).argmax().item(), output_SE)
            result_SE = overlay_mask(to_pil_image(image.squeeze(0)), to_pil_image(cam_SE[0], mode="F"), alpha=0.5)

            cam_CA = cam_extractor_CA(output_CA.squeeze(0).argmax().item(), output_CA)
            result_CA = overlay_mask(to_pil_image(image.squeeze(0)), to_pil_image(cam_CA[0], mode="F"), alpha=0.5)

            filename_Standard = os.path.join(log_dir, f"gradcam_Standard_{index}.png")
            filename_SE = os.path.join(log_dir, f"gradcam_SE_{index}.png")
            filename_CA = os.path.join(log_dir, f"gradcam_CA_{index}.png")

            result_Standard.save(filename_Standard)
            result_CA.save(filename_CA)
            result_SE.save(filename_SE)

            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]

            denormalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            )

            image_out = denormalize(image).cpu()           
            if image_out.dim() == 4:
                image_out = image_out[0]  # (B,C,H,W) -> (C,H,W)
            
            # 转换为PIL.Image
            to_pil = transforms.ToPILImage()
            image_out = to_pil(image_out)
            
            # save image
            filename_image = os.path.join(log_dir, f"gradcam_input_{index}.png")
            image_out.save(filename_image)

            index = index+1

if __name__ == "__main__":
    visualize()