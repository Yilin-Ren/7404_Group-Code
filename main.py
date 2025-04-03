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
from MobileNetV2_CA2 import MobileNetV2_CA
from torch.optim.lr_scheduler import OneCycleLR
from MobileNeXt_CA import MobileNeXt_CA
from MobileNeXt import MobileNeXt
from MobileNetV2_SE import MobileNetV2_SE
from MobileNetV2 import MobileNetV2
from MobileNeXt_SE import MobileNeXt_SE

config = {
    # 可选: "MobileNetV2_Standard", "MobileNetV2_CA", "MobileNetV2_SE", "MobileNeXt_CA", "MobileNeXt_Standard"
    "model_name": "MobileNeXt_SE",
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,  # 新增标签平滑
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
    ])
}

def setup_logging(model_variant):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{model_variant}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
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
    if model_name == "MobileNetV2_Standard":
        return MobileNetV2()
    elif model_name == "MobileNetV2_CA":
        return MobileNetV2_CA()
    elif model_name == "MobileNetV2_SE":
        return MobileNetV2_SE()
    elif model_name == "MobileNeXt_CA":
        return MobileNeXt_CA()    
    elif model_name == "MobileNeXt_Standard":
        return MobileNeXt()
    elif model_name == "MobileNeXt_SE":
        return MobileNeXt_SE()
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

def train():
    prepare_dataset()
    train_loader, val_loader = build_dataloaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=config["epochs"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    
    model_variant = config["model_name"]
    log_dir = setup_logging(model_variant)
    best_acc = 0.0
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    "loss": f"{epoch_loss/(pbar.n+1):.3f}",
                    "acc": f"{100*correct/total:.1f}%"
                })
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(), tqdm(val_loader, desc="Validating") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    "acc": f"{100*val_correct/val_total:.1f}%"
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(log_dir, f"{model_variant}_best_model.pth"))
            logging.info(f"New best model saved at epoch {epoch+1} with acc {val_acc:.2f}%")
        
        logging.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.3f} Acc: {val_acc:.1f}%"
        )
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train")
    plt.plot(metrics["val_loss"], label="Val")
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train")
    plt.plot(metrics["val_acc"], label="Val")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.savefig(os.path.join(log_dir, f"{model_variant}_training_metrics.png"))
    plt.close()

if __name__ == "__main__":
    train()