from matplotlib import pyplot as plt
import numpy as np
import nvidia_smi
import tracemalloc

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_model_mnist(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    ram_usage_list = []

    if device == "cuda":
        nvidia_smi.nvmlInit()

    else:
        tracemalloc.start()

    for img, labels in loader:
        img = img.view(img.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if device == "cuda":
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            ram_usage_list.append(gpu_info.used / 1024 ** 2)
        else:
            current, _ = tracemalloc.get_traced_memory()
            ram_usage_list.append(current)

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    if device == "cuda":
        nvidia_smi.nvmlShutdown()
    else:
        tracemalloc.stop()

    return running_loss / total, 100.0 * correct / total, ram_usage_list


def test_model_mnist(model, loader, criterion, device):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0

    for img, labels in loader:
        img = img.view(img.size(0), -1).to(device)
        labels = labels.to(device)

        outputs = model(img)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return loss_sum / total, 100.0 * correct / total


def train_model_cfar(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    ram_usage_list = []

    if device == "cuda":
        nvidia_smi.nvmlInit()

    else:
        tracemalloc.start()

    for img, labels in loader:
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if device == "cuda":
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            ram_usage_list.append(gpu_info.used / 1024 ** 2)
        else:
            current, _ = tracemalloc.get_traced_memory()
            ram_usage_list.append(current)

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    if device == "cuda":
        nvidia_smi.nvmlShutdown()
    else:
        tracemalloc.stop()

    return running_loss / total, 100.0 * correct / total, ram_usage_list


def test_model_cfar(model, loader, criterion, device):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0

    for img, labels in loader:
        img = img.to(device)
        labels = labels.to(device)

        outputs = model(img)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return loss_sum / total, 100.0 * correct / total


def get_mean_std_mnist(batch_size):
    # VAR[x] = E[X**2] - E[X]**2
    transforms_temp = transforms.Compose([transforms.ToTensor()])
    train_data_temp = datasets.MNIST(
        "../datasets", train=True, transform=transforms_temp
    )
    train_loader_temp = DataLoader(train_data_temp, batch_size=batch_size, shuffle=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for img, _ in train_loader_temp:
        channels_sum += torch.mean(img)
        channels_squared_sum += torch.mean(img ** 2)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def build_graphs(data, epochs, name, figsize=(15, 8)):
    # ? results loss
    plt.figure(figsize=figsize)
    for result in data:
        if "test_loss" in result:
            plt.plot(range(1, epochs + 1), result["test_loss"], label=result["name"])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title(f"{name} Training Loss by Optimizer")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # Render the plot
    plt.show()

    # ? results accuracy
    plt.figure(figsize=figsize)
    for result in data:
        if "test_acc" in result:
            plt.plot(range(1, epochs + 1), result["test_acc"], label=result["name"])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{name} Training Accuracy by Optimizer")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # Render the plot
    plt.show()

    # ? results training vram
    plt.figure(figsize=figsize)
    x = []
    y = []
    for result in data:
        if "train_vram_usage" in result:
            x.append(result["name"])
            y.append(np.max(result["train_vram_usage"]))

    plt.bar(x, y)
    plt.xlabel("optimizer")
    plt.ylabel("max vram (mb)")
    plt.title(f"{name} Training Max VRAM Usage")
    plt.grid(True, which="both")
    # Render the plot
    plt.show()

    # ? runtim per epoch
    plt.figure(figsize=figsize)
    x = []
    y = []
    for result in data:
        if "runtime" in result:
            x.append(result["name"])
            y.append(result["runtime"] / epochs)

    plt.bar(x, y)
    plt.xlabel("optimizer")
    plt.ylabel("runtime per epoch")
    plt.title(f"{name} Average Training Time Per Epoch")
    plt.grid(True, which="both")
    # Render the plot
    plt.show()
