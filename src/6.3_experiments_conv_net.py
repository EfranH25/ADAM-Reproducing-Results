import time
import tracemalloc
import nvidia_smi

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import ConvNet
from utilities import build_graphs


def train_model(model, loader, criterion, optimizer, device):
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
        img = img.view().to(device)
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


def test_model(model, loader, criterion, device):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0

    for img, labels in loader:
        img = img.view().to(device)
        labels = labels.to(device)

        outputs = model(img)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return loss_sum / total, 100.0 * correct / total


def main(mini_batch_size, learning_rate, weight_decay, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA or CPU:", device)

    # ? loading data
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10("../datasets", train=True, download=True, transform=transformer)
    test_dataset = datasets.CIFAR10("../datasets", train=True, download=True, transform=transformer)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=mini_batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=mini_batch_size)

    # ? training model
    optimizer_options = ["Adam", "SGDNesterov", "AdaGrad"]
    result_list = []

    for optimizer_name in optimizer_options:
        print("Optimizer: ", optimizer_name)
        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()

        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "SGDNesterov":
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        elif optimizer_name == "AdaGrad":
            optimizer = optim.Adagrad(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            print(f"Warning: Optimizer {optimizer_name} not a condition. Skipping")
            continue

        train_ram_usage_list = []
        test_loss_list = []
        test_acc_list = []

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_ram_usage = train_model(
                model, train_loader, criterion, optimizer, device)

            test_loss, test_acc = test_model(model, test_loader, criterion, device)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            train_ram_usage_list.extend(train_ram_usage)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )
        end_time = time.time()

        result_dic = {
            "name": optimizer_name,
            "test_loss": test_loss_list,
            "test_acc": test_acc_list,
            "train_vram_usage": train_ram_usage_list,
            "runtime": end_time - start_time
        }
        result_list.append(result_dic)

    print("training complete")
    build_graphs(result_list, epochs, name="CIFAR10 Conv Net")


if __name__ == "__main__":
    mini_batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 2
    main(mini_batch_size, learning_rate, weight_decay, epochs)
