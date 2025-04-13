import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import LinearRegression
from utilities import build_graphs
from utilities import get_mean_std_mnist as get_mean_std
from utilities import train_model_mnist as train_model
from utilities import test_model_mnist as test_model


def main(mini_batch_size, learning_rate, weight_decay, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA or CPU:", device)

    # ? getting mean & var
    train_mean, train_std = get_mean_std(mini_batch_size)

    # ? loading data
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=train_mean, std=train_std)]
    )
    train_dataset = datasets.MNIST(
        "../datasets", train=True, download=True, transform=transformer
    )
    test_dataset = datasets.MNIST(
        "../datasets", train=True, download=True, transform=transformer
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=mini_batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=mini_batch_size)

    # ? training model
    optimizer_options = ["Adam", "SGDNesterov", "AdaGrad"]
    result_list = []

    for optimizer_name in optimizer_options:
        print("Optimizer: ", optimizer_name)
        model = LinearRegression(28 * 28, 10).to(device)
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
    build_graphs(result_list, epochs, name="MNIST Linear Regression")


if __name__ == "__main__":
    mini_batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 2
    main(mini_batch_size, learning_rate, weight_decay, epochs)
