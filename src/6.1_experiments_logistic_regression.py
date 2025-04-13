
import time
import tracemalloc

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from models import LinearRegression

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    ram_usage_list = []

    for img, labels in loader:
        img = img.view(img.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def test(model, loader, criterion, device):
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


def get_mean_std(batch_size):
    # VAR[x] = E[X**2] - E[X]**2
    transforms_temp = transforms.Compose([transforms.ToTensor()])
    train_data_temp = datasets.MNIST(
        "../datasets", train=True, transform=transforms_temp
    )
    train_loader_temp = DataLoader(train_data_temp, batch_size=batch_size, shuffle=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for img, _ in train_loader_temp:
        channels_sum += torch.mean(img)
        channels_squared_sum += torch.mean(img**2)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean, std


def main():
    # ? setup
    mini_batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 1

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

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = test(model, test_loader, criterion, device)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )
        end_time = time.time()
        result_list.append(
            [optimizer_name, test_loss_list, test_acc_list, end_time - start_time]
        )

    print("training complete")

    #? results loss
    plt.figure(figsize=(15, 8))
    for result in result_list:
        plt.plot(range(1, epochs + 1), result[1], label=result[0])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title("MNIST Linear Regression Training Loss by Optimizer")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # Render the plot
    plt.show()

    #? results accuracy
    plt.figure(figsize=(15, 8))
    for result in result_list:
        plt.plot(range(1, epochs + 1), result[2], label=result[0])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("MNIST Linear Regression Training Accuracy by Optimizer")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # Render the plot
    plt.show()

    #? results per epoch
    plt.figure(figsize=(15, 8))
    x = []
    y = []
    for result in result_list:
        x.append(result[0])
        y.append(result[3] / epochs)

    plt.bar(x, y)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Average Training Time Per Epoch")
    plt.grid(True, which="both")
    # Render the plot
    plt.show()

if __name__ == "__main__":
    #main()
    print("hello")

    import tracemalloc
    import nvidia_smi

    tracemalloc.start()
    for i in range(5):
        print(i)
        _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak / 10 ** 6