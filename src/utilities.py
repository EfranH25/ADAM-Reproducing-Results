from matplotlib import pyplot as plt


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
    for result in data:
        if "train_vram_usage" in result:
            plt.plot(result["train_vram_usage"], label=result["name"])

    plt.xlabel("iteration")
    plt.ylabel("vram (mb)")
    plt.yscale('log')
    plt.title(f"{name} Training VRAM Usage")
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
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{name} Average Training Time Per Epoch")
    plt.grid(True, which="both")
    # Render the plot
    plt.show()
