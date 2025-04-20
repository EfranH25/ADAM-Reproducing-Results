# ADAM-Reproducing-Results
This repo is to recreate and extend the results of the [ADAM research paper](https://arxiv.org/abs/1412.6980) with Pytorch.

## Methodology
Using PyTorch we recreated the regression, neural net, and cnn models from the ADAM paper. We also implemented a vision transformer as an example of a more modern model.The Bag of Words regression model for the IMDB dataset was not used due to library incompatibilities with torchtext and pytorch.  

We tracked the loss, accuracy, vram usage, and training time. 
To save time, we used 25 epochs and used generic hyperparameters instead of tuning with dense grid search.



## Hardware
| Hardware                      | Component         |
|-------------------------------|-------------------|
| GPU                           | RTX 5090          |
| CPU                           | Ryzen 9 7900X3D   |
| RAM                           | 32GB RAM and VRAM |
| OS                            | Windows 11        |


## Charts
Below are performance charts for each model.
### Regression + MNIST

### Neural Nets + MNIST

### ConvNet + CIFAR10

### ViT + CIFAR10

## Results
As you can see, the ADAM optimizer performs the best and is similar to the results of the ADAM paper. The ADAM model performaned the best with little tuning across models, showing it's flexability across architectures and problems. 