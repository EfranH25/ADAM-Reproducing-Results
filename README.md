# ADAM Reproducing Results
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
### Regression MNIST
Architecture: 
in > FC > out
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.1_mnist_loss.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.1_mnist_acc.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.1_mnist_time.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.1_mnist_vram.png?raw=true)

### Neural Nets MNIST
Architecture: 
in > linear 1000 > relu > dropout > linear 1000 > relu > dropout > linear 10 > out
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.2_mnist_loss.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.2_mnist_acc.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.2_mnist_time.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.2_mnist_vram.png?raw=true)

### ConvNet CIFAR10
Architecture: 
in > conv 64 + kernal 5 > relu > maxpool 3x2 > conv 64 + kernal 5 > relu > maxpool 3x2 > conv 128 + kernal 5 > relu > maxpool 3x2 > linear 1000 > relu > linear 10 > out
Architecture (with dropout): 
in > conv 64 + kernal 5 > relu > dropout >  maxpool 3x2 > conv 64 + kernal 5 > relu > maxpool 3x2 > conv 128 + kernal 5 > relu > maxpool 3x2 > linear 1000 > relu > dropout > linear 10 > out

![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.3_cfar10_loss.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.3_cfar10_acc.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.3_cfar10_time.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.3_cfar10_vram.png?raw=true)

### ViT CIFAR10
Architecture:
in > normalize > mutlihead_attention > normalize > mlp > out 

![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.4_cfar10__vit_loss.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.4_cfar10__vit_acc.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.4_cfar10__vit_time.png?raw=true)
![image](https://github.com/EfranH25/ADAM-Reproducing-Results/blob/main/charts/6.4_cfar10__vit_vram.png?raw=true)


## Conclusion
As you can see, the ADAM optimizer performs the best and is similar to the results of the ADAM paper. The ADAM model performaned the best with little tuning across models, showing it's flexability across architectures and problems. 
