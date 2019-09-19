Dependencies 
- Python 3.6.9
- Pytorch 1.1.0
- torchvision 0.3.0
- numpy 1.17
- statsmodels 0.10.1
- matplotlib 3.1.1
- Pillow 6.1
- scipy 1.3.0

The dataset will be downloaded automatically for CIFAR10. For CelebA, please refer to the official website.

For toy Data, 'python train_toy.py ./config/Mog_pid.yaml'.
For CIFAR10 and celebA, the baseline can be reproduced by the following: 

'python train.py ./config/cifar_pid.yaml'

'python train.py ./config/celeba_pid.yaml'

For our proposed method, use the following commands:

'python train_pid.py ./config/cifar_pid.yaml'

'python train_pid.py ./config/celeba_pid.yaml'

The hyperparameters can be adjusted in the corresponding config files. Specifically, the 'iv' denotes the coefficient for negative feedback.