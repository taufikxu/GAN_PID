import os
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utils_log import MetricSaver
from gan_training.chainer_evaluate import evaluation
import pickle
import torchvision

import torch

from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models)

# def movingaverage(r, window_size=1):
#     if window_size == 1:
#         return r
#     pad = np.zeros([window_size]) + r[-1]
#     r = np.concatenate([r, pad], 0)
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(r, window, 'same')

device = torch.device("cuda:0")

# total_inception = dict({})

# # model_list.reverse()
# gan_type = "hinge"
# all_results = dict({})
# for epoch_id in range(80):
#     for reg_coe in [2, 5, 10]:
#         for rep in [1, 2, 3]:
#             model_name = "/home/kunxu/Workspace/GAN_PID/output/Final_cifar_pid_{}_{}_rep{}".format(
#                 gan_type, reg_coe, rep)
#             key_name = "cifar_{}_{}_rep{}".format(gan_type, reg_coe, rep)
#             if key_name not in all_results:
#                 all_results[key_name] = []

#             config = load_config(os.path.join(model_name, "config.yaml"),
#                                  'configs/default.yaml')
#             generator, discriminator = build_models(config)
#             generator = torch.nn.DataParallel(generator)
#             zdist = get_zdist(config['z_dist']['type'],
#                               config['z_dist']['dim'],
#                               device=device)
#             ydist = get_ydist(1, device=device)
#             checkpoint_io = CheckpointIO(checkpoint_dir="./tmp")
#             checkpoint_io.register_modules(generator_test=generator)
#             evaluator = Evaluator(generator,
#                                   zdist,
#                                   ydist,
#                                   batch_size=100,
#                                   device=device)

#             ckptpath = os.path.join(
#                 model_name, "chkpts",
#                 "model_{:08d}.pt".format(epoch_id * 10000 + 9999))
#             print(ckptpath)
#             load_dict = checkpoint_io.load(ckptpath)
#             img_list = []
#             for i in range(500):
#                 ztest = zdist.sample((100, ))
#                 x = evaluator.create_samples(ztest)
#                 img_list.append(x.cpu().numpy())
#             img_list = np.concatenate(img_list, axis=0)
#             m, s = evaluation(img_list)
#             all_results[key_name].append([float(m), float(s)])

# with open("./output/cifar_inception{}.pkl".format(gan_type), 'wb') as f:
#     pickle.dump(all_results, f)

import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
])
dataset = torchvision.datasets.CIFAR10(root="./data/",
                                       train=True,
                                       download=True,
                                       transform=transform)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=100,
    shuffle=False,
)
img_list = []
for batch in loader:
    x, y = batch
    img_list.append(x.cpu().numpy())
img_list = np.concatenate(img_list, axis=0)
print(img_list.shape)
m, s = evaluation(img_list)
print(m, s)
exit()

all_results = []
path = "Final_cifar_nopid_sigmoid.1_"
for epoch_id in range(40, 80):
    model_name = "/home/kunxu/Workspace/GAN_PID/output/" + path
    config = load_config(os.path.join(model_name, "config.yaml"),
                         'configs/default.yaml')
    generator, discriminator = build_models(config)
    generator = torch.nn.DataParallel(generator)
    zdist = get_zdist(config['z_dist']['type'],
                      config['z_dist']['dim'],
                      device=device)
    ydist = get_ydist(1, device=device)
    checkpoint_io = CheckpointIO(checkpoint_dir="./tmp")
    checkpoint_io.register_modules(generator_test=generator)
    evaluator = Evaluator(generator,
                          zdist,
                          ydist,
                          batch_size=100,
                          device=device)

    ckptpath = os.path.join(model_name, "chkpts",
                            "model_{:08d}.pt".format(epoch_id * 10000 + 9999))
    print(ckptpath)
    load_dict = checkpoint_io.load(ckptpath)
    img_list = []
    for i in range(500):
        ztest = zdist.sample((100, ))
        x = evaluator.create_samples(ztest)
        img_list.append(x.cpu().numpy())
    img_list = np.concatenate(img_list, axis=0)
    m, s = evaluation(img_list)
    all_results.append([float(m), float(s)])
print("RegGAN with Hinge", max(all_results))
