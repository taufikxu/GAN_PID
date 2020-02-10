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
import glob

# def movingaverage(r, window_size=1):
#     if window_size == 1:
#         return r
#     pad = np.zeros([window_size]) + r[-1]
#     r = np.concatenate([r, pad], 0)
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(r, window, 'same')

device = torch.device("cuda:0")

# total_inception = dict({})

# model_list.reverse()
all_results = dict({})
all_models = glob.glob("./output/Plot*")
print(len(all_models))
all_models.reverse()
for epoch_id in range(80):
    for model in all_models:
        model_name = "/home/kunxu/Workspace/GAN_PID/{}".format(model)
        key_name = model_name
        if key_name not in all_results:
            all_results[key_name] = []

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

        ckptpath = os.path.join(
            model_name, "chkpts",
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
        all_results[key_name].append([float(m), float(s)])

with open("./output/cifar_inception_plot.pkl", 'wb') as f:
    pickle.dump(all_results, f)
