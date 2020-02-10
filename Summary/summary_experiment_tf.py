import os
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')

fig = plt.figure(figsize=(5, 3))
markersize = 4
markevery = 4
consider_time = True

all_list = [x * 10000 + 9999 for x in range(100)]
tmp_list = [
    "Plot_cifar_standard_0_",
    "Plot_cifar_wgan_0_",
    "Plot_cifar_lsgan_0_",
    "Plot_cifar_hinge_0_",
    "Plot_cifar_standard_",
    "Plot_cifar_wgan_",
    "Plot_cifar_lsgan_",
    "Plot_cifar_hinge_",
    # "Plot_cifar_wgangp_",
    "Plot_cifar_standard_5_",
    "Plot_cifar_wgan_5_",
    "Plot_cifar_lsgan_5_",
    "Plot_cifar_hinge_5_",
]

model_name_cifar = {
    "Plot_cifar_standard_0_": "SGAN",
    "Plot_cifar_wgan_0_": "WGAN",
    "Plot_cifar_wgangp_": "WGAN-GP",
    "Plot_cifar_lsgan_0_": "LSGAN",
    "Plot_cifar_hinge_0_": "Hinge-GAN",
    "Plot_cifar_standard_": "Reg-SGAN",
    "Plot_cifar_wgan_": "Reg-WGAN",
    "Plot_cifar_lsgan_": "Reg-LSGAN",
    "Plot_cifar_hinge_": "Reg-Hinge-GAN",
    "Plot_cifar_standard_5_": "CLC-SGAN(5)",
    "Plot_cifar_wgan_5_": "CLC-WGAN(5)",
    "Plot_cifar_lsgan_5_": "CLC-LSGAN(5)",
    "Plot_cifar_hinge_5_": "CLC-Hinge-GAN(5)",
}

marker_cifar = {
    "SGAN": ".",
    "LSGAN": ".",
    "WGAN": ".",
    "Hinge-GAN": ".",
    "Reg-SGAN": "x",
    "Reg-WGAN": "x",
    "Reg-LSGAN": "x",
    "Reg-Hinge-GAN": "x",
    "WGAN-GP": "x",
    "CLC-SGAN(5)": "^",
    "CLC-WGAN(5)": "^",
    "CLC-LSGAN(5)": "^",
    "CLC-Hinge-GAN(5)": "^",
}

time_factor_cifar = {
    "SGAN": 5,
    "LSGAN": 5,
    "WGAN": 5,
    "Hinge-GAN": 5,
    "Reg-SGAN": 9,
    "Reg-WGAN": 9,
    "Reg-LSGAN": 9,
    "Reg-Hinge-GAN": 9,
    "WGAN-GP": 10,
    "CLC-SGAN(5)": 8,
    "CLC-WGAN(5)": 8,
    "CLC-LSGAN(5)": 8,
    "CLC-Hinge-GAN(5)": 8,
}

with open("./output/cifar_inception_plot.pkl", 'rb') as f:
    dat = pickle.load(f)
    total_inception = dict({})
    for item in dat:
        allis = dat[item]
        allis = [x[0] for x in allis]
        total_inception[os.path.basename(item)] = np.array(allis)

cmap = plt.get_cmap('jet')
for ind, model_name in enumerate(tmp_list):

    inception_score = total_inception[model_name]
    label = model_name_cifar[model_name]
    print(label, np.max(inception_score),
          np.where(inception_score == np.max(inception_score)))
    # inception_score = movingaverage(inception_score)
    time = list(range(len(inception_score)))
    if consider_time is True:
        time = [t / 36 * time_factor_cifar[label] for t in time]
    # print(time)
    plt.plot(
        time,
        inception_score,
        marker_cifar[label] + '-',
        linewidth=0.4,
        label=label,
        #  color=cmap(float(ind) / len(total_inception)),
        markersize=markersize,
        markevery=markevery)

if consider_time is True:
    plt.xlabel("Time/Hours")
else:
    plt.xlabel('Iterations')
plt.ylabel('Inception Score')
plt.ylim((2, 8.9))
plt.grid(False)
legend = plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('./Curve_IS_CIFAR.pdf', bbox_inches='tight')
plt.close(fig)
