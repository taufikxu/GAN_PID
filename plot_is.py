import os
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utils_log import MetricSaver

matplotlib.use('Agg')
# old_model = "./output/0Finished/Baseline/cifar_baseline__2019-08-18-20-34-20-wgangp-is5.7/"
old_model = "./output/0Finished/2019-09-03-17-02-04_cifar_pid__d1/"

log_file = os.path.join(old_model, "logs/stats.p")
with open(log_file, 'rb') as f:
    dat = pickle.load(f)

inception_score = dat['inception_score']['mean']
is_meters = MetricSaver("Inception Score", old_model, save_on_update=False)
for it, iscore in inception_score:
    is_meters.update(step=it, value=iscore, save=False)
is_meters.save()
