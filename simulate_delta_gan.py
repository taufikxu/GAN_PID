import numpy as np
from utils_log import MetricSaver

data = 1.
delta_t = 0.01


class GAN_simualte(object):
    def __init__(self, gantype, controller_d):
        self.type = gantype
        self.controller_d = controller_d
        self.d = 0.
        self.g = 0.

    def d_step(self):
        error = data - self.g
        error = self.controller_d(error)
        self.d += error * delta_t

    def g_step(self):
        self.g += self.d * delta_t


class PID_controller(object):
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d

        self.i_buffer = 0.
        self.d_buffer = 0.

    def __call__(self, error):
        p_signal = error

        self.i_buffer += error * delta_t
        i_signal = self.i_buffer

        d_signal = (error - self.d_buffer) / delta_t
        self.d_buffer = error

        return self.p * p_signal + self.i * i_signal + self.d * d_signal


p, i, d = 10, 11, 10
saver = MetricSaver("Generator_{}_{}_{}".format(p, i, d),
                    "./delta_gan/",
                    save_on_update=False)
controller = PID_controller(p, i, d)
gan = GAN_simualte('gan', controller)

for i in range(10000):
    gan.d_step()
    gan.g_step()
    saver.update(i, gan.g, save=False)
saver.save()
