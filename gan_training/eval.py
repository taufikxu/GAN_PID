import torch
from torchvision import transforms
from gan_training.metrics import inception_score
import matplotlib
from matplotlib import pyplot as plt
import PIL
import numpy as np

matplotlib.use("Agg")


class Evaluator(object):
    def __init__(self,
                 generator,
                 zdist,
                 ydist,
                 batch_size=64,
                 inception_nsamples=60000,
                 device=None):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while (len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size, ))
            ytest = self.ydist.sample((self.batch_size, ))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(imgs,
                                           device=self.device,
                                           resize=True,
                                           splits=10)

        return score, score_std

    def create_samples(self,
                       z,
                       y=None,
                       toy=False,
                       x_real=None,
                       contour_matrix=None):
        if toy is False:
            self.generator.eval()
            batch_size = z.size(0)
            # Parse y
            if y is None:
                y = self.ydist.sample((batch_size, ))
            elif isinstance(y, int):
                y = torch.full((batch_size, ),
                               y,
                               device=self.device,
                               dtype=torch.int64)
            # Sample x
            with torch.no_grad():
                x = self.generator(z, y)
            return x
        else:
            z_sample = self.zdist.sample((10000, ))
            y_sample = self.ydist.sample((10000, ))
            y_sample = torch.clamp(y_sample, None, 0)
            with torch.no_grad():
                x_fake = self.generator(z_sample, y_sample)

            np_samples_data = x_real.data.cpu().numpy()
            np_samples_gen = x_fake.data.cpu().numpy()

            fig = plt.figure(figsize=(5, 5))
            plt.scatter(np_samples_data[:, 0],
                        np_samples_data[:, 1],
                        s=8,
                        c='r',
                        edgecolor='none',
                        alpha=0.05)
            plt.scatter(np_samples_gen[:, 0],
                        np_samples_gen[:, 1],
                        s=8,
                        c='b',
                        edgecolor='none',
                        alpha=0.05)
            show_range = 1.4
            plt.xlim((-show_range, show_range))
            plt.ylim((-show_range, show_range))
            plt.grid(True)
            plt.tight_layout()

            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                            canvas.tostring_rgb())
            plt.close()
            plt.close(fig)

            img_tensor = transforms.ToTensor()(pil_image)
            img_tensor = torch.unsqueeze(img_tensor, 0)

            fig = plt.figure(figsize=(5, 5))
            n = 1000
            x = np.arange(n)
            y = np.arange(n)
            X, Y = np.meshgrid(x, y)
            cp = plt.contour(X, Y, contour_matrix, 20)
            plt.clabel(cp, inline=True, fontsize=7)
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                            canvas.tostring_rgb())
            plt.close()
            plt.close(fig)

            contour_tensor = transforms.ToTensor()(pil_image)
            contour_tensor = torch.unsqueeze(contour_tensor, 0)

            final_tensor = torch.cat([img_tensor, contour_tensor], 0)

            return final_tensor
