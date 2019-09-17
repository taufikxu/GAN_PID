import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import PIL.Image


class NumpyImageDataset(Dataset):
    def __init__(self, imgs, label, transform=None):
        self.imgs = imgs
        self.label = label
        self.transform = transform

        self.num_classes = np.unique(label).shape[0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        label = self.label[index]
        # print(img.shape)

        img = img.transpose([1, 2, 0])
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class MixtureOfGaussianDataset(Dataset):
    def __init__(self, config):
        config = config['data']
        self.shape = config['shape']
        self.radius = config['radius']
        self.std = config['std']
        self.num_mixture = config['num_mixture']

        self.centers = self.build_centers(self.shape, self.num_mixture,
                                          self.radius)
        self.rng = np.random.RandomState(1)

    def build_centers(self, shape, num_mixture, radius):
        if shape.lower() == "ring":
            thetas = np.linspace(0, 2 * np.pi, num_mixture, endpoint=False)
            xs = radius * np.sin(thetas, dtype=np.float32)
            ys = radius * np.cos(thetas, dtype=np.float32)
            centers = np.vstack([xs, ys]).T
            return centers
        elif shape.lower() == "grid":
            grid_range = int(np.sqrt(num_mixture))
            centers = []
            for i in range(-grid_range + 1, grid_range, 2):
                for j in range(-grid_range + 1, grid_range, 2):
                    centers.append(np.array([[i, j]]))
            centers = np.concatenate(centers, 0) * radius / 2.
            return centers
        else:
            raise ValueError(
                "Unknown shape for mixture of gaussian:{}".format(shape))

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        index = np.random.randint(0, self.num_mixture)
        center = self.centers[index]
        sample = self.rng.normal(size=center.shape) * self.std + center
        # print(sample, type(sample), sample.dtype, sample.shape)
        sample = torch.from_numpy(sample.astype(np.float32))
        return sample, torch.from_numpy(np.array([0]))


def get_dataset(name, data_dir, size=64, lsun_categories=None, config=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    ])

    if name == "MoG":
        dataset = MixtureOfGaussianDataset(config)
        nlabels = 1
    elif name.lower() == "celeba":
        imgs = np.load("/home/LargeData/celebA_64x64.npy")
        labels = np.zeros([imgs.shape[0]]).astype(np.int64)
        dataset = NumpyImageDataset(imgs, labels, transform)
        nlabels = 1
    elif name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, 'npy')
        nlabels = len(dataset.classes)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform)
        nlabels = 10
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir,
                                     transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    else:
        raise NotImplementedError

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img / 127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img


if __name__ == "__main__":
    # config = dict({})
    # config['data'] = dict({})
    # config['data']['shape'] = "ring"
    # config['data']['num_mixture'] = 8
    # config['data']['radius'] = 1
    # config['data']['std'] = 0.01
    # dataset = MixtureOfGaussianDataset(config)

    config = dict({})
    config['data'] = dict({})
    config['data']['shape'] = "grid"
    config['data']['num_mixture'] = 25
    config['data']['radius'] = 2
    config['data']['std'] = 0.01
    dataset = MixtureOfGaussianDataset(config)
    print(dataset.__getitem__(0))
