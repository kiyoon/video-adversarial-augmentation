import kornia.augmentation as K
import torch
from torch import nn

from . import zen_store


def t(x):
    return torch.tensor(x)


def p(x):
    return nn.Parameter(t(x))


@zen_store(name="image_augmentation")
class VideoAugmentation(nn.Module):
    def __init__(self):
        super(VideoAugmentation, self).__init__()
        self.affine = K.RandomAffine(
            degrees=p([-20.0, 20.0]),
            translate=p([0.1, 0.1]),
            scale=p([0.8, 1.2, 0.8, 1.2]),
            shear=p([0.0, 0.0, 0.0, 0.0]),
            same_on_batch=False,
            p=1.0,
        )
        self.jitter = K.ColorJiggle(
            brightness=p([0.8, 0.8]),
            contrast=p([0.7, 0.7]),
            saturation=p([0.6, 0.6]),
            hue=t([0.1, 0.1]),
            p=1.0,
            same_on_batch=False,
        )
        self.gamma = K.RandomGamma(
            gamma=p([0.5, 2.0]), gain=p([1.0, 1.0]), p=1.0, same_on_batch=False
        )
        self.posterize = K.RandomPosterize(
            bits=p([4.0, 8.0]), p=1.0, same_on_batch=False
        )
        self.solarize = K.RandomSolarize(
            thresholds=p([0.0, 0.5]),
            additions=p([0.0, 0.5]),
            p=1.0,
            same_on_batch=False,
        )
        self.grayscale = K.RandomGrayscale(p=0.5, same_on_batch=False)
        self.erase = K.RandomErasing(
            scale=p([0.02, 0.33]),
            ratio=p([0.3, 3.3]),
            p=1.0,
            same_on_batch=False,
        )
        self.horizontal_flip = K.RandomHorizontalFlip(p=0.5)

    def forward(self, input):
        if self.training:
            N, T, C, H, W = input.shape
            x = input.view(N * T, C, H, W)
            x = self.affine(x)
            x = self.jitter(x)
            x = self.gamma(x)
            x = self.posterize(x)
            x = self.solarize(x)
            x = self.grayscale(x)
            x = self.erase(x)
            x = self.horizontal_flip(x)

            x = x.view(N, T, C, H, W)
            return x
        else:
            return input


@zen_store(name="aug_and_classify")
class AugAndClassify(nn.Module):
    def __init__(self, aug_model, model):
        super(AugAndClassify, self).__init__()
        self.aug_model = aug_model
        self.model = model

    def forward(self, x):
        if self.training:
            x = self.aug_model(x)

        x = self.model(x)
        return x
