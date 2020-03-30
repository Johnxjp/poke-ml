import torch
import torch.nn as nn


def real_loss(out, smooth=False):
    """
    Used for computing loss for discriminator against real images
    and for generator against fake images.

    For the fake images we are hoping the discriminator predicted them
    as true hence the labels are all ones.

    Smoothing: 
    """
    labels = torch.ones_like(out)
    labels = labels * 0.90 if smooth else labels

    criterion = nn.BCEWithLogitsLoss()
    return criterion(out, labels)


def fake_loss(out):
    """Used for computing loss for discriminator on fake images"""
    labels = torch.zeros_like(out)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(out, labels)
