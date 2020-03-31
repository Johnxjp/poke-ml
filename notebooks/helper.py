import os
import sys
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import models, transforms

ROOT_FOLDER = ".."

if ROOT_FOLDER not in sys.path:
    sys.path.append(ROOT_FOLDER)
    
from poke_ml.loader import (
    load_index_table,
    load_image,
    load_raw_data,
    IndexDataset,
    ImageDataset
)
from poke_ml.utils import image_preprocessing, scale, generate_random
from poke_ml.gan.generators import CGAN, DCGenerator
from poke_ml.gan.discriminators import CDim, DCDiscriminator
from poke_ml.gan.losses import real_loss, fake_loss