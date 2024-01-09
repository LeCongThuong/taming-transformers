import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import PrintPaths, NumpyDepthPaths 


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class Woodblock2D(CustomBase):
    def __init__(self, size, training_images_list_file): 
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = PrintPaths(paths=paths, size=size)

class Woodblock3D(CustomBase):
    def __init__(self, size, training_images_list_file): 
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = NumpyDepthPaths(paths=paths, size=size)


