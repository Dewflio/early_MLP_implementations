# from: https://huggingface.co/datasets/imagenet-1k
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os
from collections import OrderedDict


from classes import IMAGENET2012_CLASSES
class ResnetDataLoader:
    def __init__():
        pass
    def load_data(path):
        pass

class ResnetImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = self.__create_labeled_list()
    
    def __create_labeled_list(self) -> pd.DataFrame:
        dir_content = os.listdir(self.img_dir)   
        array_2d = []
        for filename in dir_content:
            stripped_filename = filename.strip(".JPEG")
            split_filename = stripped_filename.split("_")
            key = split_filename[-1]
            label = list(IMAGENET2012_CLASSES.keys()).index(key)
            array_2d.append([filename, label])
        df = pd.DataFrame(array_2d ,columns=["filename", "label"])
        return df


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.array(self.img_labels.iloc[idx, 1], dtype=float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

if __name__ == "__main__":
    img_dir = "././data/resnet18_set/images_unpacked"
    
    ds = ResnetImageDataset(img_dir=img_dir)
