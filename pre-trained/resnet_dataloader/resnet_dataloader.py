# from: https://huggingface.co/datasets/imagenet-1k
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os


from classes import IMAGENET2012_CLASSES
class ResnetDataLoader:
    def __init__():
        pass
    def load_data(path):
        pass

class ResnetImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.array(self.img_labels.iloc[idx, [1,2,3,4]].values, dtype=float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

if __name__ == "__main__":
    img_dir = "././data/resnet18_set/images_unpacked"
    os.listdir(img_dir)
    print("Check")