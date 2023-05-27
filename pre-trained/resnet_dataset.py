# from: https://huggingface.co/datasets/imagenet-1k
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import numpy as np
import os
from resnet_classes import IMAGENET2012_CLASSES

def get_class_str(idx: int):
    return list(IMAGENET2012_CLASSES.items())[idx][1]

class ResnetImageDataset(Dataset):
    def __init__(self, img_dir, img_num_cap=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        img_num_total = len(os.listdir(self.img_dir))

        self.img_num_cap = img_num_cap if (img_num_cap != None) and (img_num_cap < img_num_total) else img_num_total

        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = self.__create_labeled_list()
    
    def __create_labeled_list(self) -> pd.DataFrame:
        dir_content = os.listdir(self.img_dir)
        array_2d = []
        unique_labels = []
        for filename in dir_content:
            stripped_filename = filename.strip(".JPEG")
            split_filename = stripped_filename.split("_")
            key = split_filename[-1]
            if key not in unique_labels:
                unique_labels.append(key)
            label = list(IMAGENET2012_CLASSES.keys()).index(key)
            array_2d.append([filename, label])

            if (len(array_2d) >= self.img_num_cap):
                break
        
        print(f"The dataset contains {len(unique_labels)} unique classes")
        df = pd.DataFrame(array_2d ,columns=["filename", "label"])
        return df


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode= ImageReadMode.RGB)
        label = np.array(self.img_labels.iloc[idx, 1], dtype=float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

if __name__ == "__main__":
    
    from torchvision import transforms

    img_dir = "././data/resnet18_set/images_unpacked"
    
    transform_pipe = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ])
    
    ds = ResnetImageDataset(img_dir=img_dir, transform=transform_pipe)
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=False)
    images, labels = next(iter(dl))
    print(labels[0])
    print(get_class_str(int(labels[0].item())))

    
