from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import albumentations as A

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class LeafDiseaseDataset(Dataset):
    """Leaf Disease Dataset"""

    def __init__(self, csv_file, root_dir, transform = A.Compose([A.RandomCrop(width=600, height=600),
                                                                  A.Resize(width=224, height=224),
                                                                  A.HorizontalFlip(p=0.5),
                                                                  A.RandomBrightnessContrast(p=0.2)])
):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_file.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels_file.iloc[idx, 1]
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = np.divide(image, 255.0).astype('float')
        sample = {"image": image, "label": label}

        return sample

    def showImage(self, idx):
        image = self[idx]['image']
        plt.imshow(image*255)
        plt.show()


def main():
    CSV_PATH = 'Data/train.csv'
    ROOT_PATH = 'Data/train_images'
    dataset = LeafDiseaseDataset(csv_file=CSV_PATH, root_dir=ROOT_PATH)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.7 * len(dataset)),
                                                                         len(dataset) - int(0.7 * len(dataset))])


if __name__ == "__main__":
    main()
