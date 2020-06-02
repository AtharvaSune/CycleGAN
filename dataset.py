# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import torch
import random
from PIL import Image
from torchvision.transforms import transforms as T


# %%
class ImageDataset(torch.utils.data.Dataset):
    def __inti__(self, root, shuffle=True):
        self.transforms = T.Compose([
            T.Resize(256, 256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])

        self.filesA = os.listdir(os.path.join(root, '/A', '/train'))
        self.filesB = os.listdir(os.path.join(root, '/B', '/train'))
        if shuffle:
            random.shuffle(self.filesA)
            random.shuffle(self.filesB)
    
    def __len__(self):
        return max(len(self.filesA, self.filesB))

    def __getitem__(self, index):
        A = self.transforms(Image.open(self.filesA[index%len(self.filesA)]))
        B = self.transforms(Image.open(self.filesB[index%len(self.filesB)]))

        return (A, B)


# %%


