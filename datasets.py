import torchvision
import math
import torch
from torch.utils.data import DataLoader, Dataset

class NoisyCleantDataset(Dataset):
    """
    Dataset for loading clean and noisy images.
    This is the dataset usef for supervised training
    """
    def __init__(self, img_files, noise_var, chunk_size):
        self.img_files = img_files
        self.noise_var = noise_var
        self.chunk_size = chunk_size
        self.chunks_clean, self.chunks_noisy = self.get_clean_and_noisy_chunks()

    def get_clean_and_noisy_chunks(self):
        chunks_clean = []
        chunks_noisy = []

        # iterate over all images
        for img_file in self.img_files:
            im_grayscale = torchvision.io.read_image(img_file, torchvision.io.image.ImageReadMode.GRAY).float()
            im_grayscale = im_grayscale / 255.0
            im_grayscale = im_grayscale.squeeze(0)
            size_y, size_x = im_grayscale.shape

            chunks_y = size_y // self.chunk_size
            chunks_x = size_x // self.chunk_size

            for y in range(chunks_y):
                for x in range(chunks_x):
                    chunk = im_grayscale[
                        y * self.chunk_size : (y + 1) * self.chunk_size,
                        x * self.chunk_size : (x + 1) * self.chunk_size,
                    ]
                    chunks_clean.append(chunk)
                    chunks_noisy.append(chunk + torch.randn_like(chunk) * math.sqrt(self.noise_var))
        
        return chunks_clean, chunks_noisy

    def __len__(self):
        return len(self.chunks_clean)

    def __getitem__(self, idx):
        return self.chunks_noisy[idx].unsqueeze(0), self.chunks_clean[idx].unsqueeze(0)
    

class NoisyNoisyDataset(Dataset):
    """
    Dataset for loading two versions of same noisy image.
    This is the dataset usef for self-supervised training
    """
    def __init__(self, img_files, noise_var, chunk_size):
        self.img_files = img_files
        self.noise_var = noise_var
        self.chunk_size = chunk_size
        self.chunks_noisy_1, self.chunks_noisy_2 = self.get_chunks()

    def get_chunks(self):
        chunks_noisy_1 = []
        chunks_noisy_2 = []

        # iterate over all images
        for img_file in self.img_files:
            im_grayscale = torchvision.io.read_image(img_file, torchvision.io.image.ImageReadMode.GRAY).float()
            im_grayscale = im_grayscale / 255.0
            im_grayscale = im_grayscale.squeeze(0)
            size_y, size_x = im_grayscale.shape

            chunks_y = size_y // self.chunk_size
            chunks_x = size_x // self.chunk_size

            for y in range(chunks_y):
                for x in range(chunks_x):
                    chunk = im_grayscale[
                        y * self.chunk_size : (y + 1) * self.chunk_size,
                        x * self.chunk_size : (x + 1) * self.chunk_size,
                    ]
                    chunks_noisy_1.append(chunk +  torch.randn_like(chunk) * math.sqrt(self.noise_var))
                    chunks_noisy_2.append(chunk + torch.randn_like(chunk) * math.sqrt(self.noise_var))
        
        return chunks_noisy_1, chunks_noisy_2

    def __len__(self):
        return len(self.chunks_noisy_1)

    def __getitem__(self, idx):
        return self.chunks_noisy_2[idx].unsqueeze(0), self.chunks_noisy_1[idx].unsqueeze(0)