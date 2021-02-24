import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import data_utils
from glob import glob
from matplotlib import pyplot as plt

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def _load_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('L')
        img = np.expand_dims(1-(np.array(img)/255.).astype('float32'), axis=0)
    return img


class Dataset(data.Dataset):
    def __init__(self, data_dir, num_examples=1_000_000):
        self.data_dir = data_dir
        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, example_index):
        fn_chars = np.random.choice(data_utils.chars, 2)
        fn_indices = [data_utils.dict_char2index[char] for char in fn_chars]
        fn_char_paths = [np.random.choice(glob(f'{self.data_dir}/{char}/*.png')) for char in fn_chars]
        fn_images = [_load_image(fn_char_path) for fn_char_path in fn_char_paths]
        fn_image = np.concatenate(fn_images, axis=2)

        ln_char = np.random.choice(data_utils.lns)
        ln_index = data_utils.dict_char2index[ln_char]
        ln_char_path = np.random.choice(glob(f'{self.data_dir}/{ln_char}/*.png'))
        ln_image = _load_image(ln_char_path)
        image = np.concatenate([ln_image, fn_image], axis=2)
        image = torch.from_numpy(image)

        indices = torch.reshape(torch.Tensor([ln_index, *fn_indices]), (1, 3)).to(torch.long)

        return image.to(device), indices.to(device)

if __name__ == '__main__':
    trainset = Dataset('data/sample', num_examples=1000)
    trainloader = data.DataLoader(trainset, batch_size=32, shuffle=True)

    model = Model(data_utils.char_count).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        running_loss = 0
        for i, (images, indices) in enumerate(trainloader):
            optimizer.zero_grad()

            prediction = model(images)

            loss = criterion(prediction, indices)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1)%10 == 0:
                print(f'Epoch{epoch+1}Step{i+1}: {running_loss/10}')
                running_loss = 0
