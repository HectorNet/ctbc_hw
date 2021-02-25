import torch
import torch.utils.data as data
from model import Model
from dataloader import Dataset
from data_utils import calc_acc, dict_index2char
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import data_utils


if __name__ == '__main__':
    model = Model(data_utils.char_count, tiny=True)
    model.load_state_dict(torch.load('./weights/tiny/epoch40.pth'))


    num_examples = 2
    with torch.no_grad():
        trainset = Dataset('data/sample', device=device, num_examples=num_examples)
        trainloader = data.DataLoader(trainset, batch_size=num_examples, shuffle=True)
        for i, (images, labels) in enumerate(trainloader):
            preds = model(images)
            results = torch.argmax(preds, dim=1)
            char_acc, name_acc = calc_acc(labels, results)

            results = torch.squeeze(results)
            labels = torch.squeeze(labels)

            print(labels.shape, results.shape)
            print([dict_index2char[index] for index in labels[0].tolist()])
            print([dict_index2char[index] for index in results[0].tolist()])
            print([dict_index2char[index] for index in labels[1].tolist()])
            print([dict_index2char[index] for index in results[1].tolist()])

            print(f'char_acc={char_acc}, name_acc={name_acc}')
