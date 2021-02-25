import torch
import torch.utils.data as data
from model import Model
from dataloader import Dataset
from data_utils import calc_acc, dict_index2char
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import data_utils


if __name__ == '__main__':
    model = Model(data_utils.char_count)
    model.load_state_dict(torch.load('./weights/epoch3.pth'))


    num_examples = 10
    with torch.no_grad():
        testset = Dataset('data/sample', device=device, num_examples=num_examples)
        testloader = data.DataLoader(testset, batch_size=num_examples, shuffle=True)
        for i, (images, labels) in enumerate(testloader):
            preds = model(images)
            results = torch.argmax(preds, dim=1)
            char_acc, name_acc = calc_acc(labels, results)

            print(f'char_acc={char_acc}, name_acc={name_acc}')
