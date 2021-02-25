import torch
import torch.utils.data as data
from model import Model
from dataloader import Dataset
from data_utils import calc_acc, dict_index2char
import argparse
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import data_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/sample-test')
    parser.add_argument('--num_test_examples', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='./weights/epoch3.pth')
    args = parser.parse_args()

    model = Model(data_utils.char_count).to(device)
    model.load_state_dict(torch.load(args.model_path))

    with torch.no_grad():
        testset = Dataset(args.testset_dir, device=device, num_examples=args.num_test_examples)
        testloader = data.DataLoader(testset, batch_size=args.num_test_examples, shuffle=False)
        for i, (images, labels) in enumerate(testloader):
            preds = model(images)
            results = torch.argmax(preds, dim=1)
            char_acc, name_acc = calc_acc(labels, results)

            print(f'char_acc={char_acc}, name_acc={name_acc}')

        print('The first prediction is', ''.join([data_utils.dict_index2char[index] for index in torch.squeeze(results)[0].cpu().tolist()]))
        plt.imshow(images[0][0].cpu(), cmap='gray')
        plt.show()
