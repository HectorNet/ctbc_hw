import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import data_utils
from dataloader import Dataset
from model import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset_dir', type=str, default='data/sample-train')
    parser.add_argument('--testset_dir', type=str, default='data/sample-test')
    parser.add_argument('--num_train_examples', type=int, default=1000)
    parser.add_argument('--num_test_examples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--tiny', action='store_true', default=False)
    args = parser.parse_args()

    trainset = Dataset(args.trainset_dir, device=device, num_examples=args.num_train_examples)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = Dataset(args.testset_dir, device=device, num_examples=args.num_test_examples)
    testloader = data.DataLoader(testset, batch_size=args.num_test_examples, shuffle=False)

    model = Model(data_utils.char_count, tiny=args.tiny).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            preds = model(images)
            results = torch.argmax(preds, dim=1)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1)%args.log_freq == 0:
                char_acc, name_acc = data_utils.calc_acc(labels, results)
                print(f'Train Epoch{epoch+1}/{args.epochs}: loss={running_loss/args.log_freq:.3f}, char_acc={char_acc:.2f}, name_acc={name_acc:.2f}')
                running_loss = 0

                with torch.no_grad():
                    for i, (images, labels) in enumerate(testloader):
                        preds = model(images)
                        results = torch.argmax(preds, dim=1)
                        char_acc, name_acc = data_utils.calc_acc(labels, results)
                        print(f'Test Epoch{epoch+1}/{args.epochs}: char_acc={char_acc:.2f}, name_acc={name_acc:.2f}')

        if (epoch+1)%args.save_freq == 0:
            if args.tiny:
                path = f'./weights/tiny/epoch{epoch+1}.pth'
            else:
                path = f'./weights/epoch{epoch+1}.pth'
            torch.save(model.state_dict(), path)
            print(f'Model {path} saved.')
