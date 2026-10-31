# python imports
import os
import argparse

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision.transforms as transforms

from dataloader import SVHN
from student_code import LeNet, train_model, test_model


SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1980, 0.2010, 0.1970]


def save_checkpoint(state, is_best,
                    file_folder="./outputs/",
                    filename='checkpoint.pth.tar'):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.makedirs(os.path.expanduser(file_folder), exist_ok=True)
    torch.save(state, os.path.join(file_folder, filename))
    if is_best:
        state.pop('optimizer', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def build_train_transform(augmentation="none"):
    transform_list = [transforms.ToTensor()]
    if augmentation == "rotation":
        transform_list.append(transforms.RandomRotation(10))
    elif augmentation == "affine":
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    elif augmentation == "colorjitter":
        transform_list.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        )
    transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    transform_list.append(transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD))
    return transforms.Compose(transform_list)


def build_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD),
    ])


def main(args):
    torch.manual_seed(0)

    train_transform = build_train_transform(augmentation=args.augmentation)
    test_transform = build_test_transform()

    data_folder = './data'
    if not os.path.exists(data_folder):
        os.makedirs(os.path.expanduser(data_folder), exist_ok=True)

    train_set = SVHN(root=data_folder, split="train", transform=train_transform)
    test_set = SVHN(root=data_folder, split="test", transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    print(f'Loaded trainset: {len(train_loader)}')
    print(f'Loaded testset: {len(test_loader)}')

    model = LeNet(num_classes=10)
    training_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{:s}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}, acc {:0.2f})".format(
                args.resume, checkpoint['epoch'], 100 * best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    print("Training the model ...\n")
    for epoch in range(start_epoch, args.epochs):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        acc = test_model(model, test_loader, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': max(best_acc, acc),
            'optimizer': optimizer.state_dict(),
        }, (acc > best_acc))
        best_acc = max(best_acc, acc)
    print("Finished Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVHN classification using Pytorch')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='number of images within a mini-batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--augmentation', default='none',
                        choices=['none', 'rotation', 'affine', 'colorjitter'],
                        help='training augmentation: none, rotation, affine, or colorjitter')
    args = parser.parse_args()
    main(args)
