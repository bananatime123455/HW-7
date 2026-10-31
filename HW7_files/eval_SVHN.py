# python imports
import os
import time
import argparse

# torch imports
import torch

# helper functions for computer vision
import torchvision.transforms as transforms

from dataloader import SVHN
from student_code import LeNet, test_model


SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1980, 0.2010, 0.1970]


def main(args):
    torch.manual_seed(0)

    model = LeNet(num_classes=10)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD),
    ])

    data_folder = './data'
    os.makedirs(os.path.expanduser(data_folder), exist_ok=True)
    test_set = SVHN(root=data_folder, split="test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False)

    if not args.load:
        args.load = "./outputs/model_best.pth.tar"

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{:s}'".format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
            args.load, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.load))
        return

    print("Evaluting the model ...\n")
    start = time.time()
    test_model(model, test_loader, epoch - 1)
    end = time.time()
    print("Evaluation took {:0.2f} sec".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVHN evaluation using Pytorch')
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
