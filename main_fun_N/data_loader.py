import torch
from torchvision import datasets
from torchvision import transforms
import os
import torchvision


def get_loader(args):
    if args.dset == 'mnist':
        tr_transform = transforms.Compose([
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = torchvision.datasets.MNIST(root='../dataset', train=True,download=False, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = torchvision.datasets.MNIST(root='../dataset', train=False,download=False, transform=te_transform)

    elif args.dset == 'fmnist':
        tr_transform = transforms.Compose([
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = torchvision.datasets.FashionMNIST(root='../dataset', train=True,download=False, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = torchvision.datasets.FashionMNIST(root='../dataset', train=False,download=False, transform=te_transform)

    else:
        print("Unknown dataset")
        exit(0)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last=True)

    return train_loader, test_loader
