"""
Author: Zaryab Rahman
Data: 21-11-2025  <<>>>>><<<<<<>>>>> Pretty bored dddddd
"""


"""train resnet18 on cifar10 with quantization-aware training (qat) example.

usage:
    python examples/cifar/train_qat_resnet18.py --data-dir ./data --epochs 20 --batch-size 128 --save-dir ./runs/qAT_resnet18
"""

import argparse
import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from pytorch_model_compression_toolkit.quantization.fake_quant import MinMaxObserver
from pytorch_model_compression_toolkit.quantization.qat import prepare_qat_model, QATTrainer, enable_qat, disable_qat, convert_qat_model


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(num_classes: int = 10, pretrained: bool = False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, preds = torch.max(outputs, dim=1)
    return (preds == targets).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--save-dir', type=str, default='./runs/qat_resnet18')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, num_workers=args.num_workers)

    model = build_model(num_classes=10, pretrained=False)

    model = prepare_qat_model(model, observer_ctor=MinMaxObserver, num_bits=8, per_channel=False, symmetric=False)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    trainer = QATTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=torch.device(args.device), max_epochs=args.epochs, save_dir=args.save_dir)

    enable_qat(model)

    trainer.train(train_loader, val_loader=test_loader, epochs=args.epochs)

    disable_qat(model)

    model = convert_qat_model(model)

    model_path = os.path.join(args.save_dir, 'model_converted.pt')
    torch.save(model.state_dict(), model_path)

    print(f'model saved to: {model_path}')


if __name__ == '__main__':
    main()
