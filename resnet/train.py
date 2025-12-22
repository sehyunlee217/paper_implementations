import torch
import torchvision
import torchvision.transforms as transforms
from model_cifar10 import ResNetCifar10
from torch import nn, optim
from tqdm import tqdm

import wandb


def train(model_name=None, config=None):
    """
    Training code for ResNet
    """
    # default config for ResNet-20 on CIFAR-10
    if config is None:
        model_name = "resnet20-cifar10"
        config = {
            "N": 3,
            "num_channels": [16, 32, 64],
            "lr": 0.1,
            "epochs": 100,
            "batch_size": 128,
            "weight_decay": 1e-4,
        }

    wandb.init(project=model_name, config=config)

    config = wandb.config
    num_layers = [1 + 2 * config.N, 2 * config.N, 2 * config.N]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    # dataset normalization code from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4, drop_last=False
    )

    model = ResNetCifar10(num_blocks=num_layers, num_channels=config.num_channels).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay
    )

    """
    Paper: We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations,
    torchvision provides 60K dataset split into train/test -> change LR at 30K & 40K isntead.
    Train: 50K / 128 ~ 391 iter per epoch -> 30K / 391 iter ~ epoch 77 | 40K / 391 iter ~ epoch 103
    """

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[77, 103], gamma=0.1
    )

    for epoch in range(1, config.epochs + 1):
        # train
        print(f"Epoch {epoch} training...")
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # test
        model.eval()
        correct, total, test_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = (
                    inputs.to(device),
                    targets.to(device),
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        accuracy = 100.0 * correct / total

        scheduler.step()

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss / len(train_loader),
                "test/loss": test_loss / len(test_loader),
                "test/accuracy": accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch}: Loss {train_loss / len(train_loader):.4f} | Acc: {accuracy:.2f}%"
        )

    wandb.finish()


if __name__ == "__main__":
    model_names = [
        "resnet20-cifar10",
        "resnet20-cifar32",
        "resnet20-cifar44",
        "resnet20-cifar56",
    ]
    configs = [
        {
            "N": 3,
            "num_channels": [16, 32, 64],
            "lr": 0.1,
            "epochs": 128,
            "batch_size": 128,
            "weight_decay": 1e-4,
        },
        {
            "N": 5,
            "num_channels": [16, 32, 64],
            "lr": 0.1,
            "epochs": 128,
            "batch_size": 128,
            "weight_decay": 1e-4,
        },
        {
            "N": 7,
            "num_channels": [16, 32, 64],
            "lr": 0.1,
            "epochs": 128,
            "batch_size": 128,
            "weight_decay": 1e-4,
        },
        {
            "N": 9,
            "num_channels": [16, 32, 64],
            "lr": 0.1,
            "epochs": 128,
            "batch_size": 128,
            "weight_decay": 1e-4,
        },
    ]

    for model_name, config in zip(model_names, configs):
        print(f"Training {model_name} with {config}")
        train(model_name, config)
