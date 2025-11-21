import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from pytorch_model_compression_toolkit.pruning.unstructured import UnstructuredPruner
from pytorch_model_compression_toolkit.pruning.metrics import print_sparsity_summary

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True)


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

pruner = UnstructuredPruner(model, amount=0.2)

epochs = 20
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    # prune after each epoch
    pruner.step()
    print(f'Epoch {epoch+1} completed.')
    print_sparsity_summary(model)

os.makedirs('./runs/pruning_resnet18', exist_ok=True)
torch.save(model.state_dict(), './runs/pruning_resnet18/model_pruned.pt')
print('pruned model saved.')
