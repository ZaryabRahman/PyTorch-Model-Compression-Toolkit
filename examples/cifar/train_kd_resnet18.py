"""train student model on cifar10 using knowledge distillation"""

import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from pytorch_model_compression_toolkit.distillation.teacher_student import TeacherStudentWrapper, DistillationLoss

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

teacher = models.resnet18(pretrained=True)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher.eval()

student = models.resnet18(pretrained=False)
student.fc = nn.Linear(student.fc.in_features, 10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher, student = teacher.to(device), student.to(device)

kd_model = TeacherStudentWrapper(teacher, student, temperature=4.0, alpha=0.7).to(device)
loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)
optimizer = SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

epochs = 20
for epoch in range(epochs := 20):
    kd_model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        student_logits, teacher_logits = kd_model(inputs)
        loss = loss_fn(student_logits, teacher_logits, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs} completed')

os.makedirs('./runs/kd_resnet18', exist_ok=True)
torch.save(student.state_dict(), './runs/kd_resnet18/student_model.pt')
print('Student model saved.')
