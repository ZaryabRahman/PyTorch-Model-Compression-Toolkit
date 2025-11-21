"""Teacher-Student knowledge distillation framework"""

import torch
from torch import nn


class TeacherStudentWrapper(nn.Module):
    """wrap teacher and student models for KD training"""

    def __init__(self, teacher: nn.Module, student: nn.Module, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()  # teacher frozen

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        return student_logits, teacher_logits


class DistillationLoss(nn.Module):
    """combine cross-entropy and KL-divergence distillation loss"""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        ce = self.ce_loss(student_logits, targets)
        kl = self.kl_loss(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        return self.alpha * ce + (1 - self.alpha) * kl
