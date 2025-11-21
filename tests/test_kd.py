import torch
from torch import nn
from pytorch_model_compression_toolkit.distillation.teacher_student import TeacherStudentWrapper, DistillationLoss


def build_models():
    teacher = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    student = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10))
    return teacher, student


def test_teacher_student_forward():
    teacher, student = build_models()
    wrapper = TeacherStudentWrapper(teacher, student)
    x = torch.randn(8, 32)
    student_logits, teacher_logits = wrapper(x)
    assert student_logits.shape == (8, 10)
    assert teacher_logits.shape == (8, 10)


def test_distillation_loss():
    teacher, student = build_models()
    wrapper = TeacherStudentWrapper(teacher, student)
    loss_fn = DistillationLoss()
    x = torch.randn(8, 32)
    targets = torch.randint(0, 10, (8,))
    student_logits, teacher_logits = wrapper(x)
    loss = loss_fn(student_logits, teacher_logits, targets)
    assert loss.item() > 0
