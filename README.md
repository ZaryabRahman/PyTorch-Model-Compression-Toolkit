# PyTorch Model Compression Toolkit

![License](https://img.shields.io/github/license/ZaryabRahman/PyTorch-Model-Compression-Toolkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch)
![GitHub stars](https://img.shields.io/github/stars/ZaryabRahman/PyTorch-Model-Compression-Toolkit?style=social)

A flexible, research-oriented toolkit for effective deep neural network **compression** using advanced **pruning** and **quantization** techniques in PyTorch. Designed for both experimenters and practitioners aiming to deploy more efficient, faster, and lighter deep learning models.

---

> **Why compress your models?**  
> Model compression yields faster inference, smaller file sizes, lower latency, and unlocks deep learning for edge devices and real-world deployment.

---

## ✨ Features

- **Structured & Unstructured Pruning**  
  Remove weights/channels globally or layer-wise.
- **Quantization Ready**  
  Code structure supports easy integration of quantization workflows.
- **Experiment Configuration**  
  All runs and hyperparameters defined via easy-to-modify YAML files.
- **Plug & Play Examples**  
  Out-of-the-box scripts for pruning, evaluation, and benchmarking.
- **Modular & Extensible Core**  
  Add your own pruning or quantization strategies.
- **Tests for Reliability**  
  Thoroughly tested code to avoid breaking research updates.

---

## 🚀 Installation

Requires Python ≥ 3.8 and PyTorch ≥ 1.9

```bash
git clone https://github.com/ZaryabRahman/PyTorch-Model-Compression-Toolkit.git
cd PyTorch-Model-Compression-Toolkit
pip install -r requirements.txt
```

> **Tip:** Want to contribute?  
> Install in editable mode:  
> `pip install -e .[dev]`

---

## 🏁 Quick Start

### 1. Prune a ResNet18 on CIFAR-10

```bash
python examples/prune_model.py --config configs/pruning_resnet18_cifar10.yaml
```

- Customize the pruning strategy and parameters in the YAML config.
- Results, logs, and checkpoints are saved automatically.

---

## 🗂️ Repository Structure

```
PyTorch-Model-Compression-Toolkit/
│
├── configs/      # Experiment configs (YAML)
├── examples/     # Scripts to run and benchmark
├── src/          # Source code (pruning, quantization, engines, etc.)
├── tests/        # Unit & integration tests
├── LICENSE
└── README.md
```

---

## 📈 Real Experiment Results: ResNet18 on CIFAR-10

Our toolkit was thoroughly tested with ResNet18 on CIFAR-10. The following results are obtained using our exact example scripts and default configs in this repository.

| Model     | Dataset  | Compression Method   | Compression Ratio | Top-1 Accuracy (Original) | Top-1 Accuracy (Compressed) | Speedup (CPU) | Speedup (GPU) |
|-----------|----------|---------------------|-------------------|--------------------------|-----------------------------|---------------|---------------|
| ResNet18  | CIFAR-10 | Structured Pruning  |        4×         |      94.8%               |         94.4%               |     2.7×      |     2.1×      |

- **All experiments performed on a Ryzen 5700G, RTX 3060, PyTorch 2.0.0, CUDA 11.8.**
- **Pruning configurations available in `/configs/pruning_resnet18_cifar10.yaml`.**
- **Full logs and training curves are provided in `/examples/logs/`.**
- **Run exactly as described for fully reproducible results.**

**Highlights:**
- ResNet18 compressed to 25% of its original size with only a 0.4% top-1 accuracy drop.
- Over 2× speedup on both CPU and GPU.

---


---

## 📂 Data & Logs

Experiment logs and weights are available in the `/examples/logs/` directory. For custom benchmarks or questions, [open an issue](https://github.com/ZaryabRahman/PyTorch-Model-Compression-Toolkit/issues).

---

## 🤝 Contributing

Feel free to contribute!
- Add a new feature, fix a bug, or improve docs.
- Pull requests are welcome against the `main` branch.

> **See:** [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon!)

---

## 📜 Citing

```
@software{PyTorchMC_Toolkit,
  author = {Zaryab Rahman},
  title = {PyTorch Model Compression Toolkit},
  url = {https://github.com/ZaryabRahman/PyTorch-Model-Compression-Toolkit},
  year = {2025}
}
```

---

## 🙏 Acknowledgements

Built with [PyTorch](https://pytorch.org/) and inspired by leading research in neural network compression.

---

## 📧 Contact

Questions, want to collaborate, or need help?  
[Open an issue](https://github.com/ZaryabRahman/PyTorch-Model-Compression-Toolkit/issues) or email: zaryabrahman848@gmail.com

---

_Serialize your nets, deploy on anything!_
