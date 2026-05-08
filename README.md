# Knowledge Distillation in Keras

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A professional, production-ready demonstration of **Knowledge Distillation** (KD) for image-based models using TensorFlow/Keras. This repository provides modular, SOLID-compliant code to compress heavy neural networks into smaller, faster models without significant accuracy drops.

This repository accompanies the blog post: [Distilling Knowledge in Neural Networks](https://app.wandb.ai/authors/knowledge-distillation/reports/Distilling-Knowledge-in-Deep-Neural-Networks--VmlldzoyMjkxODk).

## 🧠 Architecture Overview

The codebase is structured following professional Software Engineering standards:

```text
.
├── src/                # Core modular Python package
│   ├── data.py         # Data loading and preprocessing pipelines
│   ├── models.py       # Teacher and Student network factories
│   ├── losses.py       # Knowledge distillation and auxiliary loss functions
│   └── trainer.py      # Custom Keras Model encapsulating KD logic
├── examples/           # Ready-to-run execution scripts
├── notebooks/          # Exploratory Jupyter Notebooks
└── tests/              # Pytest-based unit test suite
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

Clone the repository and install the dependencies. It's recommended to use a virtual environment.

```bash
git clone https://github.com/yourusername/Knowledge-Distillation-in-Keras.git
cd Knowledge-Distillation-in-Keras

# Install the package and dependencies
pip install -e .
```

### Running the Example

We provide a complete, modular example of Knowledge Distillation using the Fashion MNIST dataset.

```bash
python examples/train_mnist.py
```

## 💻 Example Usage (Code)

You can easily integrate our custom KD `Distiller` into your own workflows:

```python
import tensorflow as tf
from src.models import get_teacher_model, get_student_model
from src.trainer import Distiller

# 1. Build and Train Teacher
teacher = get_teacher_model()
# ... train teacher ...

# 2. Build Student
student = get_student_model()

# 3. Setup Distiller
distiller = Distiller(student=student, teacher=teacher, temperature=3.0)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    alpha=0.1
)

# 4. Train Student with Knowledge Distillation
distiller.fit(train_ds, epochs=10, validation_data=test_ds)
```

## 📊 Notebooks

For experimental validation and more complex use cases (like Transfer Learning), explore our notebooks located in the `notebooks/` directory:

- `Distillation_Toy_Example.ipynb` - KD on the MNIST dataset
- `Distillation_with_Transfer_Learning.ipynb` - KD (typical KD loss) on Flowers dataset with a fine-tuned model
- `Distillation_with_Transfer_Learning_MSE.ipynb` - KD (MSE loss) on Flowers dataset
- `Effect_of_Data_Augmentation.ipynb` - Studies the effect of data augmentation on KD

## 🛠️ Running Tests

To ensure everything is working correctly, run the automated test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up your development environment and submit Pull Requests.

## 📝 Acknowledgements
I am grateful to [Aakash Kumar Nain](https://twitter.com/A_K_Nain) for providing valuable feedback on the original code.

<div align="center"><img src="https://i.ibb.co/ZXtwJjV/Webp-net-resizeimage.png" width="100" height="100"></img></div>
