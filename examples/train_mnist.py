"""
Example script: Training a Knowledge Distillation model on Fashion MNIST.
"""

import sys
import os
import tensorflow as tf

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_mnist_data
from src.models import get_teacher_model, get_student_model
from src.trainer import Distiller

def main():
    # 1. Load Data
    print("Loading data...")
    train_ds, test_ds = load_mnist_data(batch_size=64)

    # 2. Build Models
    print("Building models...")
    teacher = get_teacher_model()
    student = get_student_model()

    # 3. Train Teacher
    # In a real scenario, you might load a pre-trained teacher
    print("Training teacher model...")
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    # Train teacher for a few epochs for demonstration
    teacher.fit(train_ds, epochs=3, validation_data=test_ds)
    teacher.evaluate(test_ds)

    # 4. Distillation (Train Student)
    print("Training student model with Knowledge Distillation...")
    distiller = Distiller(student=student, teacher=teacher, temperature=3.0)
    
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.1
    )

    # Distill teacher to student
    distiller.fit(train_ds, epochs=3, validation_data=test_ds)

    print("Evaluating distilled student model...")
    distiller.evaluate(test_ds)

if __name__ == "__main__":
    main()
