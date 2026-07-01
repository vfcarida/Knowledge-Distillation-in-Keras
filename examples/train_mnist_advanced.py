"""
Example script demonstrating advanced Knowledge Distillation methods in Keras:
- Vanilla Logits KD
- Decoupled KD (DKD)
- Feature-based KD (FitNets)
- Hybrid KD
"""

import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_mnist_data
from src.trainer import Distiller

def build_named_teacher(input_shape=(28, 28, 1), num_classes=10):
    """Teacher ConvNet with explicitly named layers for feature distillation."""
    model = models.Sequential([
        layers.Conv2D(16, (5, 5), activation="relu", input_shape=input_shape, name="teacher_conv1"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (5, 5), activation="relu", name="teacher_conv2"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="teacher_dense_feature"),
        layers.Dense(num_classes, name="teacher_logits")
    ], name="named_teacher_model")
    return model

def build_named_student(input_shape=(28, 28, 1), num_classes=10):
    """Student MLP with explicitly named layers for feature distillation."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(48, activation="relu", name="student_dense_feature"),
        layers.Dense(num_classes, name="student_logits")
    ], name="named_student_model")
    return model

def main():
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Load Data
    print("Loading Fashion MNIST dataset...")
    train_ds, test_ds = load_mnist_data(batch_size=128)
    # Use a subset of data for quick demonstration
    train_ds_small = train_ds.take(50)
    test_ds_small = test_ds.take(10)

    # 2. Build and Train Teacher Model
    print("Building and training teacher model...")
    teacher = build_named_teacher()
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    # Train teacher for 2 epochs for quick demonstration
    teacher.fit(train_ds_small, epochs=2, validation_data=test_ds_small)
    teacher_acc = teacher.evaluate(test_ds_small, verbose=0)[1]
    print(f"Teacher Model Accuracy: {teacher_acc:.4f}\n")

    # 3. Vanilla Knowledge Distillation
    print("-" * 50)
    print("1. Running Vanilla Knowledge Distillation...")
    student_vanilla = build_named_student()
    distiller_vanilla = Distiller(
        student=student_vanilla,
        teacher=teacher,
        temperature=3.0,
        distillation_type="vanilla"
    )
    distiller_vanilla.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.1
    )
    distiller_vanilla.fit(train_ds_small, epochs=2, validation_data=test_ds_small)
    vanilla_acc = distiller_vanilla.evaluate(test_ds_small, verbose=0)[1]
    print(f"Vanilla KD Student Accuracy: {vanilla_acc:.4f}\n")

    # 4. Decoupled Knowledge Distillation (DKD)
    print("-" * 50)
    print("2. Running Decoupled Knowledge Distillation (DKD)...")
    student_dkd = build_named_student()
    distiller_dkd = Distiller(
        student=student_dkd,
        teacher=teacher,
        temperature=3.0,
        distillation_type="dkd",
        alpha_dkd=1.0,
        beta_dkd=2.0
    )
    distiller_dkd.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.1
    )
    distiller_dkd.fit(train_ds_small, epochs=2, validation_data=test_ds_small)
    dkd_acc = distiller_dkd.evaluate(test_ds_small, verbose=0)[1]
    print(f"DKD Student Accuracy: {dkd_acc:.4f}\n")

    # 5. Feature-based Knowledge Distillation (FitNets)
    print("-" * 50)
    print("3. Running Feature-based Knowledge Distillation...")
    student_feat = build_named_student()
    distiller_feat = Distiller(
        student=student_feat,
        teacher=teacher,
        distillation_type="feature",
        student_hint_layers=["student_dense_feature"],
        teacher_guided_layers=["teacher_dense_feature"]
    )
    distiller_feat.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.5,
        gamma=1.0
    )
    distiller_feat.fit(train_ds_small, epochs=2, validation_data=test_ds_small)
    feat_acc = distiller_feat.evaluate(test_ds_small, verbose=0)[1]
    print(f"Feature-based KD Student Accuracy: {feat_acc:.4f}\n")

    # 6. Hybrid Knowledge Distillation (Logits + Feature)
    print("-" * 50)
    print("4. Running Hybrid Knowledge Distillation...")
    student_hybrid = build_named_student()
    distiller_hybrid = Distiller(
        student=student_hybrid,
        teacher=teacher,
        temperature=3.0,
        distillation_type="hybrid",
        student_hint_layers=["student_dense_feature"],
        teacher_guided_layers=["teacher_dense_feature"]
    )
    distiller_hybrid.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.1,
        gamma=1.0
    )
    distiller_hybrid.fit(train_ds_small, epochs=2, validation_data=test_ds_small)
    hybrid_acc = distiller_hybrid.evaluate(test_ds_small, verbose=0)[1]
    print(f"Hybrid KD Student Accuracy: {hybrid_acc:.4f}\n")

    print("=" * 50)
    print("Summary of Final Student Accuracies:")
    print(f"Teacher Model Accuracy:          {teacher_acc:.4f}")
    print(f"Vanilla KD Student Accuracy:     {vanilla_acc:.4f}")
    print(f"DKD Student Accuracy:            {dkd_acc:.4f}")
    print(f"Feature-based KD Accuracy:       {feat_acc:.4f}")
    print(f"Hybrid KD Student Accuracy:      {hybrid_acc:.4f}")

if __name__ == "__main__":
    main()
