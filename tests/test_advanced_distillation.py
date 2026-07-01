import tensorflow as tf
import pytest
import numpy as np

from src.losses import get_kd_loss, get_dkd_loss
from src.trainer import Distiller

def test_get_kd_loss():
    """Test standard KL-divergence-based KD loss with dummy outputs."""
    student_logits = tf.constant([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]])
    teacher_logits = tf.constant([[1.8, 1.2, 0.2], [0.4, 2.8, 1.5]])
    
    loss = get_kd_loss(student_logits, teacher_logits, temperature=2.0)
    
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()
    assert loss.numpy() >= 0.0

def test_get_dkd_loss_sparse_labels():
    """Test DKD loss with sparse label indices."""
    student_logits = tf.constant([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]])
    teacher_logits = tf.constant([[1.8, 1.2, 0.2], [0.4, 2.8, 1.5]])
    labels = tf.constant([0, 1])  # Sparse integer labels
    
    loss = get_dkd_loss(student_logits, teacher_logits, labels, alpha=1.0, beta=2.0, temperature=2.0)
    
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()
    assert loss.numpy() >= 0.0

def test_get_dkd_loss_onehot_labels():
    """Test DKD loss with one-hot encoded labels."""
    student_logits = tf.constant([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]])
    teacher_logits = tf.constant([[1.8, 1.2, 0.2], [0.4, 2.8, 1.5]])
    labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # One-hot labels
    
    loss = get_dkd_loss(student_logits, teacher_logits, labels, alpha=1.0, beta=2.0, temperature=2.0)
    
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()
    assert loss.numpy() >= 0.0

def test_distiller_vanilla_mode():
    """Test Distiller with Vanilla KD (standard logits distillation)."""
    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(2)
    ], name="student")
    
    teacher = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(2)
    ], name="teacher")
    
    distiller = Distiller(student=student, teacher=teacher, temperature=2.0, distillation_type="vanilla")
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.2
    )
    
    # Run a single training step
    x = tf.random.normal((4, 4))
    y = tf.constant([0, 1, 0, 1])
    
    history = distiller.train_step((x, y))
    
    assert "student_loss" in history
    assert "distillation_loss" in history
    assert "total_loss" in history
    assert history["total_loss"] >= 0.0

def test_distiller_dkd_mode():
    """Test Distiller with Decoupled KD (DKD) mode."""
    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(3)
    ], name="student")
    
    teacher = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(3)
    ], name="teacher")
    
    distiller = Distiller(
        student=student, 
        teacher=teacher, 
        temperature=2.0, 
        distillation_type="dkd",
        alpha_dkd=1.5,
        beta_dkd=3.0
    )
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.3
    )
    
    x = tf.random.normal((4, 4))
    y = tf.constant([0, 2, 1, 0])
    
    history = distiller.train_step((x, y))
    
    assert "student_loss" in history
    assert "distillation_loss" in history
    assert "total_loss" in history

def test_distiller_feature_mode_dense():
    """Test Feature Distillation using intermediate Dense layer output projection."""
    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, name="student_dense"),
        tf.keras.layers.Dense(2)
    ], name="student")
    
    teacher = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, name="teacher_dense"),
        tf.keras.layers.Dense(2)
    ], name="teacher")
    
    distiller = Distiller(
        student=student,
        teacher=teacher,
        distillation_type="feature",
        student_hint_layers=["student_dense"],
        teacher_guided_layers=["teacher_dense"]
    )
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.5,
        gamma=2.0
    )
    
    x = tf.random.normal((4, 4))
    y = tf.constant([0, 1, 0, 1])
    
    history = distiller.train_step((x, y))
    
    assert "student_loss" in history
    assert "feature_loss" in history
    assert history["feature_loss"] >= 0.0

def test_distiller_hybrid_mode_cnn():
    """Test Hybrid Distillation with CNN layer output projection and resizing."""
    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(2, (3, 3), padding="same", name="student_conv"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)
    ], name="student")
    
    teacher = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(4, (3, 3), padding="same", name="teacher_conv"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)
    ], name="teacher")
    
    distiller = Distiller(
        student=student,
        teacher=teacher,
        temperature=3.0,
        distillation_type="hybrid",
        student_hint_layers=["student_conv"],
        teacher_guided_layers=["teacher_conv"]
    )
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha=0.1,
        gamma=1.0
    )
    
    x = tf.random.normal((4, 8, 8, 1))
    y = tf.constant([0, 1, 0, 1])
    
    # Run training step
    history = distiller.train_step((x, y))
    
    assert "student_loss" in history
    assert "distillation_loss" in history
    assert "feature_loss" in history
    assert "total_loss" in history
