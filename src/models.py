import tensorflow as tf
from tensorflow.keras import models, layers

def get_teacher_model(input_shape: tuple = (28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """
    Builds a shallow ConvNet teacher model.

    Args:
        input_shape (tuple): The shape of the input tensor.
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The constructed teacher model.
    """
    model = models.Sequential([
        layers.Conv2D(16, (5, 5), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes)
    ], name="teacher_model")
    return model

def get_student_model(input_shape: tuple = (28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """
    Builds a smaller student model.

    Args:
        input_shape (tuple): The shape of the input tensor.
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The constructed student model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(48, activation="relu"),
        layers.Dense(num_classes)
    ], name="student_model")
    return model
