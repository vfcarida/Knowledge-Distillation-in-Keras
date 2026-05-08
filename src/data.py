from typing import Tuple
import tensorflow as tf

def load_mnist_data(batch_size: int = 64) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads and preprocesses the Fashion MNIST dataset.

    Args:
        batch_size (int): The batch size for the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the 
        training dataset and the testing dataset.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Scale and reshape
    x_train = x_train.astype("float32").reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 28, 28, 1) / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, test_ds
