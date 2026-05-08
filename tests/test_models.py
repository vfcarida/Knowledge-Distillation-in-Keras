import tensorflow as tf
from src.models import get_teacher_model, get_student_model

def test_teacher_model_output_shape():
    """Test if teacher model returns expected output shape."""
    model = get_teacher_model(input_shape=(28, 28, 1), num_classes=10)
    
    dummy_input = tf.random.normal((1, 28, 28, 1))
    output = model(dummy_input)
    
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    assert model.name == "teacher_model"

def test_student_model_output_shape():
    """Test if student model returns expected output shape."""
    model = get_student_model(input_shape=(28, 28, 1), num_classes=10)
    
    dummy_input = tf.random.normal((1, 28, 28, 1))
    output = model(dummy_input)
    
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    assert model.name == "student_model"
