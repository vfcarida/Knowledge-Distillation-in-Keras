import tensorflow as tf
from src.losses import get_kd_loss

def test_kd_loss():
    """Test Knowledge Distillation Loss calculation."""
    # Dummy logits
    teacher_logits = tf.constant([[1.0, 2.0, 3.0]])
    student_logits = tf.constant([[0.5, 2.5, 3.5]])
    
    # Calculate loss
    loss = get_kd_loss(student_logits, teacher_logits, temperature=1.0)
    
    # Shape check
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    
    # Type check
    assert isinstance(loss, tf.Tensor)
