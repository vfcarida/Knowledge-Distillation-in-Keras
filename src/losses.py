import tensorflow as tf

def get_kd_loss(student_logits: tf.Tensor, teacher_logits: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    """
    Calculates the Knowledge Distillation loss.

    Args:
        student_logits (tf.Tensor): The raw output (logits) of the student model.
        teacher_logits (tf.Tensor): The raw output (logits) of the teacher model.
        temperature (float): The temperature to soften the probabilities.

    Returns:
        tf.Tensor: The computed KD loss.
    """
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    
    # We use temperature**2 to scale the gradients appropriately
    kd_loss = tf.compat.v1.losses.softmax_cross_entropy(
        teacher_probs, 
        student_logits / temperature, 
        weights=temperature**2
    )
    return kd_loss
