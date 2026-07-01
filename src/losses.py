import tensorflow as tf

def get_kd_loss(student_logits: tf.Tensor, teacher_logits: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    """
    Calculates the Knowledge Distillation loss using Kullback-Leibler divergence.

    Args:
        student_logits (tf.Tensor): The raw output (logits) of the student model.
        teacher_logits (tf.Tensor): The raw output (logits) of the teacher model.
        temperature (float): The temperature to soften the probabilities.

    Returns:
        tf.Tensor: The computed KD loss.
    """
    student_logits = tf.convert_to_tensor(student_logits, dtype=tf.float32)
    teacher_logits = tf.convert_to_tensor(teacher_logits, dtype=tf.float32)
    
    # 1. Softmax with temperature scaling
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(student_logits / temperature, axis=-1)
    
    # Calculate KL Divergence: KL(Teacher || Student)
    # KL = sum(p_teacher * log(p_teacher / p_student))
    eps = 1e-7
    kl_div = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + eps) - tf.math.log(student_probs + eps)), axis=-1)
    
    # Scale by temperature**2 to match the scale of gradients
    kd_loss = tf.reduce_mean(kl_div) * (temperature ** 2)
    return kd_loss

def get_dkd_loss(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
    temperature: float = 3.0
) -> tf.Tensor:
    """
    Calculates the Decoupled Knowledge Distillation (DKD) loss (CVPR 2022).
    Decomposes the distillation loss into Target Class Knowledge Distillation (TCKD)
    and Non-Target Class Knowledge Distillation (NCKD).

    Args:
        student_logits (tf.Tensor): Raw output (logits) of the student model. Shape: (batch_size, num_classes)
        teacher_logits (tf.Tensor): Raw output (logits) of the teacher model. Shape: (batch_size, num_classes)
        labels (tf.Tensor): Ground-truth labels (sparse or one-hot).
        alpha (float): Hyperparameter to weight TCKD loss.
        beta (float): Hyperparameter to weight NCKD loss.
        temperature (float): Temperature for softening probability distributions.

    Returns:
        tf.Tensor: The computed DKD loss scalar.
    """
    student_logits = tf.convert_to_tensor(student_logits, dtype=tf.float32)
    teacher_logits = tf.convert_to_tensor(teacher_logits, dtype=tf.float32)
    
    num_classes = tf.shape(student_logits)[-1]
    
    # Standardize labels to a 1D integer tensor (batch_size,)
    labels = tf.convert_to_tensor(labels)
    labels_shape = tf.shape(labels)
    labels_rank = tf.rank(labels)
    
    # Check if labels are one-hot encoded: rank is 2 and last dim is greater than 1
    is_onehot = tf.logical_and(
        tf.equal(labels_rank, 2),
        tf.greater(labels_shape[-1], 1)
    )
    
    y_true = tf.cond(
        is_onehot,
        lambda: tf.argmax(labels, axis=-1, output_type=tf.int32),
        lambda: tf.cast(labels, tf.int32)
    )
    y_true = tf.reshape(y_true, [-1])

    # 1. Create binary masks for target (gt) class
    gt_mask = tf.one_hot(y_true, depth=num_classes, on_value=True, off_value=False, dtype=tf.bool)
    other_mask = tf.logical_not(gt_mask)
    
    gt_mask_float = tf.cast(gt_mask, tf.float32)
    other_mask_float = tf.cast(other_mask, tf.float32)

    # 2. Compute probabilities
    pred_student = tf.nn.softmax(student_logits / temperature, axis=-1)
    pred_teacher = tf.nn.softmax(teacher_logits / temperature, axis=-1)

    # 3. TCKD (Target Class Knowledge Distillation)
    # Form binary distributions: [p_t, 1 - p_t]
    pt_student = tf.reduce_sum(pred_student * gt_mask_float, axis=-1, keepdims=True)
    pnt_student = tf.reduce_sum(pred_student * other_mask_float, axis=-1, keepdims=True)
    pred_student_tckd = tf.concat([pt_student, pnt_student], axis=-1)

    pt_teacher = tf.reduce_sum(pred_teacher * gt_mask_float, axis=-1, keepdims=True)
    pnt_teacher = tf.reduce_sum(pred_teacher * other_mask_float, axis=-1, keepdims=True)
    pred_teacher_tckd = tf.concat([pt_teacher, pnt_teacher], axis=-1)

    # KL Divergence for TCKD: KL(Teacher || Student)
    eps = 1e-7
    log_pred_student_tckd = tf.math.log(pred_student_tckd + eps)
    log_pred_teacher_tckd = tf.math.log(pred_teacher_tckd + eps)
    tckd_loss = tf.reduce_mean(
        tf.reduce_sum(pred_teacher_tckd * (log_pred_teacher_tckd - log_pred_student_tckd), axis=-1)
    )

    # 4. NCKD (Non-Target Class Knowledge Distillation)
    # Renormalize probabilities by masking out the target class logit using a very negative value.
    # This excludes target class from softmax normalization.
    large_neg = -1e9
    student_logits_nckd = (student_logits / temperature) + (large_neg * gt_mask_float)
    teacher_logits_nckd = (teacher_logits / temperature) + (large_neg * gt_mask_float)

    pred_student_nckd = tf.nn.softmax(student_logits_nckd, axis=-1)
    pred_teacher_nckd = tf.nn.softmax(teacher_logits_nckd, axis=-1)

    log_pred_student_nckd = tf.math.log(pred_student_nckd + eps)
    log_pred_teacher_nckd = tf.math.log(pred_teacher_nckd + eps)
    nckd_loss = tf.reduce_mean(
        tf.reduce_sum(pred_teacher_nckd * (log_pred_teacher_nckd - log_pred_student_nckd), axis=-1)
    )

    # Combine loss and scale by temperature**2
    dkd_loss = (alpha * tckd_loss + beta * nckd_loss) * (temperature ** 2)
    return dkd_loss

