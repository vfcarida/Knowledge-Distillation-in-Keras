import tensorflow as tf
from typing import Dict, Any

from src.losses import get_kd_loss

class Distiller(tf.keras.Model):
    """
    Custom Keras Model encapsulating the distillation training logic.
    """

    def __init__(self, student: tf.keras.Model, teacher: tf.keras.Model, temperature: float = 0.5):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
        # Ensure the teacher is not trained
        self.teacher.trainable = False

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: list,
        student_loss_fn: tf.keras.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha: float = 0.1,
        **kwargs
    ):
        """
        Configure the distiller for training.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student predictions and ground-truth
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """
        Custom training step for knowledge distillation.
        """
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Calculate losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = get_kd_loss(
                student_logits=student_predictions,
                teacher_logits=teacher_predictions,
                temperature=self.temperature
            )
            
            # Combine losses
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_predictions))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """
        Custom evaluation step.
        """
        x, y = data

        y_prediction = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, y_prediction)

        self.compiled_metrics.update_state(y, tf.nn.softmax(y_prediction))

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
