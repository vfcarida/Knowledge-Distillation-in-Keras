import tensorflow as tf
from typing import Dict, Any, List

from src.losses import get_kd_loss, get_dkd_loss

class Distiller(tf.keras.Model):
    """
    Custom Keras Model encapsulating advanced knowledge distillation training logic.
    Supports Vanilla KD, Decoupled KD (DKD), Feature-based KD, and Hybrid KD.
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 3.0,
        distillation_type: str = "vanilla",
        student_hint_layers: List[str] = None,
        teacher_guided_layers: List[str] = None,
        alpha_dkd: float = 1.0,
        beta_dkd: float = 2.0
    ):
        """
        Initializes the Distiller.

        Args:
            student (tf.keras.Model): The student model to be trained.
            teacher (tf.keras.Model): The pre-trained teacher model.
            temperature (float): Temperature for softening predictions.
            distillation_type (str): One of "vanilla", "dkd", "feature", "hybrid".
            student_hint_layers (List[str]): Student layer names to extract features from.
            teacher_guided_layers (List[str]): Teacher layer names to extract features from.
            alpha_dkd (float): TCKD loss multiplier (only for DKD).
            beta_dkd (float): NCKD loss multiplier (only for DKD).
        """
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
        # Validate distillation type
        valid_types = ["vanilla", "dkd", "feature", "hybrid"]
        if distillation_type not in valid_types:
            raise ValueError(f"distillation_type must be one of {valid_types}, got '{distillation_type}'")
        self.distillation_type = distillation_type
        
        self.alpha_dkd = alpha_dkd
        self.beta_dkd = beta_dkd

        # Ensure teacher model is not trainable
        self.teacher.trainable = False

        # Set up layer lists for feature-based distillation
        self.student_hint_layers = student_hint_layers or []
        self.teacher_guided_layers = teacher_guided_layers or []

        if len(self.student_hint_layers) != len(self.teacher_guided_layers):
            raise ValueError("student_hint_layers and teacher_guided_layers must have the same length.")

        # Placeholders for feature models
        self.student_feature_model = None
        self.teacher_feature_model = None

        # Build feature extraction sub-models
        if self.student_hint_layers or self.teacher_guided_layers:
            def _get_model_output(model):
                try:
                    return model.output
                except (AttributeError, ValueError):
                    return model.layers[-1].output

            def _get_model_inputs(model):
                try:
                    if hasattr(model, 'inputs') and model.inputs is not None:
                        return model.inputs
                except (AttributeError, ValueError):
                    pass
                return model.layers[0].input

            if self.student_hint_layers:
                self.student_feature_model = tf.keras.Model(
                    inputs=_get_model_inputs(self.student),
                    outputs=[_get_model_output(self.student)] + [self.student.get_layer(name).output for name in self.student_hint_layers]
                )

            if self.teacher_guided_layers:
                self.teacher_feature_model = tf.keras.Model(
                    inputs=_get_model_inputs(self.teacher),
                    outputs=[_get_model_output(self.teacher)] + [self.teacher.get_layer(name).output for name in self.teacher_guided_layers]
                )

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: list,
        student_loss_fn: tf.keras.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        alpha: float = 0.1,
        gamma: float = 1.0,
        **kwargs
    ):
        """
        Configure the distiller for training.

        Args:
            optimizer: Keras optimizer for the student weights.
            metrics: Keras metrics for evaluation.
            student_loss_fn: Loss function of difference between student predictions and ground-truth.
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn (for logits distillation).
            gamma: weight to feature_loss_fn (for feature distillation).
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha
        self.gamma = gamma

    def _project_features(self, student_feats: List[tf.Tensor], teacher_feats: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Dynamically projects student feature maps to match teacher feature map shapes.
        Constructs projection layers (Conv2D or Dense) on the first execution.
        """
        if not hasattr(self, 'proj_layers'):
            self.proj_layers = {}

        projected_student_feats = []
        for i, (sf, tf_feat) in enumerate(zip(student_feats, teacher_feats)):
            key = f"proj_{i}"
            if key not in self.proj_layers:
                sf_shape = sf.shape
                tf_shape = tf_feat.shape
                
                # Check spatial dimensions and channel counts
                if len(sf_shape) == 4 and len(tf_shape) == 4:
                    if sf_shape[-1] != tf_shape[-1]:
                        # 1x1 Convolution to align channel dimension
                        layer = tf.keras.layers.Conv2D(
                            filters=tf_shape[-1],
                            kernel_size=(1, 1),
                            padding="same",
                            use_bias=False,
                            name=f"student_proj_conv_{i}"
                        )
                        layer.build(sf_shape)
                        self.proj_layers[key] = layer
                    else:
                        self.proj_layers[key] = None
                elif len(sf_shape) == 2 and len(tf_shape) == 2:
                    if sf_shape[-1] != tf_shape[-1]:
                        # Dense layer to align fully connected dimensions
                        layer = tf.keras.layers.Dense(
                            units=tf_shape[-1],
                            use_bias=False,
                            name=f"student_proj_dense_{i}"
                        )
                        layer.build(sf_shape)
                        self.proj_layers[key] = layer
                    else:
                        self.proj_layers[key] = None
                else:
                    self.proj_layers[key] = None

            proj_layer = self.proj_layers[key]
            sf_proj = proj_layer(sf) if proj_layer is not None else sf

            # Bilinear interpolation for height/width alignment in CNN layers
            if len(sf_proj.shape) == 4 and len(tf_feat.shape) == 4:
                if sf_proj.shape[1] != tf_feat.shape[1] or sf_proj.shape[2] != tf_feat.shape[2]:
                    sf_proj = tf.image.resize(sf_proj, size=tf_feat.shape[1:3])

            projected_student_feats.append(sf_proj)

        return projected_student_feats

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """
        Custom training step implementing advanced knowledge distillation.
        """
        x, y = data

        # 1. Forward pass of teacher
        if self.teacher_feature_model is not None:
            teacher_outputs = self.teacher_feature_model(x, training=False)
            teacher_predictions = teacher_outputs[0]
            teacher_feats = teacher_outputs[1:]
        else:
            teacher_predictions = self.teacher(x, training=False)
            teacher_feats = []

        with tf.GradientTape() as tape:
            # 2. Forward pass of student
            if self.student_feature_model is not None:
                student_outputs = self.student_feature_model(x, training=True)
                student_predictions = student_outputs[0]
                student_feats = student_outputs[1:]
            else:
                student_predictions = self.student(x, training=True)
                student_feats = []

            # 3. Calculate losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Compute logit distillation loss
            distillation_loss = tf.constant(0.0)
            if self.distillation_type == "vanilla":
                distillation_loss = get_kd_loss(
                    student_logits=student_predictions,
                    teacher_logits=teacher_predictions,
                    temperature=self.temperature
                )
            elif self.distillation_type == "dkd":
                distillation_loss = get_dkd_loss(
                    student_logits=student_predictions,
                    teacher_logits=teacher_predictions,
                    labels=y,
                    alpha=self.alpha_dkd,
                    beta=self.beta_dkd,
                    temperature=self.temperature
                )

            # Compute feature distillation loss
            feature_loss = tf.constant(0.0)
            if self.distillation_type in ["feature", "hybrid"] and len(student_feats) > 0:
                projected_student_feats = self._project_features(student_feats, teacher_feats)
                for tf_feat, sf_proj in zip(teacher_feats, projected_student_feats):
                    feature_loss += tf.reduce_mean(tf.square(tf_feat - sf_proj))
                feature_loss = feature_loss / tf.cast(len(teacher_feats), tf.float32)

            # 4. Combine losses depending on distillation mode
            if self.distillation_type in ["vanilla", "dkd"]:
                loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss
            elif self.distillation_type == "feature":
                loss = self.alpha * student_loss + self.gamma * feature_loss
            elif self.distillation_type == "hybrid":
                loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss + self.gamma * feature_loss
            else:
                loss = student_loss

        # 5. Compute gradients
        trainable_vars = self.student.trainable_variables
        # Include variables of projection layers if any
        if hasattr(self, 'proj_layers'):
            for layer in self.proj_layers.values():
                if layer is not None:
                    trainable_vars = trainable_vars + layer.trainable_variables
                    
        gradients = tape.gradient(loss, trainable_vars)

        # 6. Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 7. Update metrics
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_predictions))

        # Return metric logs
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
            "total_loss": loss
        })
        return results

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """
        Custom evaluation step.
        """
        x, y = data

        # Forward pass of teacher
        if self.teacher_feature_model is not None:
            teacher_outputs = self.teacher_feature_model(x, training=False)
            teacher_predictions = teacher_outputs[0]
            teacher_feats = teacher_outputs[1:]
        else:
            teacher_predictions = self.teacher(x, training=False)
            teacher_feats = []

        # Forward pass of student
        if self.student_feature_model is not None:
            student_outputs = self.student_feature_model(x, training=False)
            student_predictions = student_outputs[0]
            student_feats = student_outputs[1:]
        else:
            student_predictions = self.student(x, training=False)
            student_feats = []

        # Calculate losses
        student_loss = self.student_loss_fn(y, student_predictions)
        
        distillation_loss = tf.constant(0.0)
        if self.distillation_type == "vanilla":
            distillation_loss = get_kd_loss(
                student_logits=student_predictions,
                teacher_logits=teacher_predictions,
                temperature=self.temperature
            )
        elif self.distillation_type == "dkd":
            distillation_loss = get_dkd_loss(
                student_logits=student_predictions,
                teacher_logits=teacher_predictions,
                labels=y,
                alpha=self.alpha_dkd,
                beta=self.beta_dkd,
                temperature=self.temperature
            )

        feature_loss = tf.constant(0.0)
        if self.distillation_type in ["feature", "hybrid"] and len(student_feats) > 0:
            projected_student_feats = self._project_features(student_feats, teacher_feats)
            for tf_feat, sf_proj in zip(teacher_feats, projected_student_feats):
                feature_loss += tf.reduce_mean(tf.square(tf_feat - sf_proj))
            feature_loss = feature_loss / tf.cast(len(teacher_feats), tf.float32)

        # Combine evaluation losses
        if self.distillation_type in ["vanilla", "dkd"]:
            loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss
        elif self.distillation_type == "feature":
            loss = self.alpha * student_loss + self.gamma * feature_loss
        elif self.distillation_type == "hybrid":
            loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss + self.gamma * feature_loss
        else:
            loss = student_loss

        self.compiled_metrics.update_state(y, tf.nn.softmax(student_predictions))

        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
            "total_loss": loss
        })
        return results

    def call(self, inputs, training=False):
        """
        Delegates the model call to the student model.
        """
        return self.student(inputs, training=training)

