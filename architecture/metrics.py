import tensorflow as tf
import numpy as np 


class PearsonCorr(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__(dtype=tf.float32)
        self.total_over_batches = self.add_weight(name="total", initializer="zeros")
        self.num_batches = self.add_weight(name="count", initializer="zeros")

    def update(self, predicted_flow, true_flow):
        p = predicted_flow - tf.reduce_mean(predicted_flow)
        t = true_flow - tf.reduce_mean(true_flow)

        skibidi = tf.reduce_sum(p * t)
        ohio = tf.math.sqrt(tf.reduce_sum(tf.math.square(p)) * tf.reduce_sum(tf.math.square(t))) 
        gyatt = skibidi/ohio

        self.total_over_batches.assign_add(gyatt)
        self.num_batches.assign_add(1.0)
        return skibidi/ohio

    def result(self):
        return self.total_over_batches / self.num_batches

class MarginAccuracy(tf.keras.metrics.Metric):
    def __init__(self, margin=5.0):
        super().__init__(dtype=tf.float32)
        self.margin = margin
        self.num_correct = self.add_weight(initializer="zeros")
        self.total = self.add_weight(initializer="zeros")

    def update_state(self, predicted_flow, true_flow):
        """
        this is gauging accuracy by determinign whether difference is less than 5% 
        of the true_flow
        """
        dif = tf.abs(true_flow - predicted_flow)
        floor = self.margin * true_flow
        mask = tf.cast(dif <= floor, tf.float32)
        self.num_correct.assign_add(tf.reduce_sum(mask))
        self.total.assign_add(tf.shape(predicted_flow)[0])

    def result(self):
        return self.num_correct / self.total

"""

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse", 
    metrics=[PearsonCorr(), WithinMarginAccuracy(margin=2.0)] 
)
"""