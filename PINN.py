"""
Code for Physics-Informed Network
This code was written based on a tensorflow 2 version
of the PINN code developed by Ching-Yao Lai and Yongji Wang,
adapted from Raissi et al. (2019).
Adapted by Ryan Eusebi for Eusebi et al. (2023)
"""

from abc import ABC, abstractmethod
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Dict
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer


"""
Define functions for
    1. Create neural network
    2. Loss Function
"""

class TunalbeXavierNormal(Initializer):
    """ Xavier initialization for the multilayer perceptron"""
    def __init__(self, lsnow: float):
        self._lsnow = float(lsnow)

    def __call__(self, shape, dtype, **kwargs):
        xavier_stddev = tf.cast(tf.sqrt(2 / tf.math.reduce_sum(shape)) * self._lsnow, dtype=dtype)
        return tf.Variable(tf.random.truncated_normal(shape, stddev=xavier_stddev, dtype=dtype))

    def get_config(self):
        return {"lsnow": self._lsnow}


def create_mlp(layers: List[int], lyscl: List[int], dtype=tf.float64, input_size=4):
    """Multilayer perceptron for PINN problem"""

    inputs = Input(shape=(input_size,), dtype=dtype)
    dense = Dense(
        layers[0], activation="tanh", dtype=dtype,
        kernel_initializer=TunalbeXavierNormal(lyscl[0]))(inputs)
    for n_unit, stddev in zip(layers[1:-1], lyscl[1:-1]):
        dense = Dense(
            n_unit, activation="tanh", dtype=dtype,
            kernel_initializer=TunalbeXavierNormal(stddev))(dense)
    dense = Dense(
        layers[-1], activation=None, dtype=dtype,
        kernel_initializer=TunalbeXavierNormal(lyscl[-1]))(dense)
    model = Model(inputs=inputs, outputs=dense)
    return model


""" Code for loss function """
class SquareLoss:
    """Calculate square loss from given physics-informed equations and data equations

    Note that Data equation can be used to set boundary conditions too.
    """

    def __init__(self, equations, equations_data, args, gamma: float) -> None:
        """
        Args:
            equations:
                an iterable of callables, with signatrue function(x, neural_net)
            equations_data:
                an iterable of callables, with signatrue function(x, y, neural_net)
            args:
                Array containing the folloing items in this order: x_0,t_0,p_0,u_0,h_0,f,beta 
                 x_0 - value which the x and y storm-relative coordinates are scaled by (units meters)
                 t_0 - scaling for the storm-relative time (in units seconds)
                 p_0 pressure scaling (p_max - p_min, in units Pa)
                 u_0 - horizontal velocity scaling (units m/s)
                 h_0 - geopotential scaling (units m^2 / s^2)
                 f - coriolis parameter at storm center
                 beta - beta parameter (df/dy)
            gamma:
                the normalized weighting factor for equation loss and data loss,
                loss = gamma * equation-loss + (1 - gamma) * data-loss
        """
        self.eqns = equations
        self.eqns_data = equations_data
        self.gamma = gamma
        self.args = args

    def __call__(self, X_eqn, data_pts, weights, net) -> Dict[str, tf.Tensor]:
        y = X_eqn[..., 0:1]
        x = X_eqn[..., 1:2]
        t = X_eqn[..., 2:3]
        p = X_eqn[..., 3:4]
        equations = self.eqns(y=y, x=x, t=t, p=p, args=self.args, neural_net=net)
        X_data, Y_data = data_pts
        y_data = X_data[..., 0:1]
        x_data = X_data[..., 1:2]
        t_data = X_data[..., 2:3]
        p_data = X_data[..., 3:4]
        datas = self.eqns_data(y=y_data, x=x_data, t=t_data, p=p_data, Y=Y_data, neural_net=net)

        loss_e = sum(tf.reduce_mean(tf.square(eqn)) for eqn in equations)
        loss_d = sum(tf.reduce_mean(weights*tf.square(data)) for data in datas)
        loss = (1 - self.gamma) * loss_d + self.gamma * loss_e
        print(loss)
        return {"loss": loss, "loss_equation": loss_e, "loss_data": loss_d}
    
"""
Define Optimization Classes
"""
def _get_indices(trainable_vars):
    # we'll use tf.dynamic_stitch and tf.dynamic_partition,
    # so we need to prepare required information
    shapes = tf.shape_n(trainable_vars)
    count = 0
    stitch_indices = []
    partition_indices = []
    for i, shape in enumerate(shapes):
        n = np.product(shape.numpy())
        stitch_indices.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        partition_indices.extend([i] * n)
        count += n
    partition_indices = tf.constant(partition_indices)
    return partition_indices, stitch_indices


def assign_new_model_parameters(params, training_vars, partition_indices):
    shapes = tf.shape_n(training_vars)
    params = tf.dynamic_partition(params, partition_indices, len(shapes))
    for i, (shape, param) in enumerate(zip(shapes, params)):
        training_vars[i].assign(tf.reshape(param, shape))


class OptimizerBase(ABC):

    def __init__(self, net, loss, collocation_points, data_points, weights) -> None:
        """
        Args:
            net: the nerual network to use
            loss: the losses to use
            collocation_points:
                the x positions for training physics-informed part
            data_points:
                A tuple of (x_data_points, y_data_points),
                x_data_points consist of [x_0, x_1, x_2, ... x_i], the x positions of training data
                y_data_points consist of [y_0, y_1, y_2, ... y_i], the y positions of training data
                set None for forward mode -- bondary condition equations requires no data.
            weights:
                the weights corresponding to each data point, weighting how important it 
                is in the loss function
        """
        self.net = net
        self.loss = loss
        self.colloc_pts = collocation_points
        self.data_pts = data_points if data_points else (None, None)
        self.weights = weights
        self._loss_records = {}
        self.history = []

    @abstractmethod
    def loss_records(self):
        pass

    @abstractmethod
    def optimize(self, nIter: int):
        pass


"""Code for LBFGS optimizer"""
class LBFGS(OptimizerBase):

    @property
    def loss_records(self) -> Dict[str, np.array]:
        return self._loss_records

    @tf.function
    def _single_iteration(self, params, partition_indices, stitch_indices):
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            # this step is critical for self-defined function for L-BFGS
            assign_new_model_parameters(params, self.net.trainable_weights, partition_indices)
            losses = self.loss(X_eqn=self.colloc_pts, data_pts=self.data_pts, weights=self.weights, net=self.net)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(losses["loss"], self.net.trainable_weights)
        grads = tf.dynamic_stitch(stitch_indices, grads)

        # store loss value so we can retrieve later
        tf.py_function(self.history.append, inp=[losses["loss"]], Tout=[])
        return losses, grads

    def optimize(self, nIter: int):
        self._loss_records = {}
        self.start_time = time.time()

        # move this outside the decorator to save the losses
        iteration = tf.Variable(0)
        partition_indices, stitch_indices = _get_indices(self.net.trainable_weights)

        def optimize_func(params):
            # A function that can be used by tfp.optimizer.lbfgs_minimize.
            # This function is created by function_factory.
            # Sub-function under function of class not need to input self
            losses, grads = self._single_iteration(params, partition_indices, stitch_indices)
            iteration.assign_add(1)

            if iteration % 1000 == 0:
                msg = f"LBFGS Iter {iteration.numpy():4d}; "
                for name, loss in losses.items():
                    record = self._loss_records.setdefault(name, [])
                    loss_val = loss.numpy()
                    record.append(loss_val)
                    msg += f"{name}: {loss_val:4e}; "
                print(msg)

            return losses["loss"], grads

        max_nIter = tf.cast(nIter/3, dtype=tf.int32)
        init_params = tf.dynamic_stitch(stitch_indices, self.net.trainable_weights)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=optimize_func,
            initial_position=init_params,
            tolerance=10e-14, max_iterations=max_nIter)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        assign_new_model_parameters(results.position, self.net.trainable_weights, partition_indices)


""" Code for Adam Optimizer """
class Adam(OptimizerBase):

    def __init__(self, net, loss, collocation_points, data_points, weights) -> None:
        super().__init__(net, loss, collocation_points, data_points, weights)
        self._adam = tf.optimizers.Adam()

    @property
    def loss_records(self) -> Dict[str, np.array]:
        return self._loss_records

    @tf.function
    def _single_iteration(self):
        with tf.GradientTape() as tape:
            losses = self.loss(X_eqn=self.colloc_pts, data_pts=self.data_pts, weights=self.weights, net=self.net)
        grads = tape.gradient(losses["loss"], self.net.trainable_weights)
        self._adam.apply_gradients(zip(grads, self.net.trainable_weights))
        return losses

    def optimize(self, nIter: int):
        for it in range(nIter):
            losses = self._single_iteration()

            # Print and store
            if it % 1000 == 0:
                msg = f"Adam Iter {it:4d}; "
                for name, loss in losses.items():
                    record = self._loss_records.setdefault(name, [])
                    loss_val = loss.numpy()
                    record.append(loss_val)
                    msg += f"{name}: {loss_val:4e}; "
                print(msg)
