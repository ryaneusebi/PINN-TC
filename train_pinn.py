# Author: Ryan Eusebi
# Date: 11/20/2023
# Code for training a PINN using PINN.py and equations.py

# Code is incomplete, requires the user to create their
# own data and collocation points, along with some other 
# parameters

# Refer to readme.txt for more information

import sys
import time
import numpy as np
import tensorflow as tf
from numpy import cos, sin

from PINN import create_mlp, SquareLoss, Adam, LBFGS
from equations import Inverse_1stOrder_Equations, Data_Equations


# fix seed to remove randomness across multiple runs
# np.random.seed(234)
# tf.random.set_seed(234)

# Data type for calculation
_data_type = tf.float64

def to_tensor(var):
    return tf.constant(var, dtype=_data_type)

def train():
    """
    Define Constants and Normalization Factors
    Set a whole bunch of parameters
    """

    omega = 7.2921e-5 # earth angular velocity
    R_e = 6378100 # Earth's radius in meters
    lat = 20 # latitude of storm center
    f = 2 * omega * sin(np.radians(lat)) # coriolis parameter
    beta = 2 * omega * cos(np.radians(lat)) / R_e # beta value
    u_0 = 60 # m/s, maximum value for scaling velocity
    x_0 = 400000 #m domain width to scale x and y to -1 to 1
    h_max = # max geopotential
    h_min = # min geopotential
    h_0 = h_max - h_min # scaling parameter for gepotential #m^2 / s^2
    t_0 = 60*60*6 # time scale by 6 hours to get time to -1 to 1
    p_min = 150 # min pressure of system
    p_max = 900 # max pressure of system
    p_0 = (p_max - p_min)*100 # pressure scaling parameter units Pa

    # set the bounds of the domain (X)
    lx = np.array([-x_0, -x_0, -t_0, p_min*100])
    ux = np.array([x_0, x_0, t_0, p_max*100])

    # set the bounds for the target fields (Y)
    ly = np.array([-u_0, -u_0, h_min])
    uy = np.array([u_0, u_0, h_max])


    ######################################################################
    # Create and train PINN
    
    
    # Paramters
    num_iterations_adam = 20000
    num_iterations_lbfgs = 180000
    gamma = 0.99
    num_hlayers = 4
    layer_size = 50
    input_size = 4

    layers = [layer_size for i in range(num_hlayers)]
    layers.append(4) # output layer
    lyscl = [1 for i in range(len(layers))]

    X_data, Y_data = # data points
    collocation_pts = # collocation points

    # scale the X and Y values of the data points and
    # collocation points so everything is between -1 and 1
    X_data = (X_data - lx)/(ux - lx) * 2 - 1
    Y_data = (Y_data - ly)/(uy -ly) * 2 - 1
    (collocation_pts - lx)/(ux - lx) * 2 - 1


    weights = # the relative weights of each data point in PINN training
    weights = weights/np.mean(weights)

    # Arguments that will be used by the PINN equations
    args = x_0,t_0,p_0,u_0,h_0,f,beta

    # Create the MLP, the function for the equations, and the loss function
    model = create_mlp(layers, lyscl, dtype=_data_type, input_size=input_size)
    equations = Inverse_1stOrder_Equations()
    loss = SquareLoss(equations=equations, equations_data=Data_Equations, args=args, gamma=gamma)

   

    # train the model for Adam optimizer first, then LBFGS
    start_time = time.time()
    adam = Adam(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights = to_tensor(weights)
    )
    
    adam.optimize(nIter=num_iterations_adam)
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights=to_tensor(weights),
    )

    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed) # Print training time

    # Access loss history
    total_losses = np.array(adam.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))
    equation_losses = np.array(adam.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", []))
    data_losses = np.array(adam.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))

    # Add code to save your model if desired

if __name__ == '__main__':
    train()