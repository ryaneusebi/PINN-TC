Author: Ryan Eusebi

Contact email: reusebi@caltech.edu

Date: 11/20/2023

Please contact me if you are using this code and have any questions!

This code was written based on a tensorflow 2 version of the PINN code developed by Ching-Yao Lai and Yongji Wang. Note that the original PINN code from Raissi et al. (2019) was written in tensorflow 1 and available at: https://github.com/maziarraissi/PINNs

Information for running the PINN code in this repo:

The file 'PINN.py' contains code for creating a physics-informed 
neural network, training a neural network, and creating a loss
function to be used while training the neural network. This code is
general in the sense that it is not equation dependent - any set of
equations can be supplied to it through the loss function defined by 
the function SquareLoss(). The neural network structure can also be
modified through the inputs to creat_mlp() function.

The differential equations for the PINN, the collocation points, and
the data points must be provided to the SquareLoss() function. Two 
optimization functions are included in the 'PINN.py' code: one which
uses the LBFGS algorithm, and another which uses the ADAM algorithm.
It is recommended to use the ADAM algorithm for the first subset of
iterations, and then finish the training with LBFGS. In our examples
in the paper, we used 20,000 iterations of ADAM first, followed by
180,000 iterations of LBFGS.

The 'equations.py' file contains two functions. Data_equations()
handles calculating the errors in the PINN output at the data points
compared to the ground truth observations provided to the function. 
The Inverse_1stOrder_Equations() function calculates all the relevant
derivative terms needed for your chosen PDEs and calculates the equation
residuals at your collocation points. The physical equations in this section can be modified to suit your needs.

The "train_pinn.py" provides a nearly complete file for training your own
PINN given the provided equations in equations.py (which can be modified). 
The file requires the user to add code to get their own collocation and data points, and the weights for the data points. The user should also adjust all of the parameters to best suit their problem domain, such as all of the scaling parameters. The user can also adjust the training iterations and the neural network structure.

The file tf2_gpu_env.yml is the conda environment file that was used when running all of this code.


