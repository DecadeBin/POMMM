Our code can train a single-layer vit model and cnn+fc, to classify the mnist and fmnist datasets, with optional methods involving matrix multiplication:
1、Standard matrix multiplication
2、Normalised light matrix multiplication
3、8bit quantised matrix multiplication
4、8bit quantised light matrix multiplication

The dataset is replaced in the main.py file on line 41, change the 'default' parameter

Model replacement at the beginning of the main.py file
The vit model is used:
from solver import Solver
from model_optical_matrix_wo_norm import batch
The cnn+fc model is used:
from solver_2fc import Solver
from model_2fc import batch

Matrix multiplication replacement in model_optical_matrix_wo_norm.py and model_2fc.py at the beginning def quantum_matrix_mutiply_fc_optical(input,weight):
Replacement of standard vit, normalised quantit and 8bit quantit by adjusting whether the weights are non-negative or not, the quantisation method, and the matrix multiplication method.