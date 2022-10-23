# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:28:35 2022

@author: sigma
"""
#pip install tensorflow
import tensorflow as tf
import numpy as np
import datetime, os

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')
# Check the version
tf.__version__

# Create a constant tensor A
A = tf.constant([[3, 2],
                 [5, 2]])


print(A)

A.shape
A.dtype
A.ndim
A.numpy()


# Create another tensor B
B = tf.constant([[9, 5],
                 [1, 3]])



# Create a Variable tensor V
V = tf.Variable([[3, 2],
                 [5, 2]])

print(B.numpy())
# Concatenate columns
AB_col_concate = tf.concat(values=[A, B], axis=1)
AB_col_concate.numpy()
# Concatenate rows
AB_row_concate = tf.concat(values=[A, B], axis=0)
AB_row_concate.numpy()
# Tensor filled with zeros
tf_zeros = tf.zeros(shape=[3, 4], dtype=tf.int32)
tf_zeros.numpy()
# Tensor filled with ones
tf_ones = tf.ones(shape=[5, 3], dtype=tf.float32)
tf_ones.numpy()
# Reshape the tensor 
reshaped = tf.reshape(tensor = AB_col_concate, shape = [1, 8])
reshaped.numpy()
# dtype cast
tf.cast(A, tf.float64).numpy()
A.numpy()
# Tranpose tensor
tf.transpose(A).numpy()
# Define vector v
v = tf.constant([[5], [2]])
# Matrix multiplication 
tf.matmul(A, v).numpy()
# Element-wise multiplication
tf.multiply(A, v).numpy()
# Get the rows and columns
rows, cols = (3,3)
# Create identity matrix
tf_identity = tf.eye(num_rows = rows,
                    num_columns = cols,
                    dtype = tf.float64)

tf_identity.numpy()

# Determinant of tensor 
A.numpy()
tf.linalg.det(tf.cast(A, tf.float64)).numpy()

A
B
tf.tensordot(a=A, b=B, axes=1).numpy()
#Gradient calculation
# Initialize a variable
x = tf.Variable(5.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
    y = x ** 3
    
# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 75.0

# Initialize a random value x
yhat = tf.Variable([tf.random.normal([1])])
print(f'Initializing yhat={yhat.numpy()}')

learning_rate = 1e-2                            # learning rate for SGD
result = []
y = 4                                           # Define the target value

# Compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(750):
    with tf.GradientTape() as tape:
        loss = (yhat - y)**2.3 

    # loss minimization using gradient tape
    grad = tape.gradient(loss, yhat)
    new_yhat = yhat - learning_rate*grad        # sgd update
    yhat.assign(new_yhat)                       # update the value of f(x)
    result.append(yhat.numpy()[0])

# Plot the evolution of yhat as we optimize towards y
plt.figure(figsize=(20,10))

plt.plot(result)
plt.plot([0, 750],[y,y])

plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('y value')
plt.title('Evoluation of yhat as we optimize towards y');