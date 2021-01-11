import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random as rand

# todo 1: read training data (input and output vectors) [ i think we can use lists]
# todo 2: create neurals                                                                    -----DONE
# todo for program 1:
#           todo a: Normalize features                                                      ----DONE
#           todo b: calculate values at each neural (a-s)                                   ----DONE
#           todo c: calculate MSE
#           todo d: back propagation algorithm and equation
#           todo e: gradient descent
#           todo f: stop when reaching an acceptable MSE [[[OR]]] 500 itr
#           todo g: print MSE
#           todo h: save final weights to file

# todo for program 2:
#           todo a: read input file
#           todo b: read file from program 1
#           todo c: apply feedforward propagation
#           todo d: calculate MSE
#           todo e: print MSE



# feed-forward Propagation
def to_hidden(x_values, weights):
    a =[]
    for j in range(hidden_neurals):
        sum_ax = 0
        for i in range (x_values):
            ax = x_values[i] * weights[j, [i]]
            sum_ax += ax
        a.append(activation(sum_ax))
    return a


def to_output(a_values, weights):
    y = []
    for j in range(out_neurals):
        sum_ax = 0
        for i in range(hidden_neurals):
            ax = a_values[i] * weights[j, [i]]
            sum_ax += ax
        y.append(activation(sum_ax))
    return y


def activation(sum):
    return 1 / (1 + np.math.exp(-sum))


# feed_backward Propagation
def delta_output(a_values, training_data, input_neurals):
    delta_values = []
    for col in range(input_neurals, len(training_data)):
        for row in range(training_data.shape[0]):
            delta = (a_values[col] - training_data.loc[row, col]) * (a_values[col]* (1-a_values[col]))
            delta_values.append(delta)
    return delta_values


def delta_hidden(a_values, weights_out, delta_out):
    delta_values = []
    for j in range(hidden_neurals):
        delta = 0
        for k in range(input_neurals):
            delta += delta_out[j]*weights_out[k,j]
        delta = delta * (a_values[j] * (1 - a_values[j]))
        delta_values.append(delta)
    return delta_values


# update weights
def update(old_weights, learning_rate, delta, a_values):
    for k in range(len(delta)):
        current_delta = delta[k]
        for j in range(len(a_values)):
            current_a = a_values[j]
            old_w = old_weights[k, [j]]
            new_w = old_w - learning_rate* current_delta * current_a
            old_weights[k,[j]] = new_w
    return old_weights


# calculate MSE
def MSE(training_data, new_out):
    MSE_errors =[]
    for neural in range(input_neurals, out_neurals):
        accurate = 0
        actual_neural_values = training_data[neural]
        for i in range (len(actual_neural_values)):
            if actual_neural_values[i] == new_out[i]:
                accurate += 1
        error = accurate / len(actual_neural_values) * 100
        MSE_errors.append(error)

#         error += (1.0 / 2) * pow(ao[k] - y[k], 2)


# read neural sizes
line = pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2)
input_neurals = line.iloc[0, 0]
hidden_neurals = line.iloc[0, 1]
out_neurals = line.iloc[0, 2]
n_examples = line.iloc[1,0]

# input training data
training_data = pd.read_csv("train.txt", header=None, delim_whitespace=True, skiprows=2)


#  Data Normalization
file1 = open("mean_values.txt", "w")
file1.write("mean \t std \n")


for i in range(input_neurals):
    neural_mean = np.mean(training_data[i])
    neural_sd = np.std(training_data[i])
    file1.write(str(neural_mean))
    file1.write("\t")
    file1.write(str(neural_sd))
    file1.write("\n")
    for y in range(training_data.shape[0]):
        training_data.loc[y, i] = (training_data.loc[y, i] - neural_mean) / neural_sd


# create 2D arrays for weights randomly
input_hidden_weights = np.random.rand(int(input_neurals), int(hidden_neurals))
hidden_out_weights = np.random.rand(int(hidden_neurals), int(out_neurals))
print(input_hidden_weights)
# print(input_hidden_weights[0, [0]])
print("out hidden")
print(hidden_out_weights)
learning_rate = 0.3


for example in range(n_examples):
    for input in range(input_neurals):
        Aj = to_hidden(training_data, input_hidden_weights)
        Ak = to_output(Aj, hidden_out_weights)
        MSE_value = MSE(training_data, Ak)
        deltak = delta_output(Ak, training_data, input_neurals)
        deltaj = delta_hidden(Aj, hidden_out_weights, deltak)
        new_out_weights = update(hidden_out_weights, learning_rate, deltak, Ak)
        new_hidden_weights = update(input_hidden_weights, learning_rate, deltaj, Aj)

file1.close()