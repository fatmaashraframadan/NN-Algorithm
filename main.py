import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random as rand

# todo 1: read training data (input and output vectors) [ i think we can use lists]
# todo 2: create neurons                                                                    -----DONE
# todo for program 1:
#           todo a: Normalize features                                                      ----DONE
#           todo b: calculate values at each neuron (a-s)                                   ----DONE
#           todo c: calculate MSE                                                           ----DONE
#           todo d: back propagation algorithm and equation                                 ----DONE
#           todo e: gradient descent                                                        ----DONE
#           todo f: stop when reaching an acceptable MSE [[[OR]]] 500 itr                   ----DONE
#           todo g: print MSE
#           todo h: save final weights to file

# todo for program 2:
#           todo a: read input file
#           todo b: read file from program 1
#           todo c: apply feedforward propagation
#           todo d: calculate MSE
#           todo e: print MSE


# feed-forward Propagation
def to_hidden(n_neurons, x_values, example, weights):
    a =[]
    for j in range(n_neurons):
        sum_ax = 0
        for i in range(x_values.shape[1]-1):
            ax = x_values.loc[example, i] * weights[i, j]
            sum_ax += ax
        a.append(activation(sum_ax))
    return a


def to_output(n_neuron, a_values, weights):
    y = []
    for j in range(n_neuron):
        sum_ax = 0
        for i in range(hidden_neurons):
            ax = a_values[i] * weights[i, j]
            sum_ax += ax
        y.append(activation(sum_ax))
    return y


def activation(sum):
    return 1 / (1 + np.math.exp(-sum))


# feed_backward Propagation
def delta_output(a_values, training_data, example,  input_neurons):
    delta_values = []
    for col in range(input_neurons, out_neurons+input_neurons):
        delta = (a_values[col-input_neurons] - training_data.loc[example, col]) * (a_values[col-input_neurons] * (1 - a_values[col-input_neurons]))
        delta_values.append(delta)
    return delta_values


def delta_hidden(a_values, weights_out, delta_out):
    delta_values = []
    for j in range(hidden_neurons):
        delta = 0
        for k in range(input_neurons, input_neurons+out_neurons):
            delta += delta_out[k-input_neurons] * weights_out[j, k-input_neurons]
        delta = delta * (a_values[j] * (1 - a_values[j]))
        delta_values.append(delta)
    return delta_values


# update weights
def out_update(old_weights, learning_rate, delta, a_values):
    for k in range(len(delta)):
        current_delta = delta[k]
        for j in range(len(a_values)):
            current_a = a_values[j]
            old_w = old_weights[j, k]
            new_w = old_w - learning_rate * current_delta * current_a
            old_weights[j, k] = new_w
    return old_weights


def hidden_update(old_weights, learning_rate, example, delta, a_values):
    for k in range(len(delta)):
        current_delta = delta[k]
        for j in range(input_neurons):
            current_a = a_values.loc[example, j]
            old_w = old_weights[j, k]
            new_w = old_w - learning_rate * current_delta * current_a
            old_weights[j, k] = new_w
    return old_weights


# calculate MSE
def MSE(training_data, new_out):
    MSE_errors = []
    for neuron in range(input_neurons, input_neurons+out_neurons):
        error = 0
        actual_neurons_values = training_data[neuron]
        for r in range(len(actual_neurons_values)):
            error += (1.0 /2) * pow(actual_neurons_values[r] - new_out[r], 2)
        MSE_errors.append(error)
    training_example_MSE = np.mean(MSE_errors)
    return training_example_MSE


# read neuron sizes
line = pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2)
input_neurons = int(line.iloc[0, 0])
hidden_neurons = int(line.iloc[0, 1])
out_neurons = int(line.iloc[0, 2])
n_examples = int(line.iloc[1,0])

# input training data
training_data = pd.read_csv("train.txt", header=None, delim_whitespace=True, skiprows=2)


#  Data Normalization
file1 = open("mean_values.txt", "w")
file1.write("mean \t std \n")


for i in range(input_neurons):
    neuron_mean = np.mean(training_data[i])
    neuron_sd = np.std(training_data[i])
    file1.write(str(neuron_mean))
    file1.write("\t")
    file1.write(str(neuron_sd))
    file1.write("\n")
    for y in range(training_data.shape[0]):
        training_data.loc[y, i] = (training_data.loc[y, i] - neuron_mean) / neuron_sd


for x in range(input_neurons, input_neurons+out_neurons):
    max_value = np.max(training_data[x])
    min_value = np.min(training_data[x])
    for cell in range(training_data.shape[0]):
        training_data.loc[cell, x] = (training_data.loc[cell, x] - min_value) / (max_value - min_value)


# create 2D arrays for weights randomly
input_hidden_weights = np.random.rand(input_neurons, hidden_neurons)
hidden_out_weights = np.random.rand(hidden_neurons, out_neurons)
learning_rate = 0.2


AK_Predected_values =[]
# update_input_weights_array = []
# update_hidden_weights_array = []

for itr in range(0, 500):
    for example in range(n_examples):
        Aj = to_hidden(hidden_neurons, training_data.loc[[example]], example, input_hidden_weights)
        Ak = to_output(out_neurons, Aj, hidden_out_weights)
        AK_Predected_values.extend(Ak)
        deltak = delta_output(Ak, training_data, example, input_neurons)
        deltaj = delta_hidden(Aj, hidden_out_weights, deltak)
        new_out_weights = out_update(hidden_out_weights, learning_rate, deltak, Aj)
        new_hidden_weights = hidden_update(input_hidden_weights, learning_rate, example, deltaj,
                                           training_data.loc[[example]])
        input_hidden_weights = new_hidden_weights
        hidden_out_weights = new_out_weights
    MSE_value = MSE(training_data, AK_Predected_values)
    print(MSE_value)
    file1.write("iteration : ")
    file1.write(str(itr+1))
    file1.write("\t")
    file1.write(str(MSE_value))
    file1.write("\n")
    AK_Predected_values = []


file1.close()