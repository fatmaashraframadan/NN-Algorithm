import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random as rand


# todo 1: read training data (input and output vectors) [ i think we can use lists]         ----DONE
# todo 2: create neurons                                                                    ----DONE
# todo for program 1:
#           todo a: Normalize features                                                      ----DONE
#           todo b: calculate values at each neuron (a-s)                                   ----DONE
#           todo c: calculate MSE                                                           ----DONE
#           todo d: back propagation algorithm and equation                                 ----DONE
#           todo e: gradient descent                                                        ----DONE
#           todo f: stop when reaching an acceptable MSE [[[OR]]] 500 itr                   ----DONE
#           todo g: print MSE                                                               ----DONE
#           todo h: save final weights to file                                              ----DONE

# todo for program 2:
#           todo a: read input file                                                         ----DONE
#           todo b: read file from program 1                                                ----DONE
#           todo c: apply feedforward propagation                                           ----DONE
#           todo d: calculate MSE                                                           ----DONE
#           todo e: print MSE                                                               ----DONE


# feed-forward Propagation
def to_hidden(n_neurons, x_values, example, weights):
    a = []
    for j in range(n_neurons):
        sum_ax = 0
        for i in range(x_values.shape[1] - out_neurons):
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
def delta_output(a_values, training_data, example, input_neurons):
    delta_values = []
    for col in range(input_neurons, out_neurons + input_neurons):
        delta = (a_values[col - input_neurons] - training_data.loc[example, col]) * (
                    a_values[col - input_neurons] * (1 - a_values[col - input_neurons]))
        delta_values.append(delta)
    return delta_values


def delta_hidden(a_values, weights_out, delta_out):
    delta_values = []
    for j in range(hidden_neurons):
        delta = 0
        for k in range(input_neurons, input_neurons + out_neurons):
            delta += delta_out[k - input_neurons] * weights_out[j, k - input_neurons]
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
            new_w = old_w - (learning_rate * current_delta * current_a)
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
    for row in range(len(training_data)):
        error = 0
        for neuron in range(input_neurons, input_neurons+out_neurons):
            error += (1.0 / 2) * pow(training_data.iloc[row, neuron] - new_out[row], 2)
        MSE_errors.append(error)
    training_example_MSE = np.mean(MSE_errors)
    return training_example_MSE


# Part2:  Testing
def testing():
    # input testing data
    testing_data = pd.read_csv("train.txt", header=None, delim_whitespace=True, skiprows=2)

    # input weights
    # testing_input_hidden_weights = np.loadtxt('weights.txt', skiprows= 2*input_neurons+2,max_rows=input_neurons)
    # testing_hidden_out_weights = np.loadtxt('weights.txt', skiprows=2*input_neurons+2+input_neurons, usecols=out_neurons-1)
    testing_input_hidden_weights = input_hidden_weights
    testing_hidden_out_weights = hidden_out_weights

    #  Testing Input Data Normalization
    for i in range(input_neurons):
        neuron_mean = np.loadtxt("means.txt", skiprows=i, max_rows=1)
        neuron_sd = np.loadtxt("standards.txt", skiprows=i, max_rows=1)
        for y in range(testing_data.shape[0]):
            testing_data.loc[y, i] = (testing_data.loc[y, i] - neuron_mean) / neuron_sd

    #  Testing Output Data Normalization
    for x in range(input_neurons, input_neurons + out_neurons):
        max_value = np.loadtxt("max.txt", skiprows=x - input_neurons, max_rows=1)
        min_value = np.loadtxt("min.txt", skiprows=x - input_neurons, max_rows=1)
        for cell in range(testing_data.shape[0]):
            testing_data.loc[cell, x] = (testing_data.loc[cell, x] - min_value) / (max_value - min_value)

    # Testing Data
    AK_testing_Predected_values = []
    for testing_example in range(len(testing_data)):
        # calculate forward propagation
        Aj_testing = to_hidden(hidden_neurons, testing_data.loc[[testing_example]], testing_example,
                               testing_input_hidden_weights)
        Ak_testing = to_output(out_neurons, Aj_testing, testing_hidden_out_weights)
        AK_testing_Predected_values.extend(Ak_testing)

    MSE_value = MSE(testing_data, AK_testing_Predected_values)
    print("testing MSE value : ")
    print(MSE_value)

    # Denormalize Output Data
    denormalized_Predected_values = []
    counter = 0
    for row in range(len(testing_data)):
        for x in range(input_neurons, input_neurons + out_neurons):
            max_value = np.loadtxt("max.txt", skiprows=x - input_neurons, max_rows=1)
            min_value = np.loadtxt("min.txt", skiprows=x - input_neurons, max_rows=1)
            denormalized_Predected_values.append(
                (AK_testing_Predected_values[counter] + min_value) * (max_value - min_value))
            counter += 1



# training phase:
# read neuron sizes
line = pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2)
input_neurons = int(line.iloc[0, 0])
hidden_neurons = int(line.iloc[0, 1])
out_neurons = int(line.iloc[0, 2])
n_examples = int(line.iloc[1, 0])

# input training data
training_data = pd.read_csv("train.txt", header=None, delim_whitespace=True, skiprows=2)

# initialize output files
f_means = open("means.txt", "a")
f_standards = open("standards.txt", "a")
f_max = open("max.txt", "a")
f_min = open("min.txt", "a")
f_weights = open("weights.txt", "a")


#  Input Data Normalization
for i in range(input_neurons):
    neuron_mean = np.mean(training_data[i])
    neuron_sd = np.std(training_data[i])
    f_means.write('%7.9f' % neuron_mean)
    f_means.write('\n')
    f_standards.write('%7.9f' % neuron_sd)
    f_standards.write('\n')
    for y in range(training_data.shape[0]):
        training_data.loc[y, i] = (training_data.loc[y, i] - neuron_mean) / neuron_sd

# Output Data Normalization
for x in range(input_neurons, input_neurons + out_neurons):
    max_value = np.max(training_data[x])
    min_value = np.min(training_data[x])
    f_max.write('%7.9f' % max_value)
    f_max.write('\n')
    f_min.write('%7.9f' % min_value)
    f_min.write('\n')
    for cell in range(training_data.shape[0]):
        training_data.loc[cell, x] = (training_data.loc[cell, x] - min_value) / (max_value - min_value)

# create 2D arrays for weights randomly
input_hidden_weights = np.random.rand(input_neurons, hidden_neurons)
hidden_out_weights = np.random.rand(hidden_neurons, out_neurons)
learning_rate = 0.1

# Part1-  Training

AK_Predected_values = []
MSE_iterations = []
MSE_value = 0.0
itr = 0
while MSE_value > 0.0005 or itr < 500:
    for example in range(n_examples):
        # calculate forward propagation
        Aj = to_hidden(hidden_neurons, training_data.loc[[example]], example, input_hidden_weights)
        Ak = to_output(out_neurons, Aj, hidden_out_weights)
        AK_Predected_values.extend(Ak)

        # calculate backward propagation
        deltak = delta_output(Ak, training_data, example, input_neurons)
        deltaj = delta_hidden(Aj, hidden_out_weights, deltak)
        new_out_weights = out_update(hidden_out_weights, learning_rate, deltak, Aj)
        new_hidden_weights = hidden_update(input_hidden_weights, learning_rate, example, deltaj,
                                           training_data.loc[[example]])
        # updating weights
        input_hidden_weights = new_hidden_weights
        hidden_out_weights = new_out_weights
    # calculating MSE
    MSE_value = MSE(training_data, AK_Predected_values)
    MSE_iterations.append(MSE_value)
    print(MSE_value)
    AK_Predected_values = []

print("training MSE value : ")
print(MSE_value)


# save weights to output file
np.savetxt(f_weights, input_hidden_weights, fmt='%-7.9f')
np.savetxt(f_weights, hidden_out_weights, fmt='%-7.9f')

f_means.close()
f_standards.close()
f_max.close()
f_min.close()
f_weights.close()
testing()
