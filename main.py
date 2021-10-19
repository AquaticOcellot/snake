import arcade
import numpy as np
import matplotlib.pyplot as plt

ARCHITECTURE = [
    {'neuron_number': 4, 'activation': None},
    {'neuron_number': 5, 'activation': 'relu'},
    {'neuron_number': 4, 'activation': 'sigmoid'},
]
DATA = np.array((
    [1, 3, 4, 0],
    [4, 3, 2, 0],
    [2, 8, 4, 0],
    [2, 3, 4, 4.5],
    [2, 3, 4, 3.5],
))
TARGETS = np.array((
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
))


class Net:
    def __init__(self, architecture, targets, first_layers=None, weights=None, biases=None, batch_size=2):
        self.architecture = architecture
        self.layers = []
        self.weights = []
        self.biases = []
        self.targets = targets
        self.batch_size = batch_size

        self.create_layers()
        self.create_weights()
        self.create_biases()
        if first_layers is not None:
            i = 0
            while i < min(len(first_layers), batch_size):
                self.layers[0][i] = first_layers[i]
                i += 1
        if weights is not None:
            self.weights = weights
        if biases is not None:
            self.biases = biases

    def create_layers(self):
        for single_layer_info in self.architecture:
            self.layers.append(rng.uniform(low=-0.5, high=0.5,
                                           size=(self.batch_size,
                                                 single_layer_info['neuron_number'])))

    def create_weights(self):
        for i in range(len(self.architecture) - 1):
            self.weights.append(rng.uniform(low=-0.5, high=0.5,
                                            size=(self.batch_size,
                                                  self.architecture[i]['neuron_number'],
                                                  self.architecture[i + 1]['neuron_number'])))

    def create_biases(self):
        for i in range(len(self.architecture) - 1):
            self.biases.append(rng.uniform(low=-0.5, high=0.5,
                                           size=(self.batch_size,
                                                 self.architecture[i + 1]['neuron_number'])))

    def propagate(self):
        # TODO change to using functions?
        for layer_pair_number in range(len(self.architecture) - 1):
            if self.architecture[layer_pair_number + 1]['activation'] == 'relu':
                self.layers[layer_pair_number + 1] = np.maximum(np.einsum('...i,...ij->...j',
                                                                          self.layers[layer_pair_number],
                                                                          self.weights[layer_pair_number])
                                                                + self.biases[layer_pair_number],
                                                                0)
            if self.architecture[layer_pair_number + 1]['activation'] == 'sigmoid':
                self.layers[layer_pair_number + 1] = 1 / (1 + np.exp(np.einsum('...i,...ij->...j',
                                                                               self.layers[layer_pair_number],
                                                                               self.weights[layer_pair_number])
                                                                     + self.biases[layer_pair_number]))

    """
    def backpropagate(self):
        for layer_pair_number in range(0, len(self.architecture) - 1, -1):
            if self.architecture[layer_pair_number + 1]['activation'] == 'relu':
                self.layers[layer_pair_number + 1] = 
    """

    def calculate_loss(self):
        loss = -np.sum(self.targets[:self.batch_size] * np.log2(self.layers[-1]))
        return loss

    def output_layers(self):
        for layer in self.layers:
            print(layer[0])
            plt.plot(layer[0])

    def output_weights(self):
        for weight_layer in self.weights:
            print(weight_layer)


def simple_example():
    net_1 = Net(architecture=ARCHITECTURE, first_layers=DATA, targets=TARGETS)
    net_1.propagate()
    net_1.calculate_loss()
    net_1.output_layers()
    plt.show()


rng = np.random.default_rng()

simple_example()
