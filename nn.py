import enum
import numpy as np

#input layer, hidden layer, output layer

class nn:
    def __init__(self, input_size, hidden_x, hidden_y, output_size):
        self.input_size = input_size
        self.hidden_x = hidden_x
        self.hidden_y = hidden_y
        self.output_size = output_size

        self.input_layer = [0] * self.input
        self.hidden_layer = [[np.random.seed(0) for _ in range(self.hidden_y)] for _ in range(self.hidden_x)]
        self.output_layer = [0] * self.output

    def print_layers(self):
        for i in self.hidden_layer:
            print(i)
    
    def calculate(self, values):
        for index, input_neuron in enumerate(self.input_layer):
