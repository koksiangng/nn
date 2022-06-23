import numpy as np

#input layer, hidden layer, output layer
#universal approximation theorem

class NN:
    def __init__(self, input_size, hidden_x, hidden_y, output_size):
        self.input_size = input_size
        self.hidden_x = hidden_x
        self.hidden_y = hidden_y
        self.output_size = output_size

        np.random.seed(0)

        self.input_layer = [0] * self.input_size
        self.hidden_layer = [[round(np.random.uniform(0, 1.0), 1) for _ in range(self.hidden_y)] for _ in range(self.hidden_x)]
        self.output_layer = [0] * self.output_size

    def print_layers(self):
        for i in self.hidden_layer:
            print(i)
    
    def MSE(self, actual, expected):
        error = 0
        for a, e in zip(actual, expected):
            error += (a - e) ** 2
        return error / len(actual)
        
    #Matrix mult between input and hidden layer = output
    def calculate(self, inp_values):
        output = 0
        for index, inp_neuron in enumerate(inp_values):
            output += inp_neuron * self.hidden_layer[0][index]
        return output
    
    #Currently, values = 4 square values
    #Gets result from multiple "calculate"s.
    def train(self, all_values):
        outputs = []
        for values in all_values:
            outputs.append(self.calculate(values))
        return outputs