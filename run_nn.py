from nn import NN
import sys
import numpy as np

#inspired from:
#https://sirupsen.com/napkin/neural-net
#https://peterroelants.github.io/posts/neural-network-implementation-part01/

#MSE: SUM(obs(i) - pred(i))^2 / obs

def run_nn():
    try:
        inp, x, y, out = map(int, input().split(" "))
        nn = NN(inp, x, y, out)
    except:
        if inp is None:
            print("inp missing")
        if x is None:
            print("x missing")
        if y is None:
            print("y missing")
        if out is None:
            print("out missing")
        print("Unknown exception")
    
    data, exp = create_data()
    #print(data[0:10])
    #print(exp[0:10])
    trained = nn.train(data)
    #print(trained[0:10])
    print(nn.MSE(trained, exp))


#Creates data, 1000 rectangles (white - black)
#Also gets average of rectangles
def create_data():
    rectangles = []
    rectangles_average = []
    for _ in range(0, 1000):
        rectangle = [round(np.random.random(), 1), round(np.random.random(), 1), round(np.random.random(), 1), round(np.random.random(), 1)]

        rectangles.append(rectangle)
        rectangles_average.append(sum(rectangle) / 4)
    
    return rectangles, rectangles_average
        
    

if __name__ == "__main__":
    run_nn()