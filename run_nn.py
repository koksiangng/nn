import nn

def run_nn():
    inp, x, y, out = map(int, input())
    nn = nn(inp, x, y, out)
    

if __name__ == "__main__":
    run_nn()