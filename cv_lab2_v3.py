import numpy as np

def activation(out):
    out = max(0, out)
    return out

def convolution(inp, weight):
    out_shape = [weight.shape[0], inp.shape[1], inp.shape[2]]
    size = 1
    inp = np.pad(inp, ((0, 0), (size, size), (size, size)))
    inp = np.pad(inp, ((0, 0), (0, 0), (0, 0)))
    out = np.zeros(tuple(out_shape))
    r = int(weight.shape[1] / 2)
    for m in range(out.shape[0]):
        for x in range(out.shape[1]):
            for y in range(out.shape[2]):
                for i in range(weight.shape[1]):
                    for j in range(-r, r + 1):
                        for k in range(-r, r + 1):
                            out[m, x, y] += inp[i, x + j +1 , y + k + 1] * weight[m, i, j, k]
                out[m,x,y] = activation(out[m,x,y])
    return out

def max_pooling(inp):
    out_shape = [inp.shape[0], int(inp.shape[1] /2), int(inp.shape[2] / 2)]
    out = np.zeros(tuple(out_shape))
    for m in range(out.shape[0]):
        for x in range(out.shape[1]):
            for y in range(out.shape[2]):
                out[m, x, y] = max(inp[m, x * 2, y * 2], inp[m, x * 2, y * 2 + 1], inp[m, x * 2 + 1, y * 2], inp[m, x * 2 + 1, y * 2 + 1])
    return out

def main():
    inp = np.random.rand(3, 100, 100)
    weight = np.random.rand(5, 3, 3, 3)
    inp = convolution(inp, weight)
    inp = max_pooling(inp)

if __name__ == "__main__":
    main()