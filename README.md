# Micrograd in PiLang

A small, educational implementation of **Andrej Karpathy's micrograd** reimplemented in [**PiLang**](https://github.com/rolandbrake/pilang).

This project builds a scalar-valued computational graph and performs reverse-mode automatic differentiation. It also includes the core building blocks of a small neural network: neurons, layers, a multi-layer perceptron, gradient descent, and a training example.

The original project is inspired by [micrograd](https://github.com/karpathy/micrograd), while this repository adapts its design and examples to PiLang syntax and modules.

## Features

- Scalar `Value` objects with `data` and `grad`
- Automatic construction of a computational graph
- Reverse-mode backpropagation with topological graph traversal
- Addition, subtraction, multiplication, negation, and scalar powers
- ReLU activation
- Automatic parameter gradient reset
- `Neuron`, `Layer`, and `MLP` modules
- A complete XOR-style classification example with a loss chart

## Project Structure

| File | Description |
| --- | --- |
| `engine.pi` | Defines the scalar `Value` autograd engine |
| `nn.pi` | Defines `Module`, `Neuron`, `Layer`, and `MLP` |
| `main.pi` | Trains an MLP and plots the training loss |
| `main.px` | Compiled PiLang bytecode for the example |

## Requirements

- PiLang installed and available from your terminal
- PiLang modules used by the example:
  - `engine`
  - `nn`
  - `draw`
  - `plot`
  - `random`
  - `col`

## Running the Example

From the project directory, run:

```bash
pilang main.pi
```

The example trains a `3 -> 4 -> 4 -> 1` multilayer perceptron for 150 steps, prints the loss and final predictions, and opens a loss chart.

## A Simple Autograd Example

```pi
import engine.{Value}

a = Value(-4.0)
b = Value(2.0)

c = a + b
d = a * b + b**3
e = c.relu()
f = d + e

f.backward()

println(f.data)
println(a.grad)
println(b.grad)
```

The `backward()` method seeds the output gradient with `1` and propagates local gradients through the computational graph in reverse topological order.

## Training a Neural Network

```pi
import nn.{MLP}
import engine.{Value}

const xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

const ys = [1.0, -1.0, -1.0, 1.0]
const model = MLP(3, [4, 4, 1])

for step in range(150) {
    let ypred = [model(x) : x in xs],
        loss = Value(0)

    for i in range(#ypred) {
        let diff = ypred[i] - ys[i]
        loss += diff * diff
    }

    model.zero_grad()
    loss.backward()

    for p in model.parameters()
        p.data += -0.01 * p.grad

    println("step " + step + " loss " + loss.data)
}
```

This example uses:

- Mean-squared-error loss
- Full-batch training
- ReLU hidden layers
- A linear output neuron
- Gradient descent with a learning rate of `0.01`

## API Overview

### `Value`

`Value(data, prev=(), op='')` represents one scalar node in the computation graph.

```pi
x = Value(2.0)
y = x * x + Value(1.0)

y.backward()
```

Available operations include:

```pi
a + b
a - b
a * b
-a
x**2
a.relu()
```

Power operations currently support numeric scalar exponents.

### `Neuron`

`Neuron(nin, nonlin=true)` creates a neuron with randomly initialized weights and a zero bias.

```pi
neuron = Neuron(3)
output = neuron([1.0, 2.0, 3.0])
```

### `Layer`

`Layer(nin, nout, nonlin=true)` creates a collection of neurons.

```pi
layer = Layer(3, 4)
```

### `MLP`

`MLP(nin, nouts)` creates a feed-forward network. The final layer is linear; earlier layers use ReLU.

```pi
model = MLP(3, [4, 40, 1])
```

Use the following methods with any module in the network:

- `model(x)` performs a forward pass
- `model.parameters()` returns all trainable `Value` objects
- `model.zero_grad()` resets every parameter gradient to zero
- `loss.backward()` backpropagates gradients through the graph

## How Backpropagation Works

1. Operations create new `Value` nodes.
2. Each node records its input nodes and operation.
3. `_backward` functions calculate local derivatives.
4. `backward()` builds a topological ordering of the graph.
5. The output gradient is set to `1`.
6. Nodes are visited in reverse order and gradients are accumulated.
7. Graph references are cleared to release closures and reduce memory retention.

This is an educational implementation and is intentionally small. It is not designed to replace optimized tensor libraries for large-scale training.

## Limitations

- Uses scalars rather than tensors
- Supports CPU-only scalar operations
- No mini-batch abstraction or optimizer classes
- No built-in model serialization
- No softmax, sigmoid, cross-entropy, or general loss module yet
- `Value` hashing and graph identity depend on PiLang object behavior

## Inspiration and Acknowledgements

This project follows the ideas and structure of the original **micrograd** by Andrej Karpathy:

- [Micrograd on GitHub](https://github.com/karpathy/micrograd)
- [Micrograd explanation video](https://www.youtube.com/watch?v=VMj-3S1tku0)

This implementation is not affiliated with or endorsed by Andrej Karpathy.

