import matplotlib.pyplot as plt
import numpy as np
import random


# Gets a random number in a range.
def uniform_random(low, high):
    return low+(high-low)*random.random()


# Makes an exponential function.
def exp_f(a, b):
    return lambda x: a*(b**x)


# Generates data from a provided function with some perturbation
def generate_data(f, n=10, perturbation=1.0):
    return np.array([[i, f(i+uniform_random(-perturbation, perturbation))] for i in range(n)]).T


# Shows data from a reference function against provided data.
def graph_data(reference_f, guess_f, data):
    # Show the data.
    plt.scatter(data[0, :], data[1, :], label="data", color="y")

    # Show the guess function.
    x = np.linspace(data[0][0], data[0][-1])
    plt.plot(x, guess_f(x), label="guess", color="r")

    # Show the reference function.
    plt.plot(x, reference_f(x), label="reference")

    # Set the axes information and show.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


# Finds the MSE of a function given data.
def mse_exp(func, data):
    error = 0
    a, b = func[0]
    exp_func = exp_f(a, b)
    for point in data.T:
        x, y = point
        error += (y-exp_func(x))**2
    return error


# Finds the gradient of an exponential function's error given some input data.
def mse_exp_grad(func, data):
    a, b = func[0]
    exp_func = exp_f(a, b)
    grad = [0, 0]
    for point in data.T:
        x, y = point
        grad[0] += -2*(b**x)*(y-exp_func(x))
        grad[1] += -2*a*np.log(b)*(b**x)*(y-exp_func(x))
    return np.array([grad])


# Performs gradient descent on the exponential function.
def gradient_descent(data, reference_f, learning_rate=0.1, epsilon=1e-2, max_iterations=1000):
    # Initialize the guessed function randomly and show it.
    func = np.array([[random.random(), random.random()]])
    graph_data(reference_f, exp_f(func[0][0], func[0][1]), data)
    plt.draw()
    plt.title(f"Exponential Regression: Step 0")
    plt.waitforbuttonpress(timeout=60)

    # Perform gradient descent.
    for i in range(max_iterations):
        a, b = func[0]
        plt.clf()
        graph_data(reference_f, exp_f(a, b), data)
        plt.draw()
        plt.title(f"Exponential Regression: Step {i}")
        plt.waitforbuttonpress(timeout=60)
        grad = mse_exp_grad(func, data)
        grad *= (np.linalg.norm(func)/np.linalg.norm(grad))
        func -= learning_rate*grad
        if func[0][1] <= 0:
            func[0][0] = 1 + uniform_random(-0.5, 0.5)
            func[0][1] = 1+uniform_random(-0.5, 0.5)
        if mse_exp(func, data) <= epsilon:
            break
        print(f"Current Guess: y={func[0][0]} * {func[0][1]}^x")
    return func


if __name__ == "__main__":
    # Generate some data for our reference function, which will be roughly y=2^x.
    reference = exp_f(1, 2)
    perturbed_data = generate_data(reference, perturbation=5e-1)
    gradient_descent(perturbed_data, reference, learning_rate=0.1)
