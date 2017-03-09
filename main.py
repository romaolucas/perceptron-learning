import numpy as np

def calculate_activation_func(y):
    return 1.0 if y > 0.0 else 0.0

#calculate the output given weight vector w and input vector x
def calculate_output(w, x):
    return calculate_activation_func(np.dot(w, x))

'''
returns a vector x_i in X such that y(x_i) != t_i
and the obtained output y(x_i)
t is the vector with the expected outputs
X is the matrix with all the inputs
w is the weight vector
'''
def get_random_misclassfied_input(w, t, X):
    misclassified_inputs = [X[:, i] for i in range(X.T) if calculate_output(w, X[:, i]) != t[i]]
    import random
    random_index = random.uniform(0, len(misclassified_inputs) - 1)
    return misclassified_inputs[random_index], calculate_output(w, misclassified_inputs[random_index]) 


'''
updates the weight vector given expected output t
obtained output y, input x and learning rate eta
'''
def update_weights(w, t, y, x, eta):
    w += eta*(t - y)*x
    return w


