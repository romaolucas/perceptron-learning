import numpy as np

def calculate_activation_func(y):
    return 1 if y > 0.0 else 0

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
    misclassified_inputs = [[X[:, i], i] for i in range(len(X.T)) if calculate_output(w, X[:, i]) != t[i]]
    import random
    random_index = random.randint(0, len(misclassified_inputs) - 1)
    return misclassified_inputs[random_index], calculate_output(w, misclassified_inputs[random_index][0]) 


'''
updates the weight vector given expected output t
obtained output y, input x and learning rate eta
'''
def update_weights(w, t, y, x, eta):
    w = w + eta*(t - y)*x
    return w


def has_converged(w, X, t):
    y = [ calculate_output(w, x) for x in X.T]
    return (y == t).all()

w = np.array([0, 0, 0])
t = np.array([0, 0, 0, 1])
X = np.array([[1, 0, 0], [1, 0, 1], \
        [1, 1, 0], [1, 1, 1]]).T
eta = 0.5

while 1: 
    if has_converged(w, X, t):
        break
    misclassified_input, y = get_random_misclassfied_input(w, t, X)
    index = misclassified_input[-1]
    w = update_weights(w, t[index], y, misclassified_input[0], eta)
    print("new weight vector", w)
    
print("converged with weight vector ", w)
