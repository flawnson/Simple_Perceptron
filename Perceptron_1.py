from random import choice
from numpy import array, dot, random

#Normalizer
def normalize(x):
    if x < 0:
        x = 0
    else:
        x = 1
    return x

#Training dataset
dataset = [
    (array([1,0,1]), 1),
    (array([0,1,1]), 1),
    (array([1,1,1]), 1),
    (array([0,0,1]), 0),
]

#Random integer generator
w = random.rand(3)

#Initializers (Error list, Learning rate, and Iterations)
errors = []
learning_rate = .2
itr = 10

#Training function
for y in range(itr):
    x, expected = choice(dataset)
    result = dot(w, x)
    error = expected - normalize(result)
    errors.append(error)
    w += learning_rate * itr * x

#Output function
for x, _ in dataset:
    result = dot(x, w)
    print ("{}: {} ---> {}".format(x[:2], result, normalize(result)))