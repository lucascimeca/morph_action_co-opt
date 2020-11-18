# from palpation import Brain
from main_bayesian_exploration import BayesianBrain
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Body:
    skin_shape = (1, 2)
    data_folder = ""

    def __init__(self):
        pass


## FIRST TEST CASE 1D

objects = list(range(10))
actions = list(range(100))

body = Body()

brain = BayesianBrain(actions=actions)

brain.task_objects = objects
brain.actions = actions
brain.number_of_objects = len(objects)
brain.number_of_actions = len(actions)
brain.prior = 1./brain.number_of_objects

pdfs = {}
dist = 0.
mu = 5.
for act in actions:
    if act not in pdfs.keys():
        pdfs[act] = {}
    for obj in objects:
        pdfs[act][obj] = (np.array([mu]), np.array([1.]))
        mu += dist
    dist += .1

brain.pdfs = pdfs
brain._compute_cpms()

benefit = brain.get_perceived_benefit(non_biased=True)

plt.plot(sorted(brain.pdfs.keys()), benefit)
plt.show()

print("The best motion is {}. \nGraph should be a monotonically increasing function.".format(brain.get_best_actions(non_biased=True)[0]))


## SECOND TEST CASE 2D

objects = list(range(10))
actions = list(range(100))

body = Body()

brain = BayesianBrain(actions=actions)

brain.task_objects = objects
brain.actions = actions
brain.number_of_objects = len(objects)
brain.number_of_actions = len(actions)
brain.prior = 1./brain.number_of_objects

pdfs = {}
dist = np.array([0., 0.])
mu = np.array([5., 5.])
std = 1.
for act in actions:
    if act not in pdfs.keys():
        pdfs[act] = {}
    for obj in objects:
        pdfs[act][obj] = (np.array([mu]), np.array([[std, 0.], [0., std]]))
        mu += dist
    dist += np.array([.1, .1])

brain.pdfs = pdfs
brain._compute_cpms()

benefit = brain.get_perceived_benefit(non_biased=True)

plt.plot(sorted(brain.pdfs.keys()), benefit)
plt.show()

print("The best motion is {}. \nGraph should be a monotonically increasing function.".format(brain.get_best_actions(non_biased=True)[0]))


## SECOND TEST CASE 1D

objects = list(range(10))
actions = list(range(100))

body = Body()

brain = BayesianBrain(actions=actions)

brain.task_objects = objects
brain.actions = actions
brain.number_of_objects = len(objects)
brain.number_of_actions = len(actions)
brain.prior = 1./brain.number_of_objects

pdfs = {}
mu_dist = .5
std_incr = 0.
mu = 5.
for act in actions:
    if act not in pdfs.keys():
        pdfs[act] = {}
    for obj in objects:
        pdfs[act][obj] = (np.array([mu]), np.array([1.])+np.array(std_incr))
        mu += mu_dist
    std_incr += .1

brain.pdfs = pdfs
brain._compute_cpms()

benefit = brain.get_perceived_benefit(non_biased=True)

plt.plot(sorted(brain.pdfs.keys()), benefit)
plt.show()

print("The best motion is {}. \nGraph should be a monotonically decreasing function.".format(brain.get_best_actions(non_biased=True)[0]))


## SECOND TEST CASE 2D

objects = list(range(10))
actions = list(range(100))

body = Body()

brain = BayesianBrain(actions=actions)

brain.task_objects = objects
brain.actions = actions
brain.number_of_objects = len(objects)
brain.number_of_actions = len(actions)
brain.prior = 1./brain.number_of_objects

pdfs = {}
mu = np.array([5., 5.])
mu_dist = np.array([.5, .5])
std = 1.
std_incr = 0.
for act in actions:
    if act not in pdfs.keys():
        pdfs[act] = {}
    for obj in objects:
        pdfs[act][obj] = (np.array([mu]), np.array([[std, 0.], [0., std]]) + np.array([[std_incr, 0.], [0., std_incr]]))
        mu += mu_dist
    std_incr += .1

brain.pdfs = pdfs
brain._compute_cpms()

benefit = brain.get_perceived_benefit(non_biased=True)

plt.plot(sorted(brain.pdfs.keys()), benefit)
plt.show()

print("The best motion is {}. \nGraph should be a monotonically increasing function.".format(brain.get_best_actions(non_biased=True)[0]))