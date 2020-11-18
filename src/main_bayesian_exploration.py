""" this is used mainly to re-run experiments off-line from logged data, else see -> main"""

import time
import json
import sys
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import svm
import copy
from scipy.stats import multivariate_normal
from pprint import pprint
from data_operators import *
from visualization import *

# print ("Random number with seed")
np.random.seed(9905)

acc_test_dict = [
    ["15mil", "na"],
    ["8mil", "na"],
    ["15mil", "8mil", "na"],
    ["10mil", "na"],
    ["5mil", "na"],
    ["10mil", "5mil", "na"],
]


def generate_action(current_state, idx, min_vec, max_vec, number_of_actions):
    if min_vec.shape[0] <= idx:
        return current_state
    else:
        if min_vec[idx] != 0 or max_vec[idx] != 0:
            if min_vec[idx] == max_vec[idx]:
                actions = [max_vec[idx]]
            else:
                actions = np.arange(min_vec[idx], max_vec[idx], (max_vec[idx]-min_vec[idx])/(number_of_actions+1))[1:]
            if current_state is None:
                current_state = np.array([[0.0] * min_vec.shape[0]])

            state = current_state.copy()
            for i, act in enumerate(actions):
                if i == 0:
                    current_state[:, idx] = act
                else:
                    state_copy = state.copy()
                    state_copy[:, idx] = act
                    current_state = np.concatenate((current_state, state_copy), axis=0)
        return generate_action(current_state, idx+1, min_vec, max_vec, number_of_actions)


# Generate action profile, by using recursive function
def get_action_profile(eta_min, eta_max, A_min, A_max, morphs=None, number_of_actions=3):
    actions = generate_action(
        None,
        0,
        np.concatenate((eta_min, A_min), axis=0),
        np.concatenate((eta_max, A_max), axis=0),
        number_of_actions)
    action_profile = []
    if morphs is not None:
        idx = 0
        for i in range(len(morphs)):
            for j in range(actions.shape[0]):
                action_profile += [
                    (idx, list(actions[j, :eta_min.shape[0]]), list(actions[j, eta_min.shape[0]:]), morphs[i])
                ]
                idx += 1
    else:
        for i in range(actions.shape[0]):
            action_profile += [(i, list(actions[i, :eta_min.shape[0]]), list(actions[i, eta_min.shape[0]:]))]
    return action_profile


class BayesianBrain(object):

    priors = None
    pdfs = None   # expected pdfs[action][object] = (mu, sigma)
    objects = None
    actions = None
    pca = None

    alpha_discount = 10

    skin_buffer_idx = 0
    location_stack = None       # array holding the locations in the phantoms (each an x-y tuple)
    location_labels = None      # array holding the true labels for each known location
    location_inference = None   # array that will hold the inference made for an object at a location
    test_data = None            # np array holding the data gathered during testing
                                # rodo:(right now hold in RAM but eventually dump into memory!)
    data_objects = None         # labels of the training data for the best motion
    training_data = None        # PCA reduced training data retrieved from best motion parameter
    training_labels = None      # array of labels for training data
    normalize_min = None        # parameters to normalize new data based on training data
    normalize_max = None        # parameters to normalize new data based on training data
    clf = None

    data_folder = ""
    results_folder = ""

    def __init__(self, environment=None, actions=None, priors=None, sensing_resolution=10,
                 dimensionality_reduction=2, samples_per_pdf=3, training_ratio=.8, experiment_number=False, task=0,
                 data_folder='./../data/', verbose=False):

        # REMOVE THE -1 IF YOU'RE NOT SUBTRACTING THE STARTING CONDITION
        self.sensing_resolution = sensing_resolution-1
        self.dimensionality_reduction = dimensionality_reduction
        self.samples_per_pdf = samples_per_pdf
        self.verbose = verbose

        if priors is not None:
            self.prior = priors

        self.idx_count = 0
        self.current_motion = None
        self.sense_time_step = None
        self.previous_pos = None
        self.depths_levels = None
        self.current_time = None
        self.current_time_level = None
        self.time_levels = None
        self.total_number_of_experiments = 0
        self.idx_dict = {}    # dictionary of form idx_dict[action][object] = [[idx1, idx2, ...], [idx55, ...], ...]
        self.clock = time.clock()
        self.task = task

        # retrieve latest experiment
        base_name = "experiment"
        experiment_folders = list(sorted([file for file in os.listdir(data_folder) if file.startswith(base_name)],
                                         key=lambda x:int(x.split('_')[1])))
        if len(experiment_folders) == 0:
            print("Please gather some data before doing Bayesian Inference.")
            raise (FileNotFoundError("Experiment folder in {}".format(data_folder)))

        # TO REMOVE
        if experiment_number is not None:
            self.data_folder = data_folder + experiment_folders[experiment_number] + "/"
        else:
            self.data_folder = data_folder + experiment_folders.pop() + "/"
        self.results_folder = self.data_folder + "results/"


        if environment is not None:
            if environment is None or actions is None:
                raise ValueError("To learn, the doctor must be supplied a known environment and actions!")
            self.objects = sorted(environment.keys())
            self.task_objects = np.unique([obj[task] for obj in self.objects])
            self.task_object_map = {}

            for obj in self.objects:
                if obj[task] not in self.task_object_map.keys():
                    self.task_object_map[obj[task]] = [obj]
                else:
                    self.task_object_map[obj[task]] += [obj]

            self.number_of_objects = len(self.task_objects)  # D
            self.actions = actions
            self.number_of_actions = len(actions)  # N
            self.environment = environment

        self._load_experiment_details()
        self.skin_buffer = np.zeros((self.sensing_resolution,) + self.loaded_skin_shape)

        self.training_ratio = training_ratio

    def run_experiment(self, bayesian=True, initial_sample_number=2, test_object_number=1, test_idxs=None, delay=0,
                       task=0, caps=None, best_actions=None, previous_accuracies=None, maximum_iterations=False, show=True, save=False):

        if save:
            folder_create(self.results_folder, exist_ok=True)

        # load initial data from Hard Disk
        belief_data, test_data = self._load_initial_data(
            test_object_number=test_object_number,
            initial_sample_number=initial_sample_number,
            test_idxs=test_idxs
        )
        # make data low dimensional and fit pdfs on test data
        reduced_belief_data, reduced_test_data, original_belief_data, original_test_data = self._fit_pdfs(
            belief_data,
            test_data,
            task=task
        )

        # create test/train low-dim datasets (and labels)
        self._compute_cpms()
        action_benefits, ranked_actions = self.get_best_actions()
        unbiased_action_benefits, unbiased_ranked_actions = self.get_best_actions(non_biased=True)

        accuracy = self.test_bayesian(
            act=ranked_actions[0],
            reduced_belief_data=reduced_belief_data.copy(),
            reduced_test_data=reduced_test_data.copy()
        )

        unbiased_accuracy = self.test_bayesian(
            act=unbiased_ranked_actions[0],
            reduced_belief_data=reduced_belief_data.copy(),
            reduced_test_data=reduced_test_data.copy()
        )

        actions_to_explore = [ranked_actions[0]]
        best_action_benefits = [action_benefits[0]]
        unbiased_best_action_benefits = [unbiased_action_benefits[0]]
        explored_accuracies = [accuracy]
        unbiased_best_actions = [unbiased_ranked_actions[0]]
        best_accuracies = [accuracy]
        running_unbiased_accuracy = [unbiased_accuracy]
        best_reduced_belief_data = belief_data
        best_reduced_test_data = test_data
        best_running_action = ranked_actions[0]
        iteration_rankings = []
        iterations_b_coeffs = []

        robot_learning_steps = 0

        new_belief_data = belief_data.copy()
        if bayesian:
            while new_belief_data is not None and test_data is not None:
                print("iteration:{}, action to explore: {}, best action: {}, counts: {}".format(
                    robot_learning_steps, actions_to_explore[-1], best_running_action, self.action_counts[actions_to_explore[-1]])
                )
                # print(ranked_actions)

                reduced_belief_data, reduced_test_data, original_belief_data, original_test_data = self._fit_pdfs(
                    new_belief_data, test_data,
                    act=actions_to_explore[-1],
                    reduced_belief_data=reduced_belief_data,
                    reduced_test_data=reduced_test_data,
                    original_belief_data=original_belief_data,
                    original_test_data=original_test_data
                )
                self._compute_cpms()
                action_benefits, ranked_actions = self.get_best_actions()
                unbiased_action_benefits, unbiased_ranked_actions = self.get_best_actions(non_biased=True)

                accuracy = self.test_bayesian(
                    act=actions_to_explore[-1],  # best_actions[task],
                    reduced_belief_data=reduced_belief_data.copy(),
                    reduced_test_data=reduced_test_data.copy(),
                    task=task
                )

                unbiased_accuracy = self.test_bayesian(
                    act=unbiased_best_actions[-1],
                    reduced_belief_data=reduced_belief_data.copy(),
                    reduced_test_data=reduced_test_data.copy(),
                    task=task
                )

                running_unbiased_accuracy += [unbiased_accuracy]
                if accuracy >= best_accuracies[-1]:
                    best_accuracies += [accuracy]
                    best_reduced_belief_data = best_reduced_belief_data
                    best_reduced_test_data = reduced_test_data
                    best_running_action = actions_to_explore[-1]
                else:
                    best_accuracies += [best_accuracies[-1]]

                actions_to_explore += [ranked_actions[0]]
                best_action_benefits += [action_benefits[0]]
                explored_accuracies += [accuracy]
                iterations_b_coeffs += [action_benefits]
                iteration_rankings += [ranked_actions]
                unbiased_best_actions += [unbiased_ranked_actions[0]]
                unbiased_best_action_benefits += [unbiased_action_benefits[0]]

                if show:
                    show_progress(actions=actions_to_explore,
                                  benefits=best_action_benefits,
                                  previous_accuracies=previous_accuracies,
                                  windows_name="bayesian_accuracy",
                                  bayesian=bayesian,
                                  accuracies=best_accuracies,
                                  delay=delay)
                    if self.dimensionality_reduction == 2:
                        show_bayesian2d(pdfs=self.pdfs,
                                        act=actions_to_explore[-1],
                                        windows_name="progress_bayesian",
                                        reduced_train_data=reduced_belief_data.copy(),
                                        reduced_test_data=reduced_test_data.copy(),
                                        task=task)
                    else:
                        show_bayesian(pdfs=self.pdfs,
                                      act=actions_to_explore[-1],
                                      reduced_train_data=reduced_belief_data.copy(),
                                      reduced_test_data=reduced_test_data.copy(),
                                      folder="",
                                      show=show,
                                      delay=delay,
                                      windows_name="progress_bayesian")

                robot_learning_steps += 1
                if (previous_accuracies is not None and len(previous_accuracies) <= robot_learning_steps)\
                        or (maximum_iterations and maximum_iterations <= robot_learning_steps):
                    break
                belief_data = new_belief_data.copy()
                new_belief_data = self._load_next_experiment(act=actions_to_explore[-1])
                # print("robot_learning_step: {}".format(robot_learning_steps))
        else:
            act_idx = 0
            act = self.actions[act_idx][0]
            while new_belief_data is not None and test_data is not None:
                print("iteration: {}, action to explore: {}, best action: {}, counts: {}".format(
                    robot_learning_steps, act, best_running_action, self.action_counts[act]))

                reduced_belief_data, reduced_test_data, original_belief_data, original_test_data = self._fit_pdfs(
                    new_belief_data, test_data,
                    act=act,
                    reduced_belief_data=reduced_belief_data,
                    reduced_test_data=reduced_test_data,
                    original_belief_data=original_belief_data,
                    original_test_data=original_test_data
                )
                self._compute_cpms()
                action_benefits, ranked_actions = self.get_best_actions()

                accuracy = self.test_bayesian(
                    act=act,
                    reduced_belief_data=reduced_belief_data.copy(),
                    reduced_test_data=reduced_test_data.copy()
                )

                running_unbiased_accuracy += [unbiased_accuracy]
                actions_to_explore += [act]
                best_action_benefits += [0]
                explored_accuracies += [accuracy]
                iterations_b_coeffs += [action_benefits]
                # if accuracy > best_accuracies[-1]:
                if accuracy >= best_accuracies[-1]:
                    best_accuracies += [accuracy]
                    best_reduced_belief_data = best_reduced_belief_data
                    best_reduced_test_data = reduced_test_data
                    best_running_action = act
                else:
                    best_accuracies += [best_accuracies[-1]]

                if show:
                    show_progress(actions=actions_to_explore,
                                  benefits=best_action_benefits,
                                  previous_accuracies=previous_accuracies,
                                  bayesian=bayesian,
                                  delay=delay,
                                  accuracies=best_accuracies)
                    if self.dimensionality_reduction == 2:
                        show_bayesian2d(pdfs=self.pdfs,
                                  act=actions_to_explore[-1],
                                  windows_name="bayesian_svm",
                                  task=task,
                                  reduced_train_data=reduced_belief_data.copy(),
                                  reduced_test_data=reduced_test_data.copy())
                    else:
                        show_bayesian(pdfs=self.pdfs,
                                      act=actions_to_explore[-1],
                                      reduced_train_data=reduced_belief_data.copy(),
                                      reduced_test_data=reduced_test_data.copy(),
                                      folder="",
                                      show=show,
                                      delay=delay,
                                      windows_name="progress_bayesian")

                # time.sleep(.3)
                robot_learning_steps += 1
                if previous_accuracies is not None and len(previous_accuracies) <= robot_learning_steps\
                        or (maximum_iterations and maximum_iterations <= robot_learning_steps):
                    break

                act_idx = np.mod(act_idx+1, self.number_of_actions)
                act = self.actions[act_idx][0]
                new_belief_data = self._load_next_experiment(act=act)

        unbiased_action_benefits, unbiased_ranked_actions = self.get_best_actions(non_biased=True)

        if show or save:
            show_progress(actions=actions_to_explore,
                          benefits=best_action_benefits,
                          previous_accuracies=previous_accuracies,
                          bayesian=bayesian,
                          accuracies=best_accuracies,
                          folder=self.results_folder,
                          show=show,
                          save=save)
            if self.dimensionality_reduction == 2:
                show_bayesian2d(pdfs=self.pdfs,
                          act=actions_to_explore[-1],
                          windows_name="progress_bayesian",
                          task=task,
                          reduced_train_data=reduced_belief_data.copy(),
                          reduced_test_data=reduced_test_data.copy(),
                          show=show,
                          save=save)
            else:
                show_bayesian(pdfs=self.pdfs,
                              act=actions_to_explore[-1],
                              reduced_train_data=reduced_belief_data.copy(),
                              reduced_test_data=reduced_test_data.copy(),
                              folder="",
                              show=show,
                              delay=delay,
                              windows_name="progress_bayesian")
            plt.close("all")

        action_accuracies = []
        for act in sorted(self.pdfs.keys()):
            accuracy = self.test_bayesian(
                act=act,
                reduced_belief_data=reduced_belief_data,
                reduced_test_data=reduced_test_data
            )
            action_accuracies += [(act, accuracy)]

        # get accuracies for all types of inclusions

        temp_accs = []
        if bayesian:
            for act in sorted(self.pdfs.keys()):
                accuracy = self.test_bayesian(
                    act=act,
                    reduced_belief_data=reduced_belief_data.copy(),
                    reduced_test_data=reduced_test_data.copy(),
                    task=task
                )
                temp_accs += [(act, accuracy)]
            print("\n\n\nhighest accuracy: {}, avg accuracy: {}\n{}\n\n\n".format(
                sorted(temp_accs, key=lambda x: x[1], reverse=True)[0],
                np.average([x[1] for x in temp_accs]),
                pprint(self.actions)))

        data = dict()
        data['original_belief_data'] = original_belief_data
        data['original_test_data'] = original_test_data
        data['reduced_belief_data'] = reduced_belief_data
        data['reduced_test_data'] = reduced_test_data
        data['original_all_belief_data'] = copy.deepcopy(self.original_belief_data.copy)
        data['original_all_test_data'] = copy.deepcopy(self.original_test_data.copy)
        data['action_benefits'] = action_benefits
        data['unbiased_action_benefits'] = unbiased_action_benefits
        data['ranked_actions'] = ranked_actions
        data['unbiased_ranked_actions'] = unbiased_ranked_actions.copy()
        data['best_accuracies'] = best_accuracies
        data['action_accuracies'] = action_accuracies
        data['running_unbiased_accuracy'] = running_unbiased_accuracy
        data['best_running_action'] = best_running_action
        data['iteration_rankings'] = iteration_rankings
        data['iteration_benefits'] = iterations_b_coeffs
        data['results_folder'] = self.results_folder
        data['actions'] = self.actions.copy()
        data['pdfs'] = copy.deepcopy(self.pdfs)

        return data

        # if environment is not None:
        #     self.location_stack, self.location_labels = self._create_location_stack()
        #     self.location_inference = []
        #     self.inference_progress = 0  # counter towards going through all the elements in the location stack
        #     self.test_data = np.zeros((len(self.location_stack), self.dimensionality_reduction))
        # else:
        #     print("Warning: no environment has been given, so the robo-Doctor will only palpate the area "
        #           "below the palpation starting position.")

    # save current stacked motion, and clear
    def save(self):
        self.skin_buffer = np.zeros((self.sensing_resolution,) + self.loaded_skin_shape)
        self.skin_buffer_idx = 0

    def _create_location_stack(self):
        location_stack = []
        location_labels = []
        for key in self.environment.keys():
            for loc in self.environment[key]:
                location_stack += [loc]
                location_labels += [key]
        return location_stack, location_labels

    def _load_experiment_details(self):
        json_files = list(np.sort([file for file in os.listdir(self.data_folder) if file.endswith(".json")]))
        json_filename = self.data_folder + json_files.pop()
        with open(json_filename, 'r') as f:
            experiment_info = json.load(f)
        self.actions = experiment_info["actions"]
        self.objects = experiment_info["objects"]

        self.task_objects = np.unique([obj[self.task] for obj in self.objects])
        self.task_object_map = {}
        for obj in self.objects:
            if obj[self.task] not in self.task_object_map.keys():
                self.task_object_map[obj[self.task]] = [obj]
            else:
                self.task_object_map[obj[self.task]] += [obj]

        self.samples_per_pdf = experiment_info["samples_per_pdf"]
        self.sensing_resolution = experiment_info["sensing_resolution"]-1
        self.number_of_actions = len(self.actions)
        self.number_of_objects = len(self.objects)

        self.missing_actions = []
        self.missing_objects = {}

        self.loaded_skin_shape = tuple(experiment_info["skin_shape"])

    def _load_initial_data(self, test_object_number=1, initial_sample_number=2, test_idxs=None):
        # ------------ Load Data from files ----------

        if test_object_number > len(self.task_object_map[self.task_objects[0]]):
            raise ValueError("The number of test objects is too large, there are only {} objects for each class,"
                             "so the number of test objects must be striclty smaller."
                             .format(len(self.task_object_map[self.task_objects[0]])
            ))

        self.test_object_number = test_object_number

        self.test_objects = {}
        self.belief_objects = {}
        for task_obj in self.task_objects:
            if test_idxs is None:
                test_idxs = np.random.choice(len(self.task_object_map[task_obj]), 2, replace=False)
            self.test_objects[task_obj] = [obj for i, obj in enumerate(self.task_object_map[task_obj])if i in test_idxs]
            self.belief_objects[task_obj] = [obj for obj in self.task_object_map[task_obj] if obj not in self.test_objects[task_obj]]

        # skin_data filenames
        self.skin_files = list(np.sort([file for file in os.listdir(self.data_folder) if file.endswith(".h5")]))
        # indexes to find initial and test data in data_folder
        sample_belief_idxs = {}
        sample_test_idxs = {}
        self.moving_idxes = {}  # need this to know which file to retrieve at the "next" iteration during run

        self.original_belief_data = {}
        self.original_test_data = {}
        
        # indeces to remember where in the original_data[act] are the datapoints belonging to specific objects
        self.pdfs_belief_indeces = {}
        self.pdfs_test_indeces = {}

        # number of times the actions have been explored
        self.action_counts = {}

        self.pcas = {}
        self.normalize_max = {}
        self.normalize_min = {}

        for act, _, _, _ in self.actions:
            if act not in sample_belief_idxs.keys() or act not in sample_test_idxs.keys():
                sample_belief_idxs[act] = {}
                sample_test_idxs[act] = {}
                self.moving_idxes[act] = list(range(self.samples_per_pdf))
                self.missing_objects[act] = []
            for obj_task in self.task_objects:
                for obj in self.task_object_map[obj_task]:
                    obj_act_indexes = list(range(self.samples_per_pdf))
                    for sample_no in list(range(self.samples_per_pdf)):
                        # get filename and check if exists...
                        skin_filename = self.data_folder + "{}_{}_{}.h5".format(obj, act, sample_no)
                        if skin_filename.split("/")[-1] not in self.skin_files:
                            obj_act_indexes.remove(sample_no)
                            if sample_no in self.moving_idxes[act]: self.moving_idxes[act].remove(sample_no)
                    if initial_sample_number > len(obj_act_indexes):
                        if len(obj_act_indexes) == 0:
                            self.missing_actions += [act]
                            self.missing_objects[act] += [obj]
                        else:
                            raise SystemError("there isn't enough data to perform experiments in folder '{}'"
                                              .format(self.data_folder))
                    else:
                        if obj in self.belief_objects[obj_task]:
                            sample_belief_idxs[act][obj] = obj_act_indexes[0:initial_sample_number]
                            [self.moving_idxes[act].remove(x) for x in sample_belief_idxs[act][obj]
                             if x in self.moving_idxes[act]]
                        else:
                            sample_test_idxs[act][obj] = obj_act_indexes[:]
                            # [self.moving_idxes[act].remove(x) for x in sample_test_idxs[act][obj]
                            #  if x in self.moving_idxes[act]]

        # ------------ Dimensionality Reduction --------
        # fit pca to all gathered data - i.e. choose eigenvector directions in hyperspace
        # also, save indeces to create pdfs after
        self.belief_idx = {}
        self.test_idx = {}
        for act, _, _, _ in self.actions:
            if act not in self.missing_actions:
                if act not in self.original_belief_data.keys() or act not in self.original_test_data.keys():
                    self.belief_idx[act] = 0
                    self.test_idx[act] = 0
                    self.original_belief_data[act] = np.zeros((
                        self.number_of_objects * initial_sample_number,
                        self.sensing_resolution * np.prod(list(self.loaded_skin_shape))
                    ))
                    self.original_test_data[act] = np.zeros((
                        self.test_object_number * self.samples_per_pdf *2,
                        self.sensing_resolution * np.prod(list(self.loaded_skin_shape))
                    ))
                    self.action_counts[act] = 0
                    self.pdfs_belief_indeces[act] = {}
                    self.pdfs_test_indeces[act] = {}

                for obj_task in self.task_objects:
                    for obj in self.task_object_map[obj_task]:
                        idx_belief_sample_start = self.belief_idx[act]
                        idx_test_sample_start = self.test_idx[act]
                        if obj not in self.missing_objects[act]:
                            # randomize sample numbers, so non-biased distribution for training/testing
                            if obj in sample_belief_idxs[act].keys():
                                sample_idxes = sample_belief_idxs[act][obj]
                            else:
                                sample_idxes = sample_test_idxs[act][obj]

                            for sample_no in sample_idxes:
                                # get filename and check if exists...
                                skin_filename = self.data_folder + "{}_{}_{}.h5".format(obj, act, sample_no)
                                if skin_filename.split("/")[-1] in self.skin_files:

                                    self.skin_files.remove(skin_filename.split("/")[-1])  # pop data so we don't re-load it

                                    skin_file = tables.open_file(skin_filename, mode='r')
                                    skin_data = skin_file.root.data[:]
                                    skin_file.close()

                                    skin_data = skin_data - skin_data[0]

                                    if obj in sample_belief_idxs[act].keys() and sample_no in sample_belief_idxs[act][obj]:
                                        self.original_belief_data[act][self.belief_idx[act], :] = skin_data[1:].flatten()
                                        self.belief_idx[act] += 1
                                    elif obj in sample_test_idxs[act].keys() and sample_no in sample_test_idxs[act][obj]:
                                        self.original_test_data[act][self.test_idx[act], :] = skin_data[1:].flatten()
                                        self.test_idx[act] += 1
                                else:
                                    raise SystemError("there is missing data in the folder! I can't find '{}'.".format(skin_filename))

                        self.pdfs_belief_indeces[act][obj] = [(idx_belief_sample_start, self.belief_idx[act])]
                        self.pdfs_test_indeces[act][obj] = [(idx_test_sample_start, self.test_idx[act])]

                self.action_counts[act] += 1

        # normalize_min = np.min(self.original_belief_data, axis=0)
        # normalize_max = np.max(self.original_belief_data, axis=0)
        # reduced_act_belief_data = (self.original_belief_data - normalize_min) / (
        #         normalize_max - normalize_min)  # normalize
        # self.pca = PCA(n_components=self.dimensionality_reduction)
        # self.pca.fit(reduced_act_belief_data)

        return self.original_belief_data.copy(), self.original_test_data.copy()

    def _load_next_experiment(self, act=0):

        if len(self.moving_idxes[act]) == 0:
            return None
        sample_no = self.moving_idxes[act].pop()

        for obj_task in self.task_objects:
            for obj in self.belief_objects[obj_task]:
                idx_belief_sample_start = self.belief_idx[act]

                # get filename and check if exists...
                skin_filename = self.data_folder + "{}_{}_{}.h5".format(obj, act, sample_no)
                if skin_filename.split("/")[-1] in self.skin_files:

                    self.skin_files.remove(skin_filename.split("/")[-1])  # pop data so we don't re-load it

                    skin_file = tables.open_file(skin_filename, mode='r')
                    skin_data = skin_file.root.data[:]
                    skin_file.close()

                    skin_data = skin_data - skin_data[0]

                    self.original_belief_data[act] = np.append(arr=self.original_belief_data[act],
                                                          values=skin_data[1:].reshape(1, -1),
                                                          axis=0)
                    self.belief_idx[act] += 1
                else:
                    return None

                self.pdfs_belief_indeces[act][obj] += [(idx_belief_sample_start, self.belief_idx[act])]
        self.action_counts[act] += 1

        return self.original_belief_data.copy()

    def _fit_pdfs(self, belief_data, test_data, act=None, reduced_belief_data=None, reduced_test_data=None, task=0,
                  original_belief_data=None, original_test_data=None, test_mode=False):

        # ------------ Fit Likelihoods --------------------
        if self.pdfs is None:
            self.pdfs = dict()
            self.prior = 1. / len(self.task_objects)

        if act is not None and reduced_test_data is not None and reduced_belief_data is not None\
                and original_belief_data is not None and original_test_data is not None:
            iteration_actions = [(act, None, None, None)]
        else:
            iteration_actions = self.actions.copy()
            reduced_belief_data = dict()
            reduced_test_data = dict()
            original_belief_data = dict()
            original_test_data = dict()

        # n_cols = all_data.shape[1]
        for act, _, _, _ in iteration_actions:
            if act not in self.missing_actions:
                if act not in self.pdfs.keys():
                    self.pdfs[act] = {}
                if act not in reduced_test_data.keys():
                    reduced_belief_data[act] = {}
                    reduced_test_data[act] = {}

                mat_obj_belief_copy = {}
                mat_obj_test_copy = {}
                for task_obj in self.task_objects:
                    for obj in self.task_object_map[task_obj]:
                        if obj not in self.missing_objects[act]:
                            # TRAIN DATA - multiple set of indexes added at different times beacuse robot keeps training
                            for i in range(len(self.pdfs_belief_indeces[act][obj])):
                                if task_obj not in mat_obj_belief_copy.keys():
                                    mat_obj_belief_copy[task_obj] = belief_data[act][
                                                               self.pdfs_belief_indeces[act][obj][0][0]:
                                                               self.pdfs_belief_indeces[act][obj][0][1],
                                                               :
                                                               ].copy()
                                else:
                                    mat_obj_belief_copy[task_obj] = np.append(
                                        arr=mat_obj_belief_copy[task_obj],
                                        values=belief_data[act][
                                               self.pdfs_belief_indeces[act][obj][i][0]:
                                               self.pdfs_belief_indeces[act][obj][i][1],
                                               :].copy(),
                                        axis=0)

                            if task_obj not in mat_obj_test_copy.keys():
                                # TEST DATA - only one (first) set of indexes added at the beginning, robot does not
                                # change testing data
                                mat_obj_test_copy[task_obj] = test_data[act][
                                                       self.pdfs_test_indeces[act][obj][0][0]:
                                                       self.pdfs_test_indeces[act][obj][0][1],
                                                       :
                                                       ].copy()
                            else:
                                mat_obj_test_copy[task_obj] = np.append(
                                    arr=mat_obj_test_copy[task_obj],
                                    values=test_data[act][
                                           self.pdfs_test_indeces[act][obj][0][0]:
                                           self.pdfs_test_indeces[act][obj][0][1],
                                           :].copy(),
                                    axis=0)


                # all_data = np.concatenate((self.original_belief_data[act], self.original_test_data[act]), axis=0)
                # self.normalize_min[act] = np.min(all_data, axis=0)
                # self.normalize_max[act] = np.max(all_data, axis=0)
                # self.normalize_max[act] = [maxx if maxx-minx != 0 else maxx+1
                #                            for minx, maxx in zip(self.normalize_min[act], self.normalize_max[act])]
                # normalized_data = all_data / (self.normalize_max[act] - self.normalize_min[act])
                # self.pcas[act] = PCA(n_components=self.dimensionality_reduction)
                # self.pcas[act].fit(normalized_data)


                self.normalize_min[act] = np.min(self.original_belief_data[act].copy(), axis=0)
                self.normalize_max[act] = np.max(self.original_belief_data[act].copy(), axis=0)
                self.normalize_max[act] = [maxx if maxx-minx != 0 else maxx+1
                                           for minx, maxx in zip(self.normalize_min[act], self.normalize_max[act])]
                normalized_belief_data = (self.original_belief_data[act].copy() - self.normalize_min[act]) / (
                        self.normalize_max[act] - self.normalize_min[act])  # normalize
                self.pcas[act] = PCA(n_components=self.dimensionality_reduction)
                self.pcas[act].fit(normalized_belief_data)

                for task_obj in self.task_objects:
                    mat_obj_belief_copy[task_obj] = (mat_obj_belief_copy[task_obj] - self.normalize_min[act]) / (
                            self.normalize_max[act] - self.normalize_min[act])
                    reduced_belief_mat = self.pcas[act].transform(mat_obj_belief_copy[task_obj])
                    mat_obj_test_copy[task_obj] = (mat_obj_test_copy[task_obj] - self.normalize_min[act]) / (
                            self.normalize_max[act] - self.normalize_min[act])
                    reduced_test_mat = self.pcas[act].transform(mat_obj_test_copy[task_obj])
                    mean = np.mean(reduced_belief_mat, axis=0)
                    cov = np.cov(reduced_belief_mat, rowvar=False)
                    # if len(cov.shape) > 0:
                    #     np.identity(2) * .000001*cov
                    # else:
                    #     cov += .000001
                    self.pdfs[act][task_obj] = (mean, cov)
                    reduced_belief_data[act][task_obj] = reduced_belief_mat
                    reduced_test_data[act][task_obj] = reduced_test_mat
                    original_belief_data[act] = mat_obj_belief_copy
                    original_test_data[act] = mat_obj_test_copy

        return reduced_belief_data, reduced_test_data, original_belief_data, original_test_data

    def test_bayesian(self, act=0, reduced_belief_data=None, reduced_test_data=None, obj_keys=None, task=0, show=False):
        current_samples_per_pdf = reduced_belief_data[act][
            list(reduced_belief_data[act].keys())[0]
        ].shape[0]
        test_samples_per_pdf = reduced_test_data[act][
            list(reduced_test_data[act].keys())[0]
        ].shape[0]

        data_objects = sorted(reduced_belief_data[act].keys())
        label_dict = {}
        for i, obj in enumerate(data_objects):
            label_dict[obj] = i

        if obj_keys is None:
            obj_keys = data_objects

        self.ordered_action_accuracies = []
        # create training and test dataset from dictionaries
        training_data = np.zeros((current_samples_per_pdf * len(obj_keys), self.dimensionality_reduction))
        test_data = np.zeros((test_samples_per_pdf * len(obj_keys), self.dimensionality_reduction))
        training_labels = []
        test_labels = []

        for j, obj in enumerate(obj_keys):
            if obj not in self.missing_objects[act]:
                # TRAINING DATA (currently not used)
                training_data[j * current_samples_per_pdf:j * current_samples_per_pdf + current_samples_per_pdf, :] \
                    = reduced_belief_data[act][obj]
                training_labels += [label_dict[obj]] * current_samples_per_pdf
                # TEST DATA
                test_data[j * test_samples_per_pdf:j * test_samples_per_pdf + test_samples_per_pdf, :] = \
                    reduced_test_data[act][obj]
                test_labels += [label_dict[obj]] * test_samples_per_pdf

        training_labels = np.array(training_labels)

        # fit classifier on data from best motion
        # clf = svm.SVC(kernel='linear', C=1.)
        # clf.fit(training_data, training_labels)
        #
        # predictions = clf.predict(test_data)
        predictions = bayesian_predict(self.pdfs[act], test_data)
        accuracy = np.sum((test_labels == predictions).astype(np.int32))/predictions.shape[0]

        return accuracy

    def _get_metrics(self, reduced_train_data=None, reduced_test_data=None, ordered_actions=None,
                     ordered_action_accuracies=None, action_benefit=None, data_levels=None, show=False):

        fig10 = None
        fig_worst = None
        fig_best = None

        # plot_robot_inference2d(
        #     X_train=training_data,
        #     y_train=training_labels,
        #     X_test=test_data,
        #     y_test=test_labels,
        #     predictions=predictions,
        #     clf=self.clf,
        #     windows_name=self.body.name
        # )

        if ordered_actions is not None and ordered_action_accuracies is not None:
            # save fig 10
            fig10 = plot_fig_10(
                # ordered_actions[::-1],
                ordered_action_accuracies,
                data_levels=data_levels,
                folder=self.results_folder,
                windows_name="",
                show=show,
                save=True
            )

        if reduced_train_data is not None and action_benefit is not None and self.dimensionality_reduction <= 2:
            # UNCOMMENT TO SEE MOTION
            fig_best, fig_worst = plot_normals(
                self.pdfs, reduced_train_data, action_benefit,
                folder=self.results_folder,
                show=show,
                save=True
            )

        # plot_robot_inference2d(
        #     X_train=reduced_train_data,
        #     y_train=reduced_train_labels,
        #     clf=self.clf,
        #     show=show,
        #     windows_name=self.body.name
        # )

        return fig10, fig_best, fig_worst

    def _compute_cpms(self):
        # expected probability matrices for each movement, N -(DxD) matrices
        # where D is the number of objects and N is the number of actions --> should be cpms[action] = np.arfray
        # Kullback-Leibler divergence -> N -(DxD) matrices
        self.cpms = dict()
        for act in self.pdfs.keys():
            if act not in self.missing_actions:
                div_mat = np.zeros((len(self.task_objects), len(self.task_objects)))
                for i, obj1 in enumerate(sorted(self.pdfs[act].keys())):
                    for j, obj2 in enumerate(sorted(self.pdfs[act].keys())):
                        div_mat[i, j] = compute_distance_coefficient(self.pdfs[act][obj1], self.pdfs[act][obj2])
                self.cpms[act] = div_mat
        return True

    def _get_motion_stack(self):
        # returns list of (object, location, action), i.e. stack = [('sd', (1, 2), .001), ...]
        stack = []
        repetitions = self.samples_per_pdf     # number of elements to create pdf for each motion-object pair
        for act in self.actions:
            for obj in self.objects:
                # sample randomly which location to probe, within the ones belonging to the current objects
                indeces = np.random.choice(len(self.environment[obj]), repetitions)
                stack += [(obj, self.environment[obj][idx], act, sample_num) for sample_num, idx in enumerate(indeces)]
        self.total_number_of_experiments = len(stack)
        stack = stack[::-1]
        for i in range(self.total_number_of_experiments-1):
            if stack[i][1] == stack[i+1][1]:
                next_motion = stack[i+1]
                found_element_to_swap = False
                j = i+2
                while not found_element_to_swap and self.total_number_of_experiments > j:
                    if stack[j][1] != next_motion[1]:
                        stack[i+1] = stack[j]
                        stack[j] = next_motion
                        found_element_to_swap = True
                    j += 1

        return stack

    def get_perceived_benefit(self, non_biased=False):
        # N vector
        action_benefit = np.zeros(self.number_of_actions)
        # equation 11
        for i, act in enumerate(sorted(self.pdfs.keys())):
            if act not in self.missing_actions:
                if non_biased is False:
                    expected_uncertainty = 1 - np.sum(self.prior**2/np.sum(self.cpms[act]*self.prior, axis=0))
                    action_benefit[i] = 1 - expected_uncertainty**(1/self.action_counts[act])
                else:
                    action_benefit[i] = np.sum(self.prior**2/np.sum(self.cpms[act]*self.prior, axis=0))
                if math.isinf(action_benefit[i]) and action_benefit[i] < 0:
                    action_benefit[i] *= -1
        return action_benefit

    def get_best_actions(self, non_biased=False):
        # perceived benefit for action
        action_benefits = self.get_perceived_benefit(non_biased=non_biased)
        ordered_actions = [x for _, x in sorted(zip(action_benefits, sorted(self.pdfs.keys())), reverse=True)]

        return action_benefits, ordered_actions


def save_dict_to_file(data=None, path=None, filename=None, format='json'):
        # if don't want to overwrite, check next available name
        file = "{}{}.{}".format(path, filename, format)
        num = 0
        while file_exists(file):
            file = "{}{}-({}).{}".format(path, filename, num, format)
            num += 1
        with open(file, 'w') as exp_file:
            json.dump(data, exp_file)


def compute_distance_coefficient(pdf1, pdf2):
    mu1, sigma1 = pdf1
    mu2, sigma2 = pdf2
    if isinstance(sigma1, np.ndarray) and len(sigma1.shape) > 0 and len(sigma2.shape) > 0:
        sigma1 = np.diag(sigma1)
        sigma2 = np.diag(sigma2)
    coefficient = np.sum(np.sqrt((2*sigma1*sigma2)/(sigma1**2 + sigma2**2)) *
                             np.exp(-(mu1 - mu2)**2/(4*(sigma1**2 + sigma2**2))))
    if math.isnan(coefficient):
        coefficient = 0
    return coefficient


def get_time(sec):
    verbose = ["sec(s)", "min(s)", "hour(s)", "day(s)"]
    time = []
    time += [sec // (24 * 3600)]  # days
    sec %= (24 * 3600)
    time += [sec // 3600]         # hours
    sec %= 3600
    time += [sec // 60]           # minutes
    sec %= 60
    time += [sec]                 # seconds
    time = time[::-1]
    time_output = ""
    for i in range(len(time)):
        val = time.pop()
        tag = verbose.pop()
        if val != 0:
            time_output += "{}{} :".format(int(val), tag)
    return time_output[:-2]

def bayesian_predict(pdfs, data):
    labels = []
    for i in range(data.shape[0]):
        label = 0
        max_prob = 0
        for j, obj in enumerate(sorted(pdfs.keys())):
            mu, sig = pdfs[obj]
            var = multivariate_normal(mean=mu, cov=sig, allow_singular=True)
            prob_density = var.pdf(data[i])

            if prob_density > max_prob:
                max_prob = prob_density
                label = j
        labels += [label]
    return np.array(labels)

