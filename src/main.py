import numpy as np
from touchexperiment import run_learning


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


if __name__ == "__main__":

    # total exploration time approx 18s per touch

    # ------------   ROTATORY MOTION -------------
    # x, y, z, Rx, Ry, Rz
    # eta_min = np.array([0., 0., 1., 1., 1., 0.])
    # eta_max = np.array([0., 0., 1., 1., 1., 0.])
    # A_min = np.array([0., 0., .003, 0., 0., 0.])
    # A_max = np.array([0., 0., .003, np.deg2rad(10), np.deg2rad(10), 0.])  # 80 mm depth


    # ------------   RUBBING MOTION -------------
    # x, y, z, Rx, Ry, Rz
    eta_min = np.array([3., 3., 1., 0., 0, 0.])
    eta_max = np.array([3., 3., 1., 0., 0, 0.])
    A_min = np.array([-.005, -.005, .001, 0., 0, 0.])
    A_max = np.array([.005, .005, .001, 0., 0, 0.])  # 80 mm depth

    morphs = ['0mm', '3mm', '5mm']
    # morphs = ['0mm', '3mm', '5mm']  # [-255, 0, 50, 90, 130]

    # Generate 'number_of_actions' intervals for each of the min-max ranges given -- combinatorial!
    actions = get_action_profile(eta_min, eta_max, A_min, A_max, morphs, number_of_actions=3)


    # r-round, s-square
    # s-smooth, r-rough
    # s-soft, h-hard
    training_environment = {
        'ssh': [(-.37652, -.04383)],
        'srh': [(-.38334, -.11278)],
        'rsh': [(-.44258, -.03139)],
        'rrh': [(-.44944, -.10279)],
        'sss': [(-.50545, -.02674)],
        'srs': [(-.51257, -.09933)],
        'rss': [(-.56858, -.01914)],
        'rrs': [(-.57429, -.09022)]
    }
    testing_environment = {
        'ssh': [(-.37652, -.04383)],
        'srh': [(-.38334, -.11278)],
        'rsh': [(-.44258, -.03139)],
        'rrh': [(-.44944, -.10279)],
        'sss': [(-.50545, -.02674)],
        'srs': [(-.51257, -.09933)],
        'rss': [(-.56858, -.01914)],
        'rrs': [(-.57429, -.09022)]
    }

    run_learning(
        time_interval=0,                    # how much does a step last (sec) -- 0 means as little as possible
        steps=0,                            # number of time steps -- 0 means infinite
        learning=True,                     # if learning then collect data,
        testing=False,
        resume_previous_experiment=False,   # if True the previous experiment if continued from where it was left off
        number_of_samples=1,               # how many times the robot palpates the same location, to learn
        sensing_resolution=4,               # how many skin samples to take during palpation contact time
        palpation_duration=2,               # duration of palpation contact time (sec),
        downtime_between_experiments=12,    # downtime between touches, to reset skin (sec),
        dimensionality_reduction=2,         # dimensionality reduction
        training_ratio=.7,                  # percentage of data for svm training - only matters if learning=False
        task=1,
        training_environment=training_environment,
        testing_environment=testing_environment,  # phantom environm ent - only relevant(used) for testing
        actions=actions,                    # action list the robot must try
        verbose=True,
    )

