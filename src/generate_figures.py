# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import math
import copy
import matplotlib.colors as colors
from main_bayesian_exploration import *
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import datetime

np.random.seed(94325) #1399

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Myriad Pro Regular'

labeltag_dict = {
    "rrh": "Round Rough Hard",
    "srh": "Square Rough Hard",
    "rsh": "Round Smooth Hard",
    "ssh": "Square Smooth Hard",
    "rrs": "Round Rough Soft",
    "srs": "Square Rough Soft",
    "rss": "Round Smooth Soft",
    "sss": "Square Smooth Soft"
}

label_tag_task_dict = {
    0: {
        'r': 'Round',
        's': 'Edged'
    },
    1: {
        'r': 'Rough',
        's': 'Smooth'
    },
    2: {
        'h': 'Hard',
        's': 'Soft',
    }
}


exp_dictionary = {
    0: "Task 1: Object Geometry",
    1: "Task 2: Surface Roughness",
    2: "Task 3: Object Stiffness"
}

# r-round, s-square
# s-smooth, r-rough
# s-soft, h-hard
task_letters = {
    1: ['r', 's'],
    2: ['r', 's'],
    3: ['h', 's']
}


def histogram_plot(hist_array, xlabel='$principal components$', ylabel='$explained variance (%)$', xtick_prefix='p'):
    # init figure
    hist_fig = plt.figure(figsize=(13, 8))
    ax = hist_fig.add_subplot(111)

    # necessary variables
    ind = np.arange(1, len(hist_array) + 1)  # the x locations for the groups
    width = 0.85

    # the bars
    ax.bar(ind, hist_array * 100, width, color='black')

    xTickMarks = ['$' + xtick_prefix + '_{' + str(i) + '}$'  for i in ind]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=34)

    return hist_fig


def generate_pca_histogram(actions=None, original_belief_data=None, folder="", rank=0, show=False, save=True):
    plt.close('all')

    pss = []
    for action in actions:
        obj_keys = sorted(original_belief_data[action[0]].keys())
        dataset = np.zeros((1, original_belief_data[action[0]][obj_keys[0]].shape[1]))
        for obj_key in obj_keys:
            dataset = np.append(dataset, original_belief_data[action[0]][obj_key], axis=0)
        pca = PCA(n_components=min(dataset[1:, :].shape))
        pca.fit(dataset[1:, :])
        ps = np.sort(np.var(pca.transform(dataset[1:, :]), axis=0) / \
                     np.sum(np.var(dataset[1:, :], axis=0)))
        ps = np.concatenate((ps[::-1], np.zeros(dataset[1:, :].shape[0] - len(ps))), axis=0)[:20]
        pss += [ps]
    ps_data = np.array(pss)
    meanP = np.average(ps_data, axis=0)
    errP = np.std(ps_data, axis=0)  # / np.sqrt(ps_data.shape[0])
    hist_fig = plt.figure(figsize=(13, 8))
    ax = hist_fig.add_subplot(111)
    ax.set_xlabel("$principal\ components$", fontsize=28)
    ax.set_ylabel("$explained\ variance (\%)$", fontsize=34)
    ax.set_ylim([0, 100])
    # necessary variables
    ind = np.arange(1, len(meanP) + 1)  # the x locations for the groups
    xTickMarks = ['$p_{' + str(i) + '}$' for i in ind]
    width = 0.85
    ax.bar(ind, meanP * 100, width, color='black', yerr=errP * 100,
           error_kw=dict(ecolor='red', lw=3, capsize=3, capthick=2))
    plt.xticks(ind, xTickMarks, fontsize=22)
    plt.yticks(fontsize=22)
    hist_fig.tight_layout()

    if save:
        filename = '{}{}-pca.png'.format(folder, rank)
        hist_fig.savefig(filename, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return hist_fig


def generate_motion_profile2d(data=None, action=None, legend=False, show=False, save=False,
                              plot_accuracy=False, figure=None, grdspc=None, label_pos=None):
    pdfs = data["pdfs"]
    idx = [i for i, act in enumerate(sorted(pdfs.keys())) if act == action[0]][0]
    accuracy = data["action_accuracies"][idx][1]
    action_benefits = data["unbiased_action_benefits"][idx]
    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
    folder = data["results_folder"]

    if figure is not None:
        fig = figure
        ax = figure.add_subplot(grdspc, projection='3d')
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.gca(projection='3d')

    if plot_accuracy:
            ax.set_title('$\\bf{Rank = ' + str(rank) + '}'+'$\nAccuracy = {:.2f}\n$Bhattacharyya\ coeff.$ = {:.2f}\n'.format(
                accuracy, action_benefits), loc="center", fontsize=42)
    else:
            ax.set_title('$\\bf{Rank = ' + str(rank) + '}'+'$\n$Bhattacharyya\ coeff.$ = {:.2f}\n'.format(
                action_benefits), loc="center", fontsize=42)

    ax.set_xlabel('$Rx\ (deg)$', fontsize=34, labelpad=30)
    ax.set_ylabel('$Ry\ (deg)$', fontsize=34, labelpad=30)
    ax.set_zlabel('$Z\ (mm)$', fontsize=34, labelpad=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.zaxis.set_tick_params(labelsize=30)
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    ax.view_init(azim=-45)

    etas = np.array(action[1])
    alphas = np.array(action[2])

    increments = 0.01
    maximum_time = 3.  # 3 seconds
    current_time = 0.00

    positions = []

    max_iterations = np.floor((maximum_time - current_time) / increments)
    current_iteration = 0

    middle_legend_done = False

    robot_position = np.array([0., 0., 0.])
    robot_rotation = np.array([0., 0., 0.])

    zs = []
    rxs = []
    rys = []

    initial_position = alphas * np.cos(etas * current_time)

    while current_time <= maximum_time:

        current_time += increments

        if 35 > current_iteration:
            color = 'green'
            alpha = .7
        elif 4 > max_iterations - current_iteration:
            color = 'red'
            alpha = .7
        else:
            color = 'grey'
            alpha = .25

        if current_iteration == 0:
            legend_text = "initial end-effector pose"
        elif current_iteration == max_iterations - 1:
            legend_text = "final  end-effector pose"
        elif middle_legend_done is False and color == 'grey':
            legend_text = "palpation's end-effector pose"
            middle_legend_done = True
        else:
            legend_text = False

        velocities = -alphas * etas * np.sin(etas * current_time)
        velocities = np.array([0 if math.isnan(elem) or math.isinf(elem) else elem for elem in
                               velocities])
        current_position = alphas * np.cos(etas * current_time)
        robot_position += velocities[:3] * current_time * 1000  # in mm
        robot_rotation += velocities[3:] * current_time  # in radiants

        positions += [robot_position]

        x, y, z = robot_position
        Rx, Ry, Rz = robot_rotation

        zs += [(current_position[2] - initial_position[2]) * 1000]
        rxs += [(current_position[3] - initial_position[0]) * 1000]
        rys += [(current_position[4] - initial_position[1]) * 1000]

        # x2 = x + np.sin(Ry)
        # y2 = y + -np.sin(Rx ) *np.cos(Ry)
        # z2 = z + np.cos(Rx ) *np.cos(Ry)

        if legend_text:
            ax.plot([0, rxs[-1]], [0, rys[-1]], [zs[-1] - initial_position[2], zs[-1] - initial_position[2] + 10],
                    color=color, alpha=alpha, label=legend_text, zorder=10)
        else:
            ax.plot([0, rxs[-1]], [0, rys[-1]], [zs[-1] - initial_position[2], zs[-1] - initial_position[2] + 10],
                    color=color, alpha=alpha, zorder=10)

        current_iteration += 1
    ax.scatter(rxs- initial_position[0] + 10, rys- initial_position[1] + 10, zs - initial_position[2] + 10, label="Rx, Ry, z robot control", zorder=0, color='blue')
    ax.view_init(azim=30)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(handle) for handle in handles]
        [handle.set_linewidth(10) for handle in handles]
        [handle.set_alpha(1) for handle in handles]
        if label_pos == 'left':
            ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(4.36, 0.),
                        ncol=4, fancybox=True)
        else:
            ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(4.86, 0.),
                        ncol=4, fancybox=True)

    if save:
        filename = '{}{}_{}-action_profile.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def generate_motion_profile1d(data=None, action=None, legend=False, show=False, save=False,
                            plot_accuracy=False, figure=None, grdspc=None, label_pos=None):
    pdfs = data["pdfs"]
    idx = [i for i, act in enumerate(sorted(pdfs.keys())) if act == action[0]][0]
    accuracy = data["action_accuracies"][idx][1]
    action_benefits = data["unbiased_action_benefits"][idx]
    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
    folder = data["results_folder"]

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grdspc)
        ax = fig.add_subplot(grid[:, :])
        # ax = figure.add_subplot(grdspc)
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.add_subplot(111)

    if plot_accuracy:
        ax.set_title(
            '$\\bf{Rank = ' + str(rank) + '}' + '$\nAccuracy = {:.2f}\n$Bhattacharyya\ coeff.$ = {:.2f}\n'.format(
                accuracy, action_benefits), loc="center", fontsize=42)
    else:
        ax.set_title('$\\bf{Rank = ' + str(rank) + '}' + '$\n$Bhattacharyya\ coeff.$ = {:.2f}\n'.format(
            action_benefits), loc="center", fontsize=42)
    ax.set_xlabel('$time\ (ms)$', fontsize=48)
    ax.set_ylabel('$space\ (mm)$', fontsize=48)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)

    etas = np.array(action[1])
    alphas = np.array(action[2])

    increments = 0.01
    maximum_time = 3.  # 3 seconds
    current_time = 0.00

    positions = []

    max_iterations = np.floor((maximum_time - current_time) / increments)
    current_iteration = 0

    middle_legend_done = False

    robot_position = np.array([0., 0., 0.])
    robot_rotation = np.array([0., 0., 0.])

    xs = []
    ys = []
    zs = []

    initial_position = alphas * np.cos(etas * current_time)

    while current_time <= maximum_time:

        current_time += increments

        velocities = -alphas * etas * np.sin(etas * current_time)
        velocities = np.array([0 if math.isnan(elem) or math.isinf(elem) else elem for elem in
                               velocities])
        current_position = alphas * np.cos(etas * current_time)
        robot_position += velocities[:3] * current_time * 1000  # in mm
        robot_rotation += velocities[3:] * current_time  # in radiants

        positions += [robot_position]

        xs += [(current_position[0] - initial_position[0]) * 1000]
        ys += [(current_position[1] - initial_position[1]) * 1000]
        zs += [(current_position[2] - initial_position[2]) * 1000]

        current_iteration += 1

    ax.plot(np.array((range(len(xs)))) * 10, xs, label="X", linewidth=3)
    ax.plot(np.array((range(len(ys)))) * 10, ys, label='Y')
    ax.plot(np.array((range(len(zs)))) * 10, zs, label='Z')

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(handle) for handle in handles]
        if label_pos == 'left':
            ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(4.36, 0.),
                        ncol=4, fancybox=True)
        else:
            ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(2.9, -0.2),
                        ncol=4, fancybox=True)

    if save:
        filename = '{}{}_{}-action_profile.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def generate_control_profile(action=None, rank=None, folder='', legend=False, show=False, save=False):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$time\ (ms)$', fontsize=16)
    ax.set_ylabel('$space\ (mm/deg)$', fontsize=16)

    etas = np.array(action[1])
    alphas = np.array(action[2])

    increments = 0.01
    maximum_time = 6.  # 3 seconds
    current_time = 0.00

    positions = []

    max_iterations = np.floor((maximum_time - current_time) / increments)
    current_iteration = 0

    middle_legend_done = False

    robot_position = np.array([0., 0., 0.])
    robot_rotation = np.array([0., 0., 0.])

    zs = []
    rxs = []
    rys = []

    initial_position = alphas * np.cos(etas * current_time)

    while current_time <= maximum_time:

        current_time += increments

        if 35 > current_iteration:
            color = 'green'
        elif 4 > max_iterations - current_iteration:
            color = 'red'
        else:
            color = 'grey'

        if current_iteration == 0:
            legend_text = "initial end-effector pose"
        elif current_iteration == max_iterations - 1:
            legend_text = "final  end-effector pose"
        elif middle_legend_done is False and color == 'grey':
            legend_text = "palpation's end-effector pose"
            middle_legend_done = True
        else:
            legend_text = False

        velocities = -alphas * etas * np.sin(etas * current_time)
        velocities = np.array([0 if math.isnan(elem) or math.isinf(elem) else elem for elem in
                               velocities])
        current_position = alphas * np.cos(etas * current_time)
        robot_position += velocities[:3] * current_time * 1000  # in mm
        robot_rotation += velocities[3:] * current_time  # in radiants

        positions += [robot_position]

        zs += [(current_position[2] - initial_position[2]) * 1000]
        rxs += [np.rad2deg(current_position[3] - initial_position[3])]
        rys += [np.rad2deg(current_position[4] - initial_position[4])]


        current_iteration += 1

    plt.plot(np.array((range(len(rxs))))*10, rxs, label="rx")
    plt.plot(np.array((range(len(rys))))*10, rys, label='ry')
    plt.plot(np.array((range(len(zs))))*10, zs, label='Z')

    ax.legend(fontsize=28, loc='upper right', bbox_to_anchor=(0, 1.22), ncol=4, fancybox=True)

    if save:
        filename = '{}{}_{}-control_action_profile.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def generate_belief_state2d(data=None, action=None, legend=False, folder="", show=False, save=False,
                            figure=None, grdspc=None):

    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
    if figure is not None:
        fig = figure
        ax = figure.add_subplot(grdspc)
    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        ax = fig.add_subplot(111)
    # ax_1d = fig.add_subplot(grid[-1, :], sharex=ax_gauss)

    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$\\vec{p}_2$', fontsize=48)
    ax.set_xticks(())
    ax.set_yticks(())

    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000', '#4d0000', '#d2d916']
    pdfs = data['pdfs']
    reduced_train_data = data['reduced_belief_data']
    reduced_test_data = data['reduced_test_data']

    act = action[0]
    objects = sorted(list(pdfs[act].keys()))
    all_object_keys = sorted(list(label_tag_task_dict[task].keys()))
    unlabelled_object_keys = list(label_tag_task_dict[task].keys())

    X_train = None
    y_train = []

    for obj in reduced_train_data[act].keys():
        if X_train is None:
            X_train = reduced_train_data[act][obj]
        else:
            X_train = np.append(arr=X_train,
                                values=reduced_train_data[act][obj],
                                axis=0)
        y_train += [all_object_keys.index(obj)] * \
                   reduced_train_data[act][obj].shape[0]

    X_test = None
    y_test = []
    if reduced_test_data is not None:
        for obj in reduced_test_data[act].keys():
            if X_test is None:
                X_test = reduced_test_data[act][obj]
            else:
                X_test = np.append(arr=X_test,
                                   values=reduced_test_data[act][obj],
                                   axis=0)
            y_train += [objects.index(obj)] * \
                       reduced_test_data[act][obj].shape[0]

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # print object evidence and labels (if not labelled yut)
    for j in range(len(X0)):
        if all_object_keys[y_train[j]] in unlabelled_object_keys:
            ax.scatter(X0[j], X1[j], c=colors[y_train[j]], s=85, edgecolors='k',
                       label=label_tag_task_dict[task][all_object_keys[y_train[j]]])
            unlabelled_object_keys.remove(all_object_keys[y_train[j]])
        else:
            ax.scatter(X0[j], X1[j], c=colors[y_train[j]], s=85, edgecolors='k')

    if pdfs is not None:
        for j, obj in enumerate(objects):
            if len(pdfs[act][obj][1].shape) > 0:
                vals, vecs = np.linalg.eigh(pdfs[act][obj][1])  # Compute eigenvalues and associated eigenvectors
                x, y = vecs[:, 0]
                theta = np.degrees(np.arctan2(y, x))
                hws = []
                hws += [tuple(2 * np.sqrt(vals))]
                hws += [tuple(4 * np.sqrt(vals))]
                for w, h in hws:
                    circle = mpatches.Ellipse(xy=pdfs[act][obj][0],
                                              width=w,
                                              height=h,
                                              angle=theta,
                                              fill=False,
                                              color=colors[j],
                                              alpha=.35,
                                              linestyle='-',
                                              linewidth=4,
                                              label=label_tag_task_dict[task][obj])
                    ax.add_artist(circle)

    resize_ratio = 1.6
    axes = np.array(plt.axis())
    x_ctr = np.average(axes[:2])
    y_ctr = np.average(axes[2:])
    half_w = (axes[1] - axes[0]) / 2
    half_h = (axes[3] - axes[2]) / 2
    ax.set_xlim([x_ctr - half_w * resize_ratio, x_ctr + half_w * resize_ratio])
    ax.set_ylim([y_ctr - half_h * resize_ratio, y_ctr + half_h * resize_ratio])

    # plt.legend(fontsize=12, bbox_to_anchor=(1., 1.3), ncol=2, fancybox=True)

    # if legend:
    handles, labels = ax.get_legend_handles_labels()

    if save:
        filename = '{}{}_{}-belief_state.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return handles, labels


def generate_belief_state(data=None, action=None, legend=False, folder="", show=False, save=False,
                            figure=None, grdspc=None, filename=None):
    pdfs = data['pdfs']
    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
    reduced_data = data['reduced_belief_data']
    # reduced_test_data = data['reduced_test_data']
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000']

    act = action[0]

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=grdspc)
    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ax_gauss = fig.add_subplot(grid[:-1, :])
    ax_1d = fig.add_subplot(grid[-1, :], sharex=ax_gauss)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax_1d.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax_gauss.set_ylabel('$p(x)$', fontsize=48)
    plt.autoscale(True)

    ax_gauss.yaxis.set_tick_params(labelsize=30)
    ax_1d.xaxis.set_tick_params(labelsize=30)
    ax_1d.yaxis.set_ticks([])

    global_min = None
    global_max = None

    for obj in pdfs[act].keys():
        mu, sig = pdfs[act][obj]
        xmin = mu - 4 * sig
        xmax = mu + 4 * sig

        if global_min is None or global_min > xmin:
            global_min = xmin
        if global_max is None or global_max < xmax:
            global_max = xmax

    if global_min == global_max:
        xs = [0] * 100
    else:
        xs = np.arange(global_min, global_max, (global_max - global_min) / 100)

    for i, obj in enumerate(pdfs[act].keys()):
        mu, sig = pdfs[act][obj]
        ax_1d.plot(
            reduced_data[act][obj][:, 0], [0] * len(reduced_data[act][obj][:, 0]),
            marker='+',
            linestyle='None',
            markersize=25,
            markerfacecolor=colors[i],
            markeredgewidth=4,
            markeredgecolor=colors[i],
            label="{}".format(label_tag_task_dict[task][obj]),
            alpha=0.7
        )
        # ax_1d.plot(
        #     reduced_test_data[act][obj][:, 0], [0] * len(reduced_test_data[act][obj][:, 0]),
        #     marker='*',
        #     linestyle='None',
        #     markersize=25,
        #     markerfacecolor=colors[i],
        #     markeredgewidth=4,
        #     markeredgecolor=colors[i],
        #     alpha=0.7
        # )

        ax_gauss.plot(xs, gaussian(xs, mu, sig).flatten(), label=label_tag_task_dict[task][obj], c=colors[i], linewidth=4)

    # if legend:
    handles1, labels1 = ax_gauss.get_legend_handles_labels()
    handles2, labels2 = ax_1d.get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles1 + handles2]
    [handle.set_linewidth(7) for handle in handles]
    [handle.set_alpha(1) for handle in handles]

    # ax_1d.legend(handles, labels1 + labels2, fontsize=42, loc='upper right', bbox_to_anchor=(3.68, -1.5),
    #            ncol=6, fancybox=True)

    if save:
        if filename is None:
            filename = '{}{}_{}-belief_state.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return handles, labels1 + labels2

def generate_belief_state_act_morph1d(data=None, morph='0mm', task=0, legend=False, show=False, save=True):

    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']
    morph_actions = [act for act in data['actions'] if act[-1] == morph]
    morph_action_idxs = [act[0] for act in morph_actions]

    best_act = None
    worst_act = None
    for act in ranked_actions:
        if best_act is None and act in morph_action_idxs:
            best_act = [action for (action, idx) in zip(morph_actions, morph_action_idxs) if idx==act][0]
        if act in morph_action_idxs:
            worst_act = [action for (action, idx) in zip(morph_actions, morph_action_idxs) if idx==act][0]

    for i, action in enumerate([best_act, worst_act]):
        generate_belief_state(
            data=data,
            action=action,
            legend=legend,
            folder=folder,
            filename='{}belief_state_labels.png'.format(folder),
            show=False,
            save=True)


def generate_belief_state_act_morph2d(data=None, morph='0mm', task=0, legend=False, show=False, save=True):

    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']
    action_accuracies = data['action_accuracies']
    ranked_accuracies = []
    for act in ranked_actions:
        ranked_accuracies += [int(accuracy[1] * 100) for accuracy in action_accuracies if accuracy[0] == act]
    morph_actions = [act for act in data['actions'] if act[-1] == morph]
    morph_action_idxs = [act[0] for act in morph_actions]

    best_act = None
    worst_act = None
    for act in ranked_actions:
        if best_act is None and act in morph_action_idxs:
            best_act = [action for (action, idx) in zip(morph_actions, morph_action_idxs) if idx==act][0]
        if act in morph_action_idxs:
            worst_act = [action for (action, idx) in zip(morph_actions, morph_action_idxs) if idx==act][0]

    for i, action in enumerate([best_act, worst_act]):
        plt.close('all')

        rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
        acc = ranked_accuracies[rank]

        fig = plt.figure(figsize=(7, 5), constrained_layout=True)
        ax = fig.add_subplot(111)
        # ax_1d = fig.add_subplot(grid[-1, :], sharex=ax_gauss)

        ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
        ax.set_ylabel('$\\vec{p}_2$', fontsize=48)
        ax.set_xticks(())
        ax.set_yticks(())

        colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000', '#4d0000', '#d2d916']
        pdfs = data['pdfs']
        reduced_train_data = data['reduced_belief_data']
        reduced_test_data = data['reduced_test_data']

        act = action[0]
        objects = sorted(list(pdfs[act].keys()))
        all_object_keys = sorted(list(label_tag_task_dict[task].keys()))
        unlabelled_object_keys = list(label_tag_task_dict[task].keys())

        X_train = None
        y_train = []

        for obj in reduced_train_data[act].keys():
            if X_train is None:
                X_train = reduced_train_data[act][obj]
            else:
                X_train = np.append(arr=X_train,
                                    values=reduced_train_data[act][obj],
                                    axis=0)
            y_train += [all_object_keys.index(obj)] * \
                       reduced_train_data[act][obj].shape[0]

        X_test = None
        y_test = []
        if reduced_test_data is not None:
            for obj in reduced_test_data[act].keys():
                if X_test is None:
                    X_test = reduced_test_data[act][obj]
                else:
                    X_test = np.append(arr=X_test,
                                       values=reduced_test_data[act][obj],
                                       axis=0)
                y_train += [objects.index(obj)] * \
                           reduced_test_data[act][obj].shape[0]

        X0, X1 = X_train[:, 0], X_train[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        # print object evidence and labels (if not labelled yut)
        for j in range(len(X0)):
            if all_object_keys[y_train[j]] in unlabelled_object_keys:
                ax.scatter(X0[j], X1[j], c=colors[y_train[j]], s=85, edgecolors='k',
                           label=label_tag_task_dict[task][all_object_keys[y_train[j]]])
                unlabelled_object_keys.remove(all_object_keys[y_train[j]])
            else:
                ax.scatter(X0[j], X1[j], c=colors[y_train[j]], s=85, edgecolors='k')

        if pdfs is not None:
            for j, obj in enumerate(objects):
                if len(pdfs[act][obj][1].shape) > 0:
                    vals, vecs = np.linalg.eigh(pdfs[act][obj][1])  # Compute eigenvalues and associated eigenvectors
                    x, y = vecs[:, 0]
                    theta = np.degrees(np.arctan2(y, x))
                    hws = []
                    hws += [tuple(2 * np.sqrt(vals))]
                    hws += [tuple(4 * np.sqrt(vals))]
                    for w, h in hws:
                        circle = mpatches.Ellipse(xy=pdfs[act][obj][0],
                                                  width=w,
                                                  height=h,
                                                  angle=theta,
                                                  fill=False,
                                                  color=colors[j],
                                                  alpha=.35,
                                                  linestyle='-',
                                                  linewidth=4,
                                                  label=label_tag_task_dict[task][obj])
                        ax.add_artist(circle)

        resize_ratio = 1.6
        axes = np.array(plt.axis())
        x_ctr = np.average(axes[:2])
        y_ctr = np.average(axes[2:])
        half_w = (axes[1] - axes[0]) / 2
        half_h = (axes[3] - axes[2]) / 2
        ax.set_xlim([x_ctr - half_w * resize_ratio, x_ctr + half_w * resize_ratio])
        ax.set_ylim([y_ctr - half_h * resize_ratio, y_ctr + half_h * resize_ratio])

        if legend and i==0:
            handles, labels = ax.get_legend_handles_labels()

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, frameon=False)
            ax2.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
            ax2.legend(handles=handles, labels=labels, fontsize=12, loc='lower center', ncol=4, fancybox=True)
            filename = '{}belief_state_labels.png'.format(folder)
            fig2.savefig(filename, bbox_inches="tight", dpi=300)

        if save:
            filename = '{}{}-{}_{}-{:.2f}-belief_state.png'.format(folder, morph, rank, action[0], acc*100)
            fig.savefig(filename, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

    return True


def generate_figure_3(data, show=False, save=True):

    plt.close('all')
    actions = data['actions']
    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(65, 25), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    grid = gridspec.GridSpec(ncols=4, nrows=9, figure=fig)  #, wspace=.8, hspace=.1
    grid.update(wspace=0.25, hspace=0.35)  # set the spacing between axes.

    y_coord_labels = ['A', 'B']
    label_grids = [grid[0:4, :], grid[5:, :]]
    for i in range(2):
        ax_main = fig.add_subplot(label_grids[i], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=52, fontweight='bold', labelpad=135)

    handles, labels = (None, None)
    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    for i, idx in enumerate(ranked_idxs_to_extract):
        generate_motion_profile2d(action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                  data=data,
                                  show=show,
                                  legend=idx == 0,
                                  save=False,
                                  figure=fig,
                                  grdspc=grid[0:4, i],
                                  label_pos='left')
        print("generated subfig {}-{} {}, out of {} done".format(1, i, i*3+2, 12))
        handles, labels = generate_belief_state(data=data,
                              action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                              folder=folder,
                              show=show,
                              legend=idx == 0,
                              save=False,
                              figure=fig,
                              grdspc=grid[5:, i])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))

    if len(data['pdfs'][list(data['pdfs'].keys())[0]].keys()) == 4:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.5, -0.17), ncol=4, fancybox=True)
    else:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.48, -0.17), ncol=6, fancybox=True)

    if save:
        filename = '{}Figure3.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=100)
        plt.close('all')
    return fig


def generate_figure_6(data, show=False, save=True):

    plt.close('all')
    actions = data['actions']
    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(55, 20))
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    grid = gridspec.GridSpec(ncols=4, nrows=20, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.2, hspace=0.1)  # set the spacing between axes.

    y_coord_labels = ['A', 'B', 'C']
    label_grids = [grid[0:5, :], grid[7:13, :], grid[13:, :]]
    for i in range(3):
        ax_main = fig.add_subplot(label_grids[i], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=52, fontweight='bold', labelpad=135)

    handles, labels = (None, None)
    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    for i, idx in enumerate(ranked_idxs_to_extract):
        generate_motion_profile1d(action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                  data=data,
                                  show=show,
                                  legend=idx == 0,
                                  save=False,
                                  plot_accuracy=True,
                                  figure=fig,
                                  grdspc=grid[0:5, i])
        print("generated subfig {}-{} {}, out of {} done".format(1, i, i*3+2, 12))
        generate_raw_data_figure(data=data,
                                 action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                 rank=idx,
                                 folder=folder,
                                 show=show,
                                 colorbar=i == len(ranked_idxs_to_extract)-1,
                                 save=False,
                                 figure=fig,
                                 grdspc=grid[7:13, i])
        print("generated subfig {}-{} {}, out of {} done".format(0, i, i*3+1, 12))
        handles, labels = generate_belief_state2d(data=data,
                              action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                              folder=folder,
                              show=show,
                              legend=idx == 0,
                              save=False,
                              figure=fig,
                              grdspc=grid[13:, i])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))


    if len(data['pdfs'][list(data['pdfs'].keys())[0]].keys()) == 4:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.5, -0.17), ncol=4, fancybox=True)
    else:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.48, -0.17), ncol=6, fancybox=True)

    if save:
        filename = '{}Figure6.png'.format(folder)
        print("tring to generate Fig. 6....")
        fig.savefig(filename, bbox_inches="tight", dpi=100)
        print("done")
        plt.close('all')
    return fig


def generate_figure_4(all_bayesian_data, all_systematic_data, show=False, save=True):

    plt.close('all')
    folder = all_bayesian_data[list(all_bayesian_data.keys())[0]]['results_folder']

    fig = plt.figure(figsize=(70, 20), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    grid = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.3, hspace=0.5)  # set the spacing between axes.

    col_no = 0
    sub_axes = []
    main_xlim = None
    main_ylim = None
    handles, labels = (None, None)
    for exp_no in all_bayesian_data.keys():
        bayesian_data = all_bayesian_data[exp_no]

        ax_sub, handles, labels, xlim, ylim = generate_figure_ranked_acc(exp_number=exp_no,
                                                     data=bayesian_data,
                                                     legend=col_no == 0,
                                                     figure=fig,
                                                     grdspc=grid[:, col_no],
                                                     show=False,
                                                     save=False)
        sub_axes += [ax_sub]

        if main_xlim is None or main_ylim is None:
            main_xlim = xlim
            main_ylim = ylim
        else:
            main_xlim = [min(main_xlim[0], xlim[0]), max(main_xlim[1], xlim[1])]
            main_ylim = [min(main_ylim[0], ylim[0]), max(main_ylim[1], ylim[1])]

        print("generated subfig {}-{} {}, out of {} done".format(0, col_no, col_no*2+1, 6))
        col_no += 1

    for sub_ax in sub_axes:
        sub_ax.set_xlim(main_xlim)
        sub_ax.set_ylim(main_ylim)

    ax.legend(handles=handles, labels=labels, fontsize=75, loc='lower center',
              bbox_to_anchor=(.5, -0.75), ncol=2, fancybox=True)

    if save:
        filename = '{}Figure4.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=60)
        plt.close('all')
    return fig


def generate_figure_5(all_bayesian_data, all_systematic_data, show=False, unbiased=False, save=True):
    plt.close('all')
    folder = all_bayesian_data[list(all_bayesian_data.keys())[0]]['results_folder']

    fig = plt.figure(1, figsize=(55, 5), constrained_layout=True)

    grid = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.3, hspace=0.5)  # set the spacing between axes.

    # y_coord_labels = ['A', 'B   ', 'C']
    # for i in range(3):
    #     ax_main = fig.add_subplot(grid[i, 1:], frameon=False)
    #     ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
    #     ax_main.set_ylabel(y_coord_labels[i], fontsize=52, fontweight='bold', labelpad=135)

    col_no = 1
    for task in all_bayesian_data.keys():
        bayesian_data = all_bayesian_data[task]
        systematic_data = all_systematic_data[task]

        generate_figure_10(exp_number=task,
                           data_bayesian=bayesian_data,
                           data_systematic=systematic_data,
                           legend=col_no == 1,
                           figure=fig,
                           grdspc=grid[0, col_no],
                           unbiased=unbiased,
                           show=False,
                           save=False)
        print("generated subfig {}-{} {}, out of {} done".format(0, col_no, col_no*2+1, 6))

        # generate_belief_change_figure(bayesian_data=bayesian_data,
        #                               systematic_data=systematic_data,
        #                               colorbar=col_no == len(all_bayesian_data.keys()),
        #                               figure=fig,
        #                               grdspc=grid[1:, col_no],
        #                               show=False,
        #                               save=False)
        # print("generated subfig {}-{} {}, out of {} done".format(1, col_no, col_no*2+2, 6))
        col_no += 1

    if save:
        filename = '{}Figure5.png'.format(folder)
        # Save just the portion _inside_ the second axis's boundaries
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        points = bbox.get_points()
        new_points = points*np.array([[1., 1.], [1.02, 1.05]])  # 75% of the width instead of whole fig
        fig.savefig(filename, bbox_inches=transforms.Bbox(new_points), dpi=200)
    return fig


def generate_figure_ranked_acc(data=None, exp_number=0, legend=False, figure=None, grdspc=None, show=False, save=True):
    ranked_actions = data['unbiased_ranked_actions']
    action_benefits = data["unbiased_action_benefits"]
    action_accuracies = data['action_accuracies']
    # pdfs = data['pdfs']
    folder = data['results_folder']

    action_benefits = sorted(action_benefits, reverse=True)

    if figure is not None:
        fig = figure
        # grid = gridspec.GridSpecFromSubplotSpec(100, 100, subplot_spec=grdspc)
        ax = fig.add_subplot(grdspc)
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.add_subplot(111)

    if exp_number is not None:
       ax.set_title("{}\n".format(exp_dictionary[exp_number]), loc="center", fontsize=100, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=42)
    ax.xaxis.set_tick_params(labelsize=42)


    ranked_accuracies = []
    for act in ranked_actions:
        ranked_accuracies += [int(accuracy[1]*100) for accuracy in action_accuracies if accuracy[0] == act]

    plt.plot(sorted(action_benefits, reverse=True), ranked_accuracies,
             linewidth=9,
             linestyle='-',
             color='k',
             marker='*',
             markersize=12,
             # alpha=.7
             )

    plt.plot(sorted(action_benefits, reverse=True)[0], ranked_accuracies[0],
             linestyle=None,
             marker='o',
             markersize=62,
             markerfacecolor='green',
             markeredgewidth=4,
             markeredgecolor='k',
             alpha=.9,
             label="lowest ranked action"
             )
    plt.plot(sorted(action_benefits, reverse=True)[-1], ranked_accuracies[-1],
             linestyle=None,
             marker='o',
             markersize=72,
             markerfacecolor='red',
             markeredgewidth=10,
             markeredgecolor='k',
             alpha=.9,
             label="highest ranked action"
             )

    ax.set_xlabel('$Bhattacharyya\ coefficient$', fontsize=75)
    ax.set_ylabel('$Classification\ Accuracy (\%)$', fontsize=75)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()


    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles]
    [handle.set_markersize(90) for handle in handles]
    [handle.set_alpha(1) for handle in handles]
        # ax.legend(handles=handles, labels=labels, fontsize=42, loc='upper right',
        #           bbox_to_anchor=(2.8, -.25), ncol=2, fancybox=True)

    if save:
        filename = '{}fig6.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return ax, handles, labels, xlim, ylim


def generate_motion_param_comparison_figure1(data, show=False, save=False):
    plt.close('all')
    actions = data['actions']
    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(13, 10))
    grid = gridspec.GridSpec(ncols=1, nrows=6, figure=fig, wspace=.5, hspace=.5)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Ranked Motions', fontsize=28, labelpad=20)
    plt.ylabel('$Action\ Parameters$', fontsize=28, labelpad=60)

    params = [(0, (1, 2), "$\omega^Z$"),
              (1, (1, 3), "$\omega^{Rx}$"),
              (2, (1, 4), "$\omega^{Ry}$"),
              (3, (2, 2), "$A^{Z}$"),
              (4, (2, 3), "$A^{Rx}$"),
              (5, (2, 4), "$A^{Ry}$")]

    for i, idxes, param in params:
        if i == 0:
            ax_subplot = fig.add_subplot(grid[i, :])
        else:
            ax_subplot = fig.add_subplot(grid[i, :], sharex=ax_subplot)

        # ax_1d.set_xlabel('$\\vec{p}_1$', fontsize=48)
        ax_subplot.set_ylabel(param, fontsize=22)

        param_value = []
        for ranked_action in ranked_actions:
            act = [action for action in actions if action[0]==ranked_action][0]
            param_value += [act[idxes[0]][idxes[1]]]
        ax_subplot.plot(param_value)
        ax_subplot.get_xaxis().set_ticks(range(0, 64, 3))
        if i == params[-1][0]:
            ax_subplot.xaxis.set_tick_params(labelsize=18)
        ax_subplot.yaxis.set_tick_params(labelsize=14)

    if save:
        filename = '{}-action_comparison.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return None


def generate_belief_change_figure(bayesian_data=None, systematic_data=None, colorbar=False, figure=None,
                                  grdspc=None, show=False, save=False):
    plt.close('all')
    folder = bayesian_data['results_folder']

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=grdspc, wspace=0.05, hspace=0.05)
    else:
        fig = plt.figure(0, figsize=(15, 10))
        grid = gridspec.GridSpec(nrows=50, ncols=2, figure=fig)

    bar_plot_grid = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=grdspc)
    total_ax = fig.add_subplot(bar_plot_grid[1:-1, :], frameon=False)
    total_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    bayesian_benefits = np.array(bayesian_data['iteration_benefits']).T
    systematic_data_benefits = np.array(systematic_data['iteration_benefits']).T

    max_x = max(bayesian_benefits.shape[1], systematic_data_benefits.shape[1])

    for j, iterations_benefits in enumerate([bayesian_benefits, systematic_data_benefits]):
        if colorbar:
            ax = fig.add_subplot(
                grid[j*5+1:j*5+5, :-1]
            )
        else:
            ax = fig.add_subplot(
                grid[j*5+1:j*5+5, 1:]
            )
        # iterations_benefits = data['iteration_benefits']
        ordered_iteration_benefits = iterations_benefits[iterations_benefits[:,-1].argsort()]

        if ordered_iteration_benefits.shape[1] < max_x:
            # padding for missing data
            ordered_iteration_benefits = np.append(
                arr=ordered_iteration_benefits,
                values=np.broadcast_to(ordered_iteration_benefits[:, -1].reshape(-1, 1),
                                       (ordered_iteration_benefits.shape[0],
                                        max_x-ordered_iteration_benefits.shape[1])), axis=1)

        # b_coeffs = np.array(ordered_iteration_benefits).T
        image_to_print = resize(ordered_iteration_benefits,
                                (ordered_iteration_benefits.shape[0]*2, ordered_iteration_benefits.shape[1]),
                                anti_aliasing=True)
        im = ax.imshow(image_to_print, cmap='gist_gray',
                       norm=colors.LogNorm(vmin=np.min(ordered_iteration_benefits),
                                           vmax=np.max(ordered_iteration_benefits)))

        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        ax.set_xlabel('$Iteration\ number$', fontsize=34)
        ax.set_ylabel('$Ranked$ \n$Action\ Itentifiers$', fontsize=34)

        # add converted iterations in hours

        ax2 = fig.add_axes(ax.get_position(), frameon=False)
        ax2.tick_params(labelbottom=False, labeltop=True, labelleft=False, labelright=False,
                        bottom=False, left=False, right=False)

        ax1Xs = ax.get_xticks()
        upperXTickMarks = ["{:.2f}".format(x*.0302+.20) for x in ax1Xs]
        for t, tick in enumerate(upperXTickMarks):
            upperXTickMarks[t] = "{}:{}h".format(tick.split('.')[0], str(int(float(tick.split('.')[1])*60/100)))
        ax2.set_xticks(ax1Xs)
        ax2.set_xticklabels(upperXTickMarks, minor=False)
        ax2.set_xbound(ax.get_xbound())
        ax2.text(0.5, 1.28, "$Palpation\ exploration\ time$",
                 horizontalalignment='center',
                 fontsize=28,
                 transform=ax2.transAxes)
        ax2.xaxis.set_tick_params(labelsize=24)

    if colorbar:
        divider = make_axes_locatable(total_ax)
        cax = divider.append_axes("right", size="8%", pad="15%")
        cbar = plt.colorbar(im, cax=cax, ticks=np.arange(0.0, 1.1, .1))
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=38)
        cbar.ax.set_yticklabels([str(x) if x in [0.2, 0.5, 1] else "" for x in np.arange(0.0, 1.1, .1)])
        cbar.ax.set_ylabel('$Discriminative$\n$Confusion$', rotation=270, fontsize=38, labelpad=90)

    if save:
        filename = '{}-iteration_benefits.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return True


def generate_rank_change_figure(data, show=False, save=False):
    plt.close('all')
    iteration_rankings = data['iteration_rankings']
    # iterations_benefits = data['iteration_benefits']
    pdfs = data['pdfs']
    folder = data['results_folder']

    # ordered_itaration_rankings = []
    # for ranking in iteration_rankings:
    #     ordered_ranking = []
    #     for act in sorted(pdfs.keys()):
    #         ordered_ranking += [np.where(np.array(ranking) == act)[0][0]]
    #     ordered_itaration_rankings += [ordered_ranking]

    ordered_itaration_rankings = []
    for ranking in iteration_rankings:
        ordered_ranking = []
        for act in iteration_rankings[-1]:
            ordered_ranking += [np.where(np.array(ranking) == act)[0][0]]
        ordered_itaration_rankings += [ordered_ranking]

    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Iteration\ number$', fontsize=22)
    plt.ylabel('$Action\ Itentifiers$', fontsize=22)

    b_coeffs = np.array(ordered_itaration_rankings).T

    im = ax.imshow(b_coeffs, cmap='gist_gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$Bhattacharyya\ coefficient$', rotation=270)

    if save:
        filename = '{}-iteration_rankings.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return True


def generate_raw_data_figure(data=None, action=0, rank=0, folder="", show=False, legend=False, save=True,
                            figure=None, grdspc=None, colorbar=False):
    act = action[0]
    original_belief_data = data["original_belief_data"][act]

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(1, len(list(original_belief_data.keys()))*10-1, subplot_spec=grdspc)
    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = gridspec.GridSpec(ncols=len(list(original_belief_data.keys())),
                                 nrows=1,
                                 figure=fig)
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    min_glob = None
    max_glob = None

    for i, obj in enumerate(sorted(original_belief_data.keys())):
        average_obj_sample = (np.average(original_belief_data[obj][:, :], axis=0).reshape(3, -1).T*255).astype(np.int64)
        # average_obj_sample = (original_belief_data[obj][1, :].reshape(3, -1).T*255).astype(np.int64)
        min_val = np.min(average_obj_sample)
        max_val = np.max(average_obj_sample)
        if min_glob is None or min_glob > min_val:
            min_glob = min_val
        if max_glob is None or max_glob < max_val:
            max_glob = max_val
    for i, obj in enumerate(sorted(original_belief_data.keys())):
        average_obj_sample = (np.average(original_belief_data[obj][:, :], axis=0).reshape(3, -1).T*255).astype(np.int64)
        colorbarplot_shifter = 0
        colorbar_increaser = 0
        if colorbar and i == len(list(original_belief_data.keys())) - 1:
            colorbarplot_shifter = 0
            colorbar_increaser = 2

        ax_subplot = fig.add_subplot(
            grid[:, i*10+colorbarplot_shifter:i*10+7+colorbarplot_shifter+colorbar_increaser]
        )
        ax_subplot.set_ylabel('$time$', fontsize=48)
        ax_subplot.set_xlabel('$taxel\ values$', fontsize=48)
        # plt.autoscale(True)
        ax_subplot.xaxis.set_tick_params(labelsize=22)
        img = average_obj_sample
        # img = (img.T - np.average(img, axis=1)).T
        ax_subplot.set_title(label_tag_task_dict[task][obj], fontsize=28)
        ax_subplot.set_ylabel('$time\ (s)$', fontsize=28)
        ax_subplot.set_xlabel('$taxels$', fontsize=28)
        ax_subplot.set_yticks(list(np.arange(0.5, 4., 2.)))
        ax_subplot.set_yticklabels([str(sec) for sec in list(np.arange(0.5, 4., 2.))])
        ax_subplot.set_xticks(list(range(7)))
        ax_subplot.set_xticklabels(['t' + str(tax) for tax in range(7)])
        # fig.add_subplot(ax_subplot)
        im = ax_subplot.imshow(np.rot90(img, k=1, axes=(0, 1)), cmap='hot', interpolation='bilinear', vmin=min_glob, vmax=max_glob)

        if colorbar and i == len(list(original_belief_data.keys()))-1:
            divider = make_axes_locatable(ax_subplot)
            cax = divider.append_axes("right", size="20%", pad=0.5)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.tick_params(labelsize=22)
            # cbar.ax.set_xlabel('$normalized\ sensor\ values$', rotation=270, fontsize=20, labelpad=20)

    if save:
        filename = '{}{}_{}-raw_plots.png'.format(folder, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return True


def generate_motion_param_comparison_figure2(data, show=False, save=False):
    plt.close('all')
    action_accuracies = data['action_accuracies']
    actions = data['actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(13, 10))
    grid = gridspec.GridSpec(ncols=6, nrows=1, figure=fig, wspace=.5, hspace=.5)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('$Action\ Parameters$', fontsize=28, labelpad=40)
    plt.ylabel('$Accuracy$', fontsize=28, labelpad=20)

    params = [(0, (1, 2), "$\omega^Z$"),
              (1, (1, 3), "$\omega^{Rx}$"),
              (2, (1, 4), "$\omega^{Ry}$"),
              (3, (2, 2), "$A^{Z}$"),
              (4, (2, 3), "$A^{Rx}$"),
              (5, (2, 4), "$A^{Ry}$")]
    augmented_params = []

    params_values = np.zeros((len(params), len(actions)))
    for i, idxesm, param in params:
        for j, action in enumerate(actions):
            params_values[i, j] = action[idxesm[0]][idxesm[1]]
        vals = (np.unique(params_values[i, :]),)
        augmented_params += [(params[i] + vals)]

    for i, idxes, param, vals in augmented_params:
        if i == 0:
            ax_subplot = fig.add_subplot(grid[:, i])
            ax_subplot.yaxis.set_tick_params(labelsize=18)
        else:
            ax_subplot = fig.add_subplot(grid[:, i], sharey=ax_subplot)
        ax_subplot.set_xlabel(param, fontsize=22)

        box_data = []
        max_acc = 0
        for j, val in enumerate(vals):
            accuracies = []
            for act, accuracy in action_accuracies:
                action = [action for action in actions if action[0] == act][0]
                if action[idxes[0]][idxes[1]] == val:
                    accuracies += [accuracy]
                    if accuracy > max_acc:
                        max_acc = accuracy
            box_data += [accuracies]
        ax_subplot.boxplot(box_data, widths=0.5)
        ax_subplot.set_ylim([0, max_acc+.05])
        # DEBUG
        # for j, box_data in enumerate(box_data):/
        #     ax_subplot.text(j, np.median(box_data[j]), "{:.2f}".format(np.median(box_data[j])))

        if i == 3:
            ax_subplot.set_xticklabels(["{:.1f}".format(x*1000) for x in vals])
        else:
            ax_subplot.set_xticklabels(["{:.1f}".format(x) for x in vals])

        ax_subplot.xaxis.set_tick_params(labelsize=18)
        ax_subplot.get_yaxis().set_ticks(np.arange(0., 1., .1))
        ax_subplot.set_yticklabels(["{:.1f}".format(x) for x in np.arange(0., 1., .1)])

    if save:
        filename = '{}-action_comparison2.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return None


def generate_pca_figure(data_bayesian=None, data_systematic=None, show=False, save=False):
    actions = data_bayesian['actions']
    ranked_actions = data_bayesian['unbiased_ranked_actions']
    folder = data_bayesian['results_folder']
    original_belief_data = data_systematic['original_belief_data']

    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    for i in ranked_idxs_to_extract:
        generate_pca_histogram(actions=actions,
                               original_belief_data=original_belief_data,
                               rank=i,
                               folder=folder,
                               show=show,
                               save=save)
        plt.close('all')
    return True


def generate_figure_10(exp_number=None, data_bayesian=None, data_systematic=None, figure=None, legend=False,
                       unbiased=False, grdspc=None,  show=False, save=True):
    plt.close('all')
    if unbiased:
        best_bayesian_accuracies = data_bayesian["running_unbiased_accuracy"]
        best_systematic_accuracies = data_systematic["running_unbiased_accuracy"]
    else:
        best_bayesian_accuracies = data_bayesian["best_accuracies"]
        best_systematic_accuracies = data_systematic["best_accuracies"]

    max_baes = np.max(best_bayesian_accuracies)
    max_sys = np.max(best_systematic_accuracies)
    cut_off_idx = np.max([best_bayesian_accuracies.index(max_baes), best_systematic_accuracies.index(max_sys)]) + 20

    # cut_off_idx = min(len(best_bayesian_accuracies), len(best_systematic_accuracies)) + 1
    bayesian_accuracies = (np.array(best_bayesian_accuracies[:cut_off_idx])*100).astype(np.int)
    systematic_accuracies = (np.array(best_systematic_accuracies[:cut_off_idx])*100).astype(np.int)
    folder = data_bayesian['results_folder']

    if figure is not None:
        fig = figure
        # grid = gridspec.GridSpecFromSubplotSpec(100, 100, subplot_spec=grdspc)
        ax = fig.add_subplot(grdspc)
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.add_subplot(111)

    if exp_number is not None:
       ax.set_title("{}\n\n".format(exp_dictionary[exp_number]), loc="center", fontsize=42, fontweight='bold')
    ax.set_xlabel('$Experiment\ Update\ iterations$', fontsize=34)  # \ (1unit=1action&4objects)
    ax.set_ylabel('$Accuracy\ (\%)$', fontsize=34)

    ax.plot(systematic_accuracies,
             linewidth=6,
             linestyle='--',
             color='k',
             label="Systematic Search",
             marker="o",
             alpha=.7)
    ax.plot(bayesian_accuracies,
             linewidth=6,
             color='r',
             marker="o",
             label="Bayesian Exploration")

    # add converted iterations in hours
    ax2 = ax.twiny()
    ax1Xs = ax.get_xticks()
    upperXTickMarks = [str(datetime.timedelta(seconds=x*22*8)).split(':')[-3:-1] for x in ax1Xs]
    for i, tick in enumerate(upperXTickMarks):
        upperXTickMarks[i] = "{}:{}h".format(tick[0], tick[1])
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(upperXTickMarks)
    # title = ax.set_title("$Palpation\ exploration\ time$", fontsize=28)
    ax.text(0.4, 1.18, "$\ \ Experiment\ exploration\ time$",
             horizontalalignment='center',
             fontsize=34,
             transform=ax2.transAxes)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(handle) for handle in handles]
        [handle.set_linewidth(10) for handle in handles]
        [handle.set_alpha(1) for handle in handles]
        ax.legend(handles=handles, labels=labels, fontsize=42, loc='upper right',
                  bbox_to_anchor=(2.8, -.25), ncol=2, fancybox=True)

    if save:
        filename = '{}fig10.png'.format(folder)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


if __name__ == "__main__":

    # GENERATE ALL FIGURES

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

    environment = {
        'ssh': [(-.37652, -.04383)],
        'srh': [(-.38334, -.11278)],
        'rsh': [(-.44258, -.03139)],
        'rrh': [(-.44944, -.10279)],
        'sss': [(-.50545, -.02674)],
        'srs': [(-.51257, -.09933)],
        'rss': [(-.56858, -.01914)],
        'rrs': [(-.57429, -.09022)]
    }

    experiments = [
        # (0, "experiment 0"),
        (0, "Task 1: Object Geometry"),
        (1, "Task 2: Surface Roughness"),
        (2, "Task 3: Object Stiffness"),
    ]

    # TASK1 r-round, s-square
    # TASK2 s-smooth, r-rough
    # TASK3 s-soft, h-hard

    exp_bayesian_data = {}
    exp_systematic_data = {}

    DELAY = 0.
    SHOW = False
    SAVE = False

    # test_object_number = 1
    # test_idxs = np.random.choice(4, test_object_number, replace=False)
    test_object_number = 1
    test_idxs = np.random.randint(0, 4, size=test_object_number)
    test_idxs = [3]

    for task, exp_name in experiments:
        brain = BayesianBrain(
            environment=environment,
            actions=actions,
            samples_per_pdf=20,
            sensing_resolution=6,
            verbose=False,
            dimensionality_reduction=2,
            training_ratio=.7,
            experiment_number=1,
            task=task,
        )

        # simulate run of bayesian experiments from logged data
        exp_bayesian_data[task] = brain.run_experiment(bayesian=True,
                                                       initial_sample_number=1,
                                                       test_object_number=test_object_number,
                                                       delay=DELAY,
                                                       # maximum_iterations=80,
                                                       test_idxs=test_idxs,
                                                       task=task,
                                                       show=SHOW,
                                                       save=SAVE)
        # simulate run of systematic experiments from logged data
        exp_systematic_data[task] = brain.run_experiment(bayesian=False,
                                                         previous_accuracies=exp_bayesian_data[task]['best_accuracies'],
                                                         initial_sample_number=1,
                                                         test_object_number=test_object_number,
                                                         delay=DELAY,
                                                         # maximum_iterations=80,
                                                         test_idxs=test_idxs,
                                                         task=task,
                                                         show=SHOW,
                                                         save=SAVE)
        # create folder to dump results
        task_folder = '{}{}/'.format(exp_systematic_data[task]['results_folder'], "task"+str(task+1))
        exp_bayesian_data[task]['results_folder'] = task_folder
        exp_systematic_data[task]['results_folder'] = task_folder
        folder_create(exp_systematic_data[task]['results_folder'], exist_ok=True)


        for i, morph in enumerate(['0mm', '3mm', '5mm']):
            if brain.dimensionality_reduction == 2:
                generate_belief_state_act_morph2d(
                    data=exp_systematic_data[task],
                    morph=morph,
                    legend=i == 0,
                    task=task,
                    show=False,
                    save=True
                )
            else:
                generate_belief_state_act_morph1d(
                    data=exp_systematic_data[task],
                    morph=morph,
                    legend=i == 0,
                    task=task,
                    show=False,
                    save=True
                )
        generate_figure_6(
            data=exp_bayesian_data[task],
            show=False,
            save=True
        )


    # generate_figure_4(
    #     all_bayesian_data=exp_bayesian_data,
    #     all_systematic_data=exp_systematic_data,
    #     show=False,
    #     save=True
    # )
    generate_figure_5(
        all_bayesian_data=exp_bayesian_data,
        all_systematic_data=exp_systematic_data,
        show=False,
        save=True
    )
