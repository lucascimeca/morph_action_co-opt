import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_operators import SkinWriter
import ctypes
import pylab
import time
import matplotlib.gridspec as gridspec
import copy
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm
# from vision_helper import (
#     normal_rgb,
#     dense_flow,
#     sparse_flow
# )
from simple_io import *


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
# r-round, s-square
# s-smooth, r-rough
# s-soft, h-hard
task_letters = {
    1: ['r', 's'],
    2: ['r', 's'],
    3: ['h', 's']
}

task_labels = {
    1: {
        'r': 'round',
        's': 'edged'
    },
    2: {
        'r': 'rough',
        's': 'smooth'
    },
    3: {
        'h': 'hard',
        's': 'soft',
    }
}

class SincDataViewer(object):
    out = None  # variable holding writer object for saving view

    def __init__(self, window_name='cam_0', data_folder='./../data/', detector=None):
        self.window_name = window_name
        self.detector = detector
        self.data_folder = data_folder

        # screen size
        self.screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        # compute where in screen (different depends which Viewer)
        if self.detector.pos_arg == 'left':
            self.x_coord = 0
        elif self.detector.pos_arg == 'center':
            self.x_coord = np.int(self.screen_size[0]/3)
        elif self.detector.pos_arg == 'right':
            self.x_coord = np.int(self.screen_size[0]/3)*2
        else:
            raise ValueError("Please enter the correct screen position value (i.e. one of: 'left', 'center' or 'right')")

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        if self.out:
            self.out.vacuum()
        cv2.destroyAllWindows()

    def show_frame(self, step):
        raise NotImplementedError


class CameraViewer(SincDataViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frame_width = self.detector.camera.get(3)
        self.frame_height = self.detector.camera.get(4)

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        if self.detector.save and not self.detector.from_saved:
            self.out = cv2.VideoWriter(
                self.data_folder+'out_cam_{}.avi'.format(self.detector.camera_num),
                cv2.VideoWriter_fourcc(*'XVID'),
                25,  # todo: parameter must be based on time interval
                (np.int32(self.frame_width), np.int32(self.frame_height)),
                True
            )

    def show_frame(self, step, roi=None):
        # save only if connected to live cameras
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay, :, :, :])

        # print the image in the data at step, unless an image was passed to this function
        if roi is None:
            roi = self.detector.data[step - self.detector.step_delay, :, :, :]
        bw_img = self.detector.bw_data[step - self.detector.step_delay, :, :]

        # show data frames
        cv2.imshow(self.window_name, roi)
        cv2.moveWindow(self.window_name, self.x_coord, 0)

        # show detection
        cv2.imshow(self.window_name + 'detection', bw_img)
        cv2.moveWindow(self.window_name + 'detection', self.x_coord, np.int(self.frame_height + 40))

        cv2.waitKey(1)
        return True

class SkinViewer(SincDataViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_size = tuple([scr/2 for scr in self.screen_size])

        #compute dimension of windows to fit skin readings
        self.ihb_num = self.detector.skin.shape[0]
        self.module_num = self.detector.skin.shape[1]
        division_to_screen = np.ceil(np.sqrt(self.ihb_num * self.module_num))
        self.frame_height = self.screen_size[1]/division_to_screen
        self.frame_width = self.screen_size[0]/division_to_screen

        self.x_coord = 0
        self.y_coord = 0

        # define writer object to save skin data
        if self.detector.save and not self.detector.from_saved:
            self.out = SkinWriter(
                shape=(0,) + self.detector.data.shape[1:],
                name="skin_out",
                format="h5",
                folder="./../data/"
            )

    def show_frame(self, step):
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay-1, :])

        horizontal_count = 0
        for ihb in range(self.ihb_num):
            for module in range(self.module_num):
                heatmap, _ = skin_heatmap(self.detector.data[step - self.detector.step_delay-1, ihb, module, :], max=16000)

                width_to_display = int(np.floor((self.frame_height / heatmap.shape[1]) * heatmap.shape[0]))
                height_to_display = int(self.frame_height)
                reshaped_heatmap = cv2.resize(heatmap, (width_to_display, height_to_display), interpolation=cv2.INTER_CUBIC)

                # show data frames
                cv2.imshow("{}-{}-{}".format(self.window_name, ihb, module), reshaped_heatmap)
                cv2.moveWindow("{}-{}-{}".format(self.window_name, ihb, module), int(self.x_coord), int(self.y_coord))

                if not self.x_coord + self.frame_width > self.screen_size[1]:
                    self.x_coord += self.frame_width
                else:
                    self.x_coord = 0
                    self.y_coord += self.frame_height


        cv2.waitKey(1)

        self.x_coord = 0
        self.y_coord = 0

        return True

    def save_frame(self, step):
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay-1, :])



# Viewer for a Doctorobject
class DoctorViewer(SincDataViewer):
    out = None  # variable holding writer object for saving view

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # SKIN PARAMS
        self.skin_screen_size = tuple([scr/3 for scr in self.screen_size])

        #compute dimension of windows to fit skin readings
        self.ihb_num = self.detector.skin.shape[0]
        self.module_num = self.detector.skin.shape[1]
        division_to_screen = np.ceil(np.sqrt(self.ihb_num * self.module_num))
        self.skin_frame_height = self.skin_screen_size[1]/division_to_screen
        self.skin_frame_width = self.skin_screen_size[0]/division_to_screen

        self.skin_x_coord = np.int((self.screen_size[0] / 3)*1.2)  # later just put it equal to 0
        self.skin_y_coord = 100

        # ROBOT PARAMS
        self.robot_frame_width = int(self.screen_size[0] / 3)
        self.robot_frame_height = int(self.screen_size[1] / 3)

        self.buffer_length = 500
        self.action_profile_idx = 0
        self.action_profile_buffer = np.zeros((self.buffer_length, 6))
        self.state_profile_buffer = np.zeros((self.buffer_length, 6))

        self.robot_y_coord = 0
        self.robot_x_coord = np.int((self.screen_size[0] / 4))

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        cv2.destroyAllWindows()

    def show_frame(self, step, frame_rate=1):
        if frame_rate == 0:
            frame_rate = 1

        # -------------------------------
        # ---- SHOW SKIN -------
        for ihb in range(self.ihb_num):
            for module in range(self.module_num):
                heatmap, skin_array = skin_heatmap(self.detector.skin_snapshot, max=2000)

                width_to_display = int(np.floor((self.skin_frame_height / heatmap.shape[1]) * heatmap.shape[0]))
                height_to_display = int(self.skin_frame_height)

                reshaped_heatmap = cv2.resize(heatmap, (width_to_display, height_to_display), interpolation=cv2.INTER_CUBIC)

                # show data frames
                cv2.imshow("{}-{}-{}".format(self.window_name, ihb, module), reshaped_heatmap)
                cv2.moveWindow("{}-{}-{}".format(self.window_name, ihb, module), int(self.skin_x_coord), int(self.skin_y_coord))

                skin_3droi = image_to_3dplot(skin_array, heatmap, width_to_display, height_to_display)
                cv2.imshow("skin_3d", skin_3droi)
                cv2.moveWindow("skin_3d".format(self.window_name, ihb, module), int(self.skin_x_coord), int(self.screen_size[1]/2))


                if not self.skin_x_coord + self.skin_frame_width > self.skin_screen_size[1]:
                    self.skin_x_coord += self.skin_frame_width
                else:
                    self.skin_x_coord = np.int((self.screen_size[0] / 3)*1.2)  # later just put it equal to 0
                    self.skin_y_coord += self.skin_frame_height

        self.skin_x_coord = 0
        self.skin_y_coord = 0

        # -------------------------------
        # ---- SHOW ACTION PROFILE -------
        learing_figure = plt.figure(0, figsize=(8, 6))

        # update action buffer
        if self.detector.velocities is None:
            if np.any(self.action_profile_buffer != 0) or np.any(self.state_profile_buffer != 0):
                self.action_profile_buffer = np.zeros((self.buffer_length, 6))
                self.state_profile_buffer = np.zeros((self.buffer_length, 6))
                self.action_profile_idx = 0
        else:
            self.action_profile_buffer[self.action_profile_idx, :] = self.detector.velocities
            self.state_profile_buffer[self.action_profile_idx, :] = np.array(self.detector.state['actual_TCP_pose']) - \
                                                                    np.array(self.detector.palpation_start_pose)
            self.action_profile_idx += 1

        # todo: add as many plots as there are non zero parameters + label them
        axes_z = learing_figure.add_subplot(311)
        axes_z.title.set_text('Z profile')
        axes_z.plot(np.arange(40)*(1/frame_rate), self.action_profile_buffer[:40, 2], 'r--', label="control velocity")
        # axes_z.plot(np.arange(40)*(1/frame_rate), self.state_profile_buffer[:40, 2], 'k-', label="robot position")
        axes_z.set_ylabel('displacement\n(mm & mm/s)')
        axes_z.set_autoscale_on(True)
        axes_z.relim()
        axes_z.autoscale_view(True, True, True)
        axes_z.legend()

        axes_rx = learing_figure.add_subplot(312)
        axes_rx.title.set_text('X Profile')
        axes_rx.plot(np.arange(40)*(1/frame_rate), self.action_profile_buffer[:40, 0], 'r--', label="control velocity")
        # axes_rx.plot(np.arange(self.buffer_length)*(1/frame_rate), self.state_profile_buffer[:, 3], 'k-', label="robot position")
        axes_rx.set_ylabel('displacement\n(rad & rad/s)')
        axes_rx.relim()
        axes_rx.autoscale_view(True, True, True)
        axes_rx.legend()

        axes_ry = learing_figure.add_subplot(313)
        axes_ry.title.set_text('Y Profile')
        axes_ry.plot(np.arange(40)*(1/frame_rate), self.action_profile_buffer[:40, 1], 'r--', label="control velocity")
        # axes_ry.plot(np.arange(self.buffer_length)*(1/frame_rate), self.state_profile_buffer[:, 4], 'k-', label="robot position")
        axes_ry.set_xlabel('palpation time (s)')
        axes_ry.set_ylabel('displacement\n(rad & rad/s)')
        axes_ry.relim()
        axes_ry.autoscale_view(True, True, True)
        axes_ry.legend()
        learing_figure.tight_layout()

        learing_figure.canvas.draw()

        learning_curve_img = np.fromstring(learing_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        learning_curve_img = learning_curve_img.reshape(learing_figure.canvas.get_width_height()[::-1] + (3,))

        # print the image in the data at step, unless an image was passed to this function
        height_to_display = int(np.floor((learning_curve_img.shape[0] * self.robot_frame_height) / self.robot_frame_width))
        reshaped_learningroi = cv2.resize(learning_curve_img, (self.robot_frame_width, height_to_display))
        plt.close('all')

        cv2.imshow(self.window_name + 'action_profile', reshaped_learningroi)
        cv2.moveWindow(self.window_name + 'action_profile', self.robot_x_coord, self.robot_y_coord)

        cv2.waitKey(1)
        return True


"""Function printing the skin data from a SkinData time_snapshot onto a blank canvas.
        Inputs: canvas = (MxN) numpy array
                skin_snapshot = (AxB) numpy array where A<M and B<N
        Output: (MxN) matrix of snapshot in canvas"""
def fill_canvas(canvas, skin_snapshot):
    skin_snapshot = skin_snapshot.flatten()
    mid = np.floor(canvas.shape[1]/2).astype(np.int16)
    # ---------------- LAYOUT FOR EMBEDDED SKIN SENSOR IN EXAGON ------------
    canvas[0:2, mid-2:mid+2] =                       [skin_snapshot[4]]*2 + [skin_snapshot[3]]*2
    canvas[2:4, mid-3:mid+3] =          [skin_snapshot[5]]*2 + [np.average(skin_snapshot[1:])]*2 + [skin_snapshot[2]]*2
    canvas[4:6, mid-2:mid+2] =                       [skin_snapshot[6]]*2 + [skin_snapshot[1]]*2
    return canvas


def skin_heatmap(skin_data, max=None, interpolation=None):
    min = np.min(skin_data)
    if max is None:
        max = np.max(skin_data)
    if max - min != 0:
        norm_data = (skin_data - min) / (max - min)
    else:
        print("min-max of skin is same.. check!")
        norm_data = skin_data
    # skin_canvas = np.multiply(np.ones((6, 6)), np.min(norm_data))
    skin_canvas = np.zeros((6, 6))
    skin_array = fill_canvas(skin_canvas, norm_data)

    colormap = pylab.get_cmap('hot')
    mapped_img = colormap(skin_array)

    # img = plt.imshow(skin_array, cmap='hot', interpolation='gaussian')
    return cv2.cvtColor((255 * mapped_img).astype(np.uint8), cv2.COLOR_RGBA2BGR), skin_array


"""Function printing the time snapshot of the skin sensor onto a canvas.
        Inputs: time_snapshot = (MxN) numpy array 
        Output: (10x10) plot of the skin time snapshot onto a canvas """
def print_skin_data(time_snapshot):
    skin_canvas = np.ones((10, 10)) * np.min(time_snapshot, axis=(0, 1))
    skin_array = fill_canvas(skin_canvas, time_snapshot)
    return plt.imshow(skin_array, cmap='gray', interpolation='bicubic')




def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_bayesian_contours(ax, pdfs, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = bayesian_predict(pdfs, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def plot_normals(pdfs, reduced_data, action_benefit, folder="", show=True, save=False):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000']
    actions = [x for _, x in sorted(zip(action_benefit, sorted(pdfs.keys())), reverse=True)]
    worst_action = actions[-1]
    best_action = actions[0]
    print("best action is {}".format(best_action))

    figs = []  # figures, [best, worst]

    plots = dict()
    for act in [best_action, worst_action]:

        if act == best_action:
            name = "best action"
        else:
            name = "worst action"

        if len(reduced_data[act][list(pdfs[act].keys())[0]][0]) == 1:
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111)
            plt.title('Distributions for action: {}-{}'.format(act, name))
            ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
            ax.set_ylabel('$p(x)', fontsize=48)
            plt.autoscale(True)

            global_min = None
            global_max = None

            for obj in pdfs[act].keys():
                mu, sig = pdfs[act][obj]
                xmin = mu - 4*sig
                xmax = mu + 4*sig

                if global_min is None or global_min > xmin:
                    global_min = xmin
                if global_max is None or global_max < xmax:
                    global_max = xmax

            if global_min == global_max:
                xs = [0]*100
            else:
                xs = np.arange(global_min, global_max, (global_max-global_min)/100)

            for i, obj in enumerate(pdfs[act].keys()):
                mu, sig = pdfs[act][obj]
                ax.scatter(
                    reduced_data[act][obj][:, 0], [0]*len(reduced_data[act][obj][:, 0]),
                    edgecolor='k',
                    color=colors[i],
                    label=obj,
                    s=640,
                    alpha=0.7
                )
                plt.plot(xs, gaussian(xs, mu, sig), label=obj, c=colors[i])
            ax.legend()

        else:

            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111)
            plt.title('Distributions for action: {}-{}, pump at {}'.format(act, name))
            ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
            ax.set_ylabel('$\\vec{p}_2$', fontsize=48)
            plt.autoscale(True)

            for i, obj in enumerate(pdfs[act].keys()):
                vals, vecs = np.linalg.eigh(pdfs[act][obj][1])  # Compute eigenvalues and associated eigenvectors
                x, y = vecs[:, 0]
                theta = np.degrees(np.arctan2(y, x))
                hws = []
                hws += [tuple(2 * np.sqrt(vals))]
                hws += [tuple(4 * np.sqrt(vals))]
                hws += [tuple(6 * np.sqrt(vals))]

                ax.scatter(
                    reduced_data[act][obj][:, 0], reduced_data[act][obj][:, 1],
                    edgecolor='k',
                    color=colors[i],
                    s=640,
                    alpha=0.7
                )

                for w, h in hws:
                    circle = mpatches.Ellipse(xy=pdfs[act][obj][0],
                                              width=w,
                                              height=h,
                                              angle=theta,
                                              linestyle='-',
                                              fill=False,
                                              color=colors[i],
                                              linewidth=2.5)
                    ax.add_artist(circle)
                ax.text(pdfs[act][obj][0][0], pdfs[act][obj][0][1],
                        obj,
                        style='italic',
                        fontsize=34)


            resize_ratio = 2
            axes = np.array(plt.axis())
            x_ctr = np.average(axes[:2])
            y_ctr = np.average(axes[2:])
            half_w = (axes[1] - axes[0])/2
            half_h = (axes[3] - axes[2])/2
            plt.xlim([x_ctr-half_w*resize_ratio, x_ctr+half_w*resize_ratio])
            plt.ylim([y_ctr-half_h*resize_ratio, y_ctr+half_h*resize_ratio])

        figs += [fig]

        if show is True:
            plt.show()
        if save is True:
            folder_create(folder, exist_ok=True)
            fig.savefig(
                folder + name + '.png',
                bbox_inches="tight"
            )
    return figs[0], figs[1]

def plot_fig_10(accuracies, data_levels=None, folder=None, windows_name="", show=False, save=True):
    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    # expect rows to be different datasets, so transpose
    if data_levels is not None:
        for i in range(len(data_levels)):
            plt.plot(list(range(1, len(accuracies[i])+1)), accuracies[i], label=data_levels[i])
    else:
        plt.plot(list(range(1, len(accuracies)+1)), accuracies)

    ax.set_xlabel('$Baeysian\ Exploration\ Rank (low\ to\ high)$')
    ax.set_ylabel('$Classification\ Accuracy$')
    ax.legend()

    fig.canvas.draw()
    inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (900, 700)), cv2.COLOR_RGB2BGR)

    if save is True:
        fig.savefig(
            folder + '' 'fig10_' + windows_name + '.png',
            bbox_inches="tight"
        )

    if show:
        plt.show()
    plt.close('all')

    cv2.destroyAllWindows()
    cv2.imshow(windows_name, reshaped_roi)
    cv2.moveWindow(windows_name, 30, 30)
    cv2.waitKey(1)

    return fig


def show_progress(actions=None, benefits=None, accuracies=None, folder="",show=True, save=False,
                  previous_accuracies=None, delay=False, bayesian=False, windows_name="progress_bar"):
    plt.close('all')
    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    # plot progress
    if bayesian: label="bayesian_exploration"; label_vs="systematic_action_search";
    else: label_vs="bayesian_exploration"; label="systematic_search";
    plt.plot(list(range(1, len(accuracies)+1)), accuracies, label=label)
    if previous_accuracies is not None:
        plt.plot(list(range(1, len(previous_accuracies) + 1)), previous_accuracies, label=label_vs)

    for i in [0, len(accuracies)-1]:  # range(len(accuracies)):
        # plt.text(i+1, accuracies[i], "{}-{:.2f}".format(actions[i], benefits[i]))
        plt.text(i+1, accuracies[i], "{}".format(actions[i]))

    ax.set_xlabel('$palpation\ iterations\ (1unit=1action&4objects)$')
    ax.set_ylabel('$Classification\ Accuracy$')
    ax.legend()

    fig.canvas.draw()
    inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (600, 450)), cv2.COLOR_RGB2BGR)

    if save is True:
        fig.savefig(
            folder + '' 'fig10_' + windows_name + '.png',
            bbox_inches="tight"
        )

    plt.close('all')

    if show:
        # cv2.destroyAllWindows()
        cv2.imshow(windows_name, reshaped_roi)
        cv2.moveWindow(windows_name, 30, 30)
        cv2.waitKey(1)
    if delay:
        time.sleep(delay)

    return fig


def show_bayesian(pdfs=None, act=None, reduced_train_data=None, reduced_test_data=None, folder="", show=True, save=False,
                  delay=False, windows_name="progress_svm"):
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000']

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    plt.title('Distributions for action: {}'.format(act))
    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$p(x)', fontsize=48)
    plt.autoscale(True)

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
        ax.scatter(
            reduced_train_data[act][obj][:, 0], [0] * len(reduced_train_data[act][obj][:, 0]),
            edgecolor='k',
            color=colors[i],
            label='train_{}'.format(obj),
            s=640,
            alpha=0.7
        )

        ax.scatter(
            reduced_test_data[act][obj][:, 0], [0] * len(reduced_test_data[act][obj][:, 0]),
            edgecolor='k',
            color=colors[i],
            label='test_{}'.format(obj),
            s=640,
            marker='*',
            alpha=0.6
        )
        plt.plot(xs, gaussian(xs, mu, sig), label=obj, c=colors[i])

    ax.legend()
    fig.canvas.draw()
    inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (600, 450)), cv2.COLOR_RGB2BGR)

    if save is True:
        fig.savefig(
            folder + windows_name + '.png',
            bbox_inches="tight"
        )

    plt.close('all')

    if show:
        # cv2.destroyAllWindows()
        cv2.imshow(windows_name, reshaped_roi)
        cv2.moveWindow(windows_name, 600, 30)
        cv2.waitKey(1)
    if delay:
        time.sleep(delay)

    return fig


def show_bayesian2d(pdfs=None, act=None, reduced_train_data=None, reduced_test_data=None, folder="", task=0,
                    show=True, save=False, delay=False, windows_name="progress_svm"):
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000', '#4d0000', '#d2d916']
    objects = sorted(list(pdfs[act].keys()))
    all_object_keys = sorted(list(label_tag_task_dict[task].keys()))
    unlabelled_object_train_keys = list(label_tag_task_dict[task].keys())
    unlabelled_object_test_keys = list(label_tag_task_dict[task].keys())

    if windows_name is not None:
        windows_name = windows_name + "_online-inference"

    screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

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
            y_test += [objects.index(obj)] * \
                       reduced_test_data[act][obj].shape[0]


    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    X0_train, X1_train = X_train[:, 0], X_train[:, 1]
    X0_test, X1_test = X_test[:, 0], X_test[:, 1]

    # print object evidence and labels (if not labelled yut)
    for i in range(len(X0_train)):
        if all_object_keys[y_train[i]] in unlabelled_object_train_keys:
            ax.scatter(X0_train[i], X1_train[i], c=colors[y_train[i]], s=20, edgecolors='k',
                       label=label_tag_task_dict[task][all_object_keys[y_train[i]]])
            unlabelled_object_train_keys.remove(all_object_keys[y_train[i]])
        else:
            ax.scatter(X0_train[i], X1_train[i], c=colors[y_train[i]], s=20, edgecolors='k')
    for i in range(len(X0_test)):
        if all_object_keys[y_test[i]] in unlabelled_object_test_keys:
            ax.scatter(X0_test[i], X1_test[i], c=colors[y_test[i]], s=20, edgecolors='k', marker='+',
                       label=label_tag_task_dict[task][all_object_keys[y_test[i]]])
            unlabelled_object_test_keys.remove(all_object_keys[y_test[i]])
        else:
            ax.scatter(X0_test[i], X1_test[i], c=colors[y_test[i]], s=20, marker='+', edgecolors='k')

    if pdfs is not None:
        for i, obj in enumerate(objects):
            if len(pdfs[act][obj][1].shape) > 0:
                vals, vecs = np.linalg.eigh(pdfs[act][obj][1])  # Compute eigenvalues and associated eigenvectors
                x, y = vecs[:, 0]
                theta = np.degrees(np.arctan2(y, x))
                hws = []
                hws += [tuple(2 * np.sqrt(vals))]
                hws += [tuple(4 * np.sqrt(vals))]

                if obj == 'r':
                    linestyle = '--'
                else:
                    linestyle = '-'
                for w, h in hws:
                    circle = mpatches.Ellipse(xy=pdfs[act][obj][0],
                                              width=w,
                                              height=h,
                                              angle=theta,
                                              fill=False,
                                              color=colors[i],
                                              alpha=.35,
                                              linestyle=linestyle,
                                              linewidth=2,
                                              label=label_tag_task_dict[task][obj])
                    ax.add_artist(circle)

    resize_ratio = 2
    axes = np.array(plt.axis())
    x_ctr = np.average(axes[:2])
    y_ctr = np.average(axes[2:])
    half_w = (axes[1] - axes[0]) / 2
    half_h = (axes[3] - axes[2]) / 2
    plt.xlim([x_ctr - half_w * resize_ratio, x_ctr + half_w * resize_ratio])
    plt.ylim([y_ctr - half_h * resize_ratio, y_ctr + half_h * resize_ratio])

    ax.set_xlim(x_ctr - half_w * resize_ratio, x_ctr + half_w * resize_ratio)
    ax.set_ylim(y_ctr - half_h * resize_ratio, y_ctr + half_h * resize_ratio)
    ax.set_xlabel('$p1$')
    ax.set_ylabel('$p2$')
    # ax.set_xticks(())
    # ax.set_yticks(())

    plt.legend()
    fig.canvas.draw()
    inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img_height = int(np.floor((inference_img.shape[1] * inference_img.shape[0])) / (screen_size[0] / 10))

    reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (int(screen_size[0]/3), img_height)), cv2.COLOR_RGB2BGR)

    if save is True:
        fig.savefig(
            folder + windows_name + '.png',
            bbox_inches="tight"
        )

    plt.close('all')

    if show:
        # cv2.destroyAllWindows()
        cv2.imshow(windows_name, reshaped_roi)
        cv2.moveWindow(windows_name, int(screen_size[0]*2/3), 30)
        cv2.waitKey(1)
    if delay:
        time.sleep(delay)

    return fig


def plot_robot_inference2d(X_train=None, y_train=None, X_test=None, y_test=None, predictions=None, objects=None, clf=None, pdfs=None,
                           task=0, act=None, filter=None, windows_name="inference", folder="", show=False, save=False):
    # reduced_test_data = data['reduced_test_data']
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000', '#4d0000', '#d2d916']
    all_object_keys = objects
    unlabelled_object_keys = copy.deepcopy(all_object_keys)

    if windows_name is not None:
        windows_name = windows_name + "_online-inference"

    screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # print object evidence and labels (if not labelled yet)
    for i in range(len(X0)):
        if all_object_keys[y_train[i]] in unlabelled_object_keys:
            ax.scatter(X0[i], X1[i], c=colors[y_train[i]], s=20, edgecolors='k',
                       label=label_tag_task_dict[task][all_object_keys[y_train[i]]])
            unlabelled_object_keys.remove(all_object_keys[y_train[i]])
        else:
            ax.scatter(X0[i], X1[i], c=colors[y_train[i]], s=20, edgecolors='k')

    if pdfs is not None:
        for j, obj in enumerate(pdfs.keys()):
            if len(pdfs[obj][1].shape) > 0:
                vals, vecs = np.linalg.eigh(pdfs[obj][1])  # Compute eigenvalues and associated eigenvectors
                x, y = vecs[:, 0]
                theta = np.degrees(np.arctan2(y, x))
                hws = []
                hws += [tuple(2 * np.sqrt(vals))]
                hws += [tuple(4 * np.sqrt(vals))]
                for w, h in hws:
                    circle = mpatches.Ellipse(xy=pdfs[obj][0],
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

    resize_ratio = 2
    axes = np.array(plt.axis())
    x_ctr = np.average(axes[:2])
    y_ctr = np.average(axes[2:])
    half_w = (axes[1] - axes[0]) / 2
    half_h = (axes[3] - axes[2]) / 2
    plt.xlim([x_ctr - half_w * resize_ratio, x_ctr + half_w * resize_ratio])
    plt.ylim([y_ctr - half_h * resize_ratio, y_ctr + half_h * resize_ratio])

    if predictions is not None and X_test is not None and y_test is not None:
        # inference_cmap = ['g' if y_test[i] == predictions[i] else 'r' for i in range(len(predictions))]
        ax.scatter(X_test[:, 0], X_test[:, 1], c='y', s=85, edgecolor='k', marker='X')
        for i in range(len(predictions)):
            if y_test[i] == predictions[i]:
                ax.annotate('correct!', (X_test[i, 0], X_test[i, 1]),
                            color='green', fontsize=11)
            else:
                ax.annotate('incorrect!', (X_test[i, 0], X_test[i, 1]),
                            color='red', fontsize=11)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('$p1$')
    ax.set_ylabel('$p2$')
    # ax.set_xticks(())
    # ax.set_yticks(())

    plt.legend()
    fig.canvas.draw()
    inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img_height = int(np.floor((inference_img.shape[1] * inference_img.shape[0])) / (screen_size[0] / 3))

    reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (int(screen_size[0]/3), img_height)), cv2.COLOR_RGB2BGR)

    if save is True:
        folder_create(folder, exist_ok=True)

        if act is not None and filter is not None:
            filename = "{}_{}_{}.png".format(act, filter, windows_name)
        else:
            filename = "{}.png".format(windows_name)

        fig.savefig(
            folder + filename,
            bbox_inches="tight"
        )

    plt.close('all')

    cv2.destroyAllWindows()
    cv2.imshow(windows_name, reshaped_roi)
    cv2.moveWindow(windows_name, int(screen_size[0]*2/3), 30)
    cv2.waitKey(1)

    return True


def plot_robot_inference1d(pdfs=None, X_train=None, y_train=None, X_test=None, y_test=None, predictions=None,
                           windows_name="inference", folder="", show=False, save=False):

        screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        objects = sorted(pdfs.keys())


        # reduced_test_data = data['reduced_test_data']
        colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000', '#4d0000', '#d2d916']

        fig = plt.figure(figsize=(13, 13), constrained_layout=True)
        grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
        ax_gauss = fig.add_subplot(grid[:-1, :])
        ax_1d = fig.add_subplot(grid[-1, :], sharex=ax_gauss)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        # if action_number is not None:
        #     ax_gauss.set_title('Exploratory Palpation Number {}'.format(action_number), fontsize=48)
        # ax_1d.set_xlabel('$\\vec{p}_1$', fontsize=48)
        # ax_gauss.set_ylabel('$p(x)$', fontsize=48)
        plt.autoscale(True)

        ax_gauss.yaxis.set_tick_params(labelsize=30)
        ax_1d.xaxis.set_tick_params(labelsize=30)
        ax_1d.yaxis.set_ticks([])

        global_min = None
        global_max = None

        for obj in pdfs.keys():
            mu, sig = pdfs[obj]
            xmin = mu - 4 * sig
            xmax = mu + 4 * sig

            if global_min is None or global_min > xmin:
                global_min = xmin
            if global_max is None or global_max < xmax:
                global_max = xmax

        if global_min == global_max:
            xs = [0] * 500
        else:
            xs = np.arange(global_min, global_max, (global_max - global_min) / 500)

        for i in range(len(y_train)):
            ax_1d.plot(
                [X_train[i, 0]], [0],
                marker='o',
                linestyle='None',
                markersize=15,
                markerfacecolor=colors[y_train[i]],
                markeredgewidth=4,
                markeredgecolor=colors[y_train[i]],
                alpha=0.7
            )

        if predictions is not None and X_test is not None and y_test is not None:
            # inference_cmap = ['g' if y_test[i] == predictions[i] else 'r' for i in range(len(predictions))]
            for i in range(len(predictions)):
                ax_1d.plot(
                    [X_test[i, 0]], [0],
                    marker='+',
                    linestyle='None',
                    markersize=25,
                    markerfacecolor=colors[predictions[i]],
                    markeredgewidth=4,
                    markeredgecolor='r',
                    alpha=0.7
                )
                if y_test[i] == predictions[i]:
                    ax_1d.annotate("correct", (X_test[i, 0], 0),
                                   color='green', fontsize=11)
                else:
                    ax_1d.annotate("incorrect", (X_test[i, 0], 0),
                                   color='red', fontsize=11)

        for i, obj in enumerate(sorted(pdfs.keys())):
            mu, sig = pdfs[obj]
            if obj[1] == 'r':
                linestyle = '--'
            else:
                linestyle = '-'
            ax_gauss.plot(xs, gaussian(xs, mu, sig),
                          label=labeltag_dict[obj],
                          c=colors[i],
                          linestyle=linestyle,
                          linewidth=4,
                          alpha=.5)
            ax_gauss.legend(fontsize=22)


        # if legend:
        plt.legend()
        fig.canvas.draw()
        inference_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        inference_img = inference_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_height = int(np.floor((inference_img.shape[1] / (inference_img.shape[0] * 1.3)) * (screen_size[0] / 3)))
        reshaped_roi = cv2.cvtColor(cv2.resize(inference_img, (int(screen_size[0] / 3), img_height)), cv2.COLOR_RGB2BGR)
        plt.close('all')
        cv2.destroyAllWindows()
        cv2.imshow(windows_name, reshaped_roi)
        cv2.moveWindow(windows_name, int(screen_size[0] * 2 / 3), 30)
        cv2.waitKey(1)

        if save:
            folder_create(folder, exist_ok=True)
            fig.savefig(
                folder + windows_name + '.png',
                bbox_inches="tight"
            )

        if show:
            plt.show()

        return True
    

def plot_conf_matrix(targets=None, outputs=None, ordered_labels=None, folder="", filename="conf_matrix"):

    n = np.unique(outputs).shape[0]

    conf_fig = plt.figure(figsize=(10, 5))
    ax = conf_fig.add_subplot(1, 1, 1)
    cm = sm.confusion_matrix(targets, outputs)
    plt.imshow(cm, interpolation='none', cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center', fontsize=24)
    plt.xlabel("kmeans labels", fontsize=16)
    plt.ylabel("true labels", fontsize=16)
    plt.xticks(range(n), [ordered_labels[i] + "\nguess" for i in range(n)], size='medium')
    plt.yticks(range(n), [ordered_labels[i] for i in range(n)], size='small')

    conf_fig.savefig(
        folder + filename + '.png',
        bbox_inches="tight"
    )


def image_to_3dplot(image, heatmap, width, height):
    image = cv2.resize(image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, 1)
    ax.plot_surface(xx, yy, image, facecolors=heatmap/255)
    fig.canvas.draw()

    learning_curve_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    learning_curve_img = learning_curve_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # print the image in the data at step, unless an image was passed to this function
    reshaped_roi = cv2.resize(learning_curve_img, (width, height))
    return reshaped_roi


def bayesian_predict(pdfs, data):
    labels = []
    for i in range(data.shape[0]):
        label = 0
        max_prob = 0
        for j, obj in enumerate(sorted(pdfs.keys())):
            mu, sig = pdfs[obj]
            prob = gaussian(data[i, :], mu, sig)
            if prob > max_prob:
                max_prob = prob
                label = j
        labels += [label]
    return np.array(labels)
