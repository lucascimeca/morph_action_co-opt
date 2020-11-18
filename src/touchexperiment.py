import threading
import json
import signal
import sys
import math
from sklearn.decomposition import PCA
from sklearn import svm

from data_operators import *
from rtde.UR5_protocol import UR5
from visualization import *
from main_bayesian_exploration import BayesianBrain
from morph_filter import MorphingFilter

print ("Random number with seed 30")
np.random.seed(123)

FRAME_RATE = 0
TIME_LEFT = 10000000

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


"""function running the learning live
Attributes:
     time (double): time-step (sec) for image sampling and data collection"""
def run_learning(time_interval=1.,
                 steps=0,
                 learning=True,
                 resume_previous_experiment=False,
                 number_of_samples=3,
                 sensing_resolution=10,
                 palpation_duration=3,
                 dimensionality_reduction=2,
                 training_environment=None,
                 testing_environment=None,
                 actions=None,
                 verbose=True,
                 task=0,
                 downtime_between_experiments=8,
                 training_ratio=.6,
                 testing=False):

    if testing:
        # retrieve best actions for test experiments
        bayesian_brain = BayesianBrain(
            environment=testing_environment,
            actions=actions,
            samples_per_pdf=20,
            sensing_resolution=6,
            experiment_number=-1,
            verbose=False,
            dimensionality_reduction=2,
            training_ratio=.7
        )
        data = bayesian_brain.run_experiment(bayesian=True,
                                             initial_sample_number=1,   #3
                                             test_object_number=1,      #6
                                             show=False,
                                             save=True)
        unbiased_ranked_actions = data['unbiased_ranked_actions']

        # actions = [actions[i] for i in unbiased_ranked_actions[:20]]
        actions = [unbiased_ranked_actions[0]]
        environment = testing_environment
    else:
        environment = training_environment

    # ----create instances and threads----
    palpation = TouchExperiment(interval=time_interval)  # start palpation on main thread
    doctor = Agent(
        robot_ip="169.254.94.83",
        learning=learning,
        resume_previous_experiment=resume_previous_experiment,
        sample_number=number_of_samples,
        environment=environment,
        actions=actions,
        task=task,
        sensing_resolution=sensing_resolution,
        palpation_duration=palpation_duration,
        dimensionality_reduction=dimensionality_reduction,
        downtime_between_experiments=downtime_between_experiments,
        training_ratio=training_ratio,
        data_folder='./../data/',
        verbose=verbose
    )

    # ---- subscribe cameras to belt updates----
    palpation.subscribe(doctor)

    # start threads
    doctor.start()

    # ---- Handle for graceful shutdowns -----
    def signal_handler(sig, frame):
        palpation.end_run()
        sys.exit()
    signal.signal(signal.SIGINT, signal_handler)  # to handle gracefull shutdowns

    # ----run ---
    try:
        palpation.run(steps)

    except Exception as e:
        raise Exception(e)

    finally:
        palpation.end_run()


class TouchExperiment:
    """Conveyer belt, when it runs it synchronises events for the cameras to take snapshots, variant of MVC pattern

    Attributes:
        time_delay (double): time delay after which the detector returns a sample.
    """
    def __init__(self, interval):
        print("Initializing PALPATION...")
        self.interval = interval
        self._observers = set()
        self.notified = False
        self.t0 = time.clock()
        self.current_time = time.clock()
        print("PALPATION initialized.")

    def subscribe(self, observer):
        self._observers.add(observer)
        print("subscribed {} to finger.".format(observer.name))

    def unsubscribe(self, observer):
        self._observers.discard(observer)
        print("unsubscribed {} to finger.".format(observer.name))

    def run(self, steps):
        print("Running finger updates!")
        self.t0 = time.clock()
        self.current_time = time.clock()

        # notify observers of how many steps to expect
        [observer.init_data(steps) for observer in self._observers]

        notifications = [False, False]
        current_time = time.clock() - self.t0

        idx = 0
        if self.interval > 0:
            idx = np.floor(current_time / self.interval).astype(np.int32)  # get interval index from time
        else:
            idx += 1
        while True:
            prev_time = current_time
            current_time = time.clock() - self.t0

            prev_idx = idx
            if self.interval > 0:
                idx = np.floor(current_time/self.interval).astype(np.int32)  # get interval index from time
            else:
                idx += 1
            if prev_idx != idx:
                notifications[np.mod(prev_idx, 2)] = False

            if not notifications[np.mod(idx, 2)]:
                # notify observers of step
                nxt_state = np.array([observer.update(idx) for observer in self._observers])
                if any(nxt_state==False):
                    print("At least some observers do not have any more data!")
                    break
                notifications[np.mod(idx, 2)] = True
                global FRAME_RATE
                FRAME_RATE = (FRAME_RATE + 1/(current_time-prev_time))/2

                # sys.stdout.write("\rFrame rate: {}Hz".format(1/(current_time-prev_time)))
                # sys.stdout.flush()

            if steps != 0 and idx < steps - 1:
                break
        print("Reached {} steps, run complete!".format(steps))
        self.end_run()

    def end_run(self):
        observers = self._observers.copy()
        print("un-subscribing observers")
        [self.unsubscribe(observer) for observer in observers]
        print("releasing...")
        [observer.release() for observer in observers]
        print("end")


class BodyObserver(threading.Thread):
    """Observer simulating a general observer for the body

    Attributes:
        time_delay (double): time delay after which the detector returns a sample.
    """

    data = None

    def __init__(self, name, step_delay=0, data_folder='./../data/', pos_arg='left', from_saved=False, save=True):
        super().__init__()
        self.name = name
        self.step_delay = step_delay
        self.data_folder = data_folder
        self.from_saved = from_saved
        self.save = save
        self.pos_arg = pos_arg   # for viewer
        self.data_buffer = []
        self.nxt_buffer = []
        # connect a viewer to this camera

    # function called at start()
    def run(self):
        pass

    def init_data(self, steps):
        raise NotImplementedError

    # function that should be called at the end
    def release(self):
        self.join()

    def get_snapshot(self):
        raise NotImplementedError

    def update(self, step):
        raise NotImplementedError


class SkinObserver(BodyObserver):

    def __init__(self, name="skin", pos_arg='center', *args, **kwargs):
        print("Initializing {} Thread...".format(name.upper()))
        super().__init__(name=name, *args, **kwargs)
        # skin specific arguments
        self.skin = SkinData(
            live=self.from_saved is False,
            filename='{}{}_out.{}'.format(self.data_folder, self.name, 'h5')
        )
        self.step_delay = self.step_delay
        self.pos_arg = pos_arg
        self.step = 0
        print("{} Thread initialized.".format(name.upper()))

    # function called at start()
    def init_data(self, steps):
        self.data = np.zeros((steps,) + self.get_snapshot()[1].shape)
        # connect a viewer to this camera
        self.data_viewer = SkinViewer(window_name=self.name, data_folder=self.data_folder, detector=self)

    def get_snapshot(self):
        next, skin_state = self.skin.read()
        return next, skin_state

    def skin_contact(self):
        return self.skin.skin_contact()

    def update(self, step):
        self.step = step  # last logged step now

        # ----- store frame in data -----
        if self.step_delay != 0 and not self.from_saved:
            # add buffer if taking images live with cameras
            nxt_state, snapshot = self.get_snapshot()
            self.data_buffer.insert(0, snapshot)
            self.nxt_buffer.insert(0, nxt_state)
            if step >= self.step_delay:
                self.data[step - self.step_delay, :] = self.data_buffer.pop()
                nxt_state = self.nxt_buffer.pop()
        else:
            nxt_state, self.data[step - self.step_delay, :] = self.get_snapshot()

        # -- trigger viewer & saver
        self.data_viewer.show_frame(step)

        return nxt_state

    def calibrate_skin(self):
        self.skin._calibrate_skin()

    def release(self):
        print("{} releasing files and objects...".format(self.name.upper()))
        self.data_viewer.release()
        super(SkinObserver, self).release()


class Agent(BodyObserver):

    current_motion = None
    prev_motion = None

    def __init__(self, name="Robo-doctor", robot_ip="169.254.71.113", learning=False, environment=None, actions=None,
                 sample_number=3, sensing_resolution=10, palpation_duration=3, downtime_between_experiments=8, task=0,
                 verbose=False, resume_previous_experiment=False, dimensionality_reduction=2, training_ratio=.6,
                 *args, **kwargs):
        print("Initializing {} Thread...".format(name.upper()))
        super().__init__(name=name, *args, **kwargs)

        self.verbose = verbose
        self.resume_previous_experiment = resume_previous_experiment
        # ---------------------- DATA ------------------------
        base_name = "experiment"
        if learning:
            num = 0
            exp_folder = "{}{}_{}\\".format(self.data_folder, base_name, num)
            prev_folder = exp_folder
            while folder_exists(exp_folder):
                prev_folder = exp_folder
                exp_folder = "{}{}_{}\\".format(self.data_folder, base_name, num)
                num += 1
            if self.resume_previous_experiment:
                self.data_folder = prev_folder
            else:
                self.data_folder = exp_folder
                folder_create(exp_folder)
        else:
            # retrieve latest experiment
            experiment_folders = list(sorted([file for file in os.listdir(self.data_folder) if file.startswith(base_name)],
                                             key=lambda x: int(x.split('_')[1])))
            if len(experiment_folders) == 0:
                print("Please gather some data before doing Bayesian Inference.")
                raise(FileNotFoundError("Experiment folder in {}".format(self.data_folder)))
            self.data_folder = self.data_folder + experiment_folders.pop() + "/"
        self.results_folder = self.data_folder + "results/"

        # ---------------------- ROBOT BODY ----------------------
        # Initialize robot and go to initial palpation position
        # self.palpation_joint_start = np.deg2rad([85.12, -78.13, 121.96, 226.76, -87.81, -3.34]) # training
        # self.palpation_joint_start = np.deg2rad([83.73, -109.20, 121.96, 258.39, -89.19, -14.06]) # testing
        # self.palpation_joint_start = np.deg2rad([78.67, -100.30, 133.39, 240.26, -90.06, -17.36]) # demo
        self.touch_joint_start = np.deg2rad([-9.08, -70.20, 128.89, -147.63, -90.53, 1.39]) # demo2
        self.robot = UR5(robot_ip=robot_ip)
        self.robot.joint_go_to(self.touch_joint_start, acc=.5, vel=1)
        print('Waiting for robot to reach start position...')
        while not self.robot.reached_point(point=self.touch_joint_start, mode='joint'): pass
        print("Start position reached!")
        # self.filter = MorphingFilter(
        #     COM=4,
        #     baud_rate=9600,
        #     verbose=False
        # )

        self.displacement = 0.17
        self.palpation_duration = palpation_duration
        self.current_time = time.clock()
        self.robot_stop = False
        self.force_limit_procedure = False
        self.force_reset_initiated = False
        self.robot_connection_lost = False
        self.downtime_between_experiments = downtime_between_experiments   # seconds downtime, to make skin rest
        self.alpha_max = None
        self._reset_palpation()

        # ---------------------- ROBOT SKIN ----------------------
        self.skin = SkinData(
            live=True,
            filename='{}{}_out.{}'.format(self.data_folder, self.name, 'h5')
        )
        self.skin_shape = self.skin.get_shape()
        self.skin_snapshot = np.zeros((2,)+self.skin_shape)
        self.skin_time_lag = .3              # time window to ignore sensor switch
        self.skin_contact_time = -2
        self.skin_error_count = 0

        # -- initialize some variables
        self._update_doctor_state()
        self.palpation_cart_start = self.state['actual_TCP_pose']
        print("{} Thread initialized.".format(name.upper()))

        # ---------------------- BRAIN ------------------------
        self.brain = Brain(
            body=self,
            learning=learning,
            environment=environment,
            actions=actions,
            samples_per_pdf=sample_number,
            sensing_resolution=sensing_resolution,
            verbose=self.verbose,
            task=task,
            dimensionality_reduction=dimensionality_reduction,
            training_ratio=training_ratio
        )

    # function called at start()
    def init_data(self, steps):
        # connect a viewer to this doctor
        self.data_viewer = DoctorViewer(window_name=self.name, data_folder=self.data_folder, detector=self)

    def _reset_palpation(self):
        self.robot_start_palpate = False
        self.detected_contact_number = 0
        # current_motion is (object_label(str), object_position(tuple), motion_parameters, sample_number)
        self.prev_motion = copy.deepcopy(self.current_motion)
        self.current_motion = None
        self.reached_palpation_end = False
        self.reached_palpation_positon = None
        self.descending = True
        self.palpation = False
        self.downtime = None
        self.downtime_start = None
        self.touched = False
        self.end_palpate = False
        self.velocities = None

        self.writer = None

        self.descent_start_pose = None
        self.palpation_start_pose = None
        self.palpation_start_time = None
        self.palpation_end_time = None

        self.force_limit_procedure = False
        self.force_reset_initiated = False
        self.robot_connection_lost = False

    def get_snapshot(self):
        state = self.robot.get_state()
        return True, state

    def _update_doctor_state(self):
        # update internal state --- keeps two readings on buffer
        self.skin_snapshot[0, :] = self.skin_snapshot[-1, :]
        skin_state, self.skin_snapshot[-1, :] = self.skin.read()
        if np.all(self.skin_snapshot[-1, :] == self.skin_snapshot[0, :]):
            self.skin_error_count += 1
        else:
            self.skin_error_count = 0
        robot_state, self.state = self.get_snapshot()
        self.current_time = time.clock()
        self.moving = self.robot.is_moving()
        # print(self.state['actual_TCP_force'])
        if self.state is None or self.moving is None:
            self.robot_connection_lost = True
        elif np.any(np.array(self.state['actual_TCP_force']) > 130):
            self.force_limit_procedure = True
        return skin_state and robot_state

    def update(self, step):

        # update sensor states
        nxt_state = self._update_doctor_state()

        # todo: add remote printing
        if self.verbose:
            self.data_viewer.show_frame(step, frame_rate=FRAME_RATE)

        if not self.robot_stop:
            if not self.force_limit_procedure or self.skin_error_count <= 4 or self.robot_connection_lost:

                # -- Retrieve palpation info
                if self.reached_palpation_positon is None:
                    self.current_motion = self.brain.get_motion()

                    # -- Handle bad cases
                    if self.current_motion is None:
                        return False
                    elif self.alpha_max is not None and self.current_motion[2][2][2] >= self.alpha_max:
                        print("## Skipping the palpation of object {}, with action {}. The action has a Z alpha of {}, "
                              "higher or equal to a previously failed palpation experiment with an alpha of {}.".format(
                            self.current_motion[0],
                            self.current_motion[2],
                            self.current_motion[2][2][2],
                            self.alpha_max
                        ))
                        self.robot.joint_stop(acc=5.)
                        self.robot_stop = True
                        self.brain.discard()
                        self._reset_palpation()
                        return True

                    # if you're resuming experiment, then check if the data is already present
                    if self.brain.learning and self.resume_previous_experiment:
                        obj = self.current_motion[0]
                        action_idx = self.current_motion[2][0]
                        sample_no = self.current_motion[3]
                        exp_name = "{}_{}_{}".format(obj, action_idx, sample_no)
                        exp_files = list([file for file in os.listdir(self.data_folder) if file.startswith(exp_name)])
                        if len(exp_files) != 0:
                            print(
                                "The experiment for object '{}' action '{}' sample '{}' is already present, skipping..."
                                .format(obj, action_idx, sample_no))
                            self.robot.joint_stop(acc=5.)
                            self.robot_stop = True
                            self.brain.discard(keep_motion=False)  # don't keep the motion in the stack
                            self._reset_palpation()
                            return True

                    # first case: an environment was given, I may or may not be learning
                    if self.current_motion[1] is not None:
                        if len(self.current_motion[1]) == 2:
                            self.descent_start_pose = list(self.current_motion[1])+self.palpation_cart_start[2:]
                        else:
                            self.descent_start_pose = self.current_motion[1]
                            # second case: no environment was given, I'll palpate on the start location
                    else:
                        self.descent_start_pose = self.palpation_cart_start[:2]+self.palpation_cart_start[2:]
                    self.robot.cart_go_to(pose=self.descent_start_pose, acc=.3, vel=.3)
                    self.reached_palpation_positon = False
                    if self.verbose:
                        print("Moving to bead location {}, for object {}, with action {}".format(self.descent_start_pose,
                                                                                                 self.current_motion[0],
                                                                                                 self.current_motion[2]))

                # -- Reach palpation position
                elif self.reached_palpation_positon is False:
                    self.reached_palpation_positon = self.robot.reached_point(point=self.descent_start_pose, mode='pose')
                    if self.reached_palpation_positon is True:
                        self.robot.joint_stop(acc=1.)
                        self.robot_stop = True
                        self.robot_start_palpate = True

                # -- Wait for sensor to re-set
                elif self.reached_palpation_positon:
                    if self.downtime is None:
                        self.downtime_start = time.clock()
                        self.downtime = True

                        if self.prev_motion is not None \
                                and self.current_motion is not None \
                                and self.prev_motion[2][-1] != self.current_motion[2][-1]:
                            input(
                                "ROBOT PAUSED: please swap the current filter '{}' with filter '{}' and press ENTER..."
                                .format(self.prev_motion[2][-1], self.current_motion[2][-1]))

                    elif self.downtime is True and time.clock() - self.downtime_start > self.downtime_between_experiments:
                        self.downtime = False

                    # -- Palpate
                    if self.downtime is False:
                        # in the middle of palation here
                        self.brain._execute_motion(self.moving, step)
            else:
                if self.skin_error_count >=4:
                    print("Skin error encountered: the skin values are not changing, there might be an issue "
                          "with the skin, aborting experiments.")
                    return False
                elif self.moving and not self.force_reset_initiated and not self.robot_connection_lost:
                    # if self.verbose:
                    print("## Stopping current motion since it would have triggered the safety force stop!\n"
                          "## Failed because robot forces were observed to be: {}.".format(self.state['actual_TCP_force']))
                    print("## The problem was most likely caused by a setting of Max Alpha on the Z axis which brough the robot to palpate too deeply."
                          "## The observed alpha was of {}. Alphas equal or higher than the observed ones will be automatically skipped in future palpations".format(self.current_motion[2][2][2]))
                    self.alpha_max = self.current_motion[2][2][2]
                    self.robot.joint_stop(acc=5.)
                    self.robot_stop = True
                    self.force_reset_initiated = True
                else:
                    if self.descent_start_pose[2] - self.state['actual_TCP_pose'][2] <= 0.0001:
                        if self.verbose:
                            print("All done! on to another bead -- DOCTOR RESET")
                        self.robot.joint_stop(acc=5.)
                        self.robot_stop = True
                        self.brain.discard()
                        self._reset_palpation()
                    elif not self.moving:
                        self.robot.cart_go_to(pose=self.descent_start_pose, acc=.3, vel=.3)  # go back

        else:
            if self.verbose:
                print("Stopping the robot!")
            if not self.moving:
                self.robot_stop = False

        return nxt_state

    def release(self):
        # ----- save skin data relevant to experiments -----
        self.brain.discard()
        if self.verbose:
            print("{} releasing files and objects...".format(self.name.upper()))
            print("Stopping robot...")
        self.robot.joint_stop(acc=5.)
        while self.robot.is_moving():
            pass
        if self.verbose:
            print("Going home!")
        self.robot.joint_go_to(self.touch_joint_start, acc=.5, vel=1)
        while not self.robot.reached_point(point=self.touch_joint_start, mode='joint'):
            pass
        print("All done.")
        # ----- disconnect -----
        self.robot.disconnect()
        # self.data_viewer.release()
        pass


class Brain(object):

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

    def __init__(self, body=None, learning=False,  environment=None, actions=None, priors=None, sensing_resolution=10,
                 dimensionality_reduction=2, samples_per_pdf=3, task=0, training_ratio=.8, test_mode=False, verbose=False):
        self.sensing_resolution = sensing_resolution
        self.dimensionality_reduction = dimensionality_reduction
        self.samples_per_pdf = samples_per_pdf
        self.learning = learning
        self.verbose = verbose

        if body is None:
            raise ValueError("The brain need a body to be able to get tactile feedback!!")
        self.body = body

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

        self.skin_buffer = np.zeros((self.sensing_resolution,) + self.body.skin_shape)

        if environment is not None:
            if environment is None or actions is None:
                raise ValueError("To learn, the doctor must be supplied a known environment and actions!")
            self.objects = sorted(environment.keys())
            self.number_of_objects = len(self.objects)  # D
            self.actions = actions
            self.number_of_actions = len(actions)  # N
            self.environment = environment

        if not test_mode:
            if learning is True:
                self.motion_stack = self._get_motion_stack()
                if self.body.resume_previous_experiment:
                    # load experiment information
                    self._load_experiment_details()
                else:
                    # dump experiment information
                    self._dump_experiment_details()
                self.belief_data = {}
            else:
                self._load_experiment_details()

                # training_ratios = np.arange(.1, .9, .1)
                training_ratios = [.8]
                all_ordered_actions = []
                all_ordered_action_accuracies = []
                self.training_ratio = training_ratio

                # load initial data from Hard Disk
                self.belief_data = self._load_data()
                # make data low dimensional and fit pdfs on test data
                self.reduced_belief_data = self._fit_pdfs(self.belief_data)

                if environment is not None:
                    self.location_stack, self.location_labels = self._create_location_stack()
                    self.location_inference = []
                    self.inference_progress = 0
                else:
                    print("Warning: no environment has been given, so the robo-Doctor will only palpate the area "
                          "below the palpation starting position.")


                self._compute_cpms()
                self._create_datasets(
                    reduced_train_data=self.reduced_belief_data
                )
                action_benefits, ranked_actions = self.get_best_actions()
                all_ordered_actions += [ranked_actions]

                plot_robot_inference1d(
                    pdfs=self.pdfs[ranked_actions[0]],
                    X_train=self.training_data[ranked_actions[0]],
                    y_train=self.training_labels[ranked_actions[0]],
                    X_test=self.test_data[ranked_actions[0]][:self.inference_progress, :],
                    y_test=self.location_labels[:self.inference_progress],
                    show=True
                )

            self.task_objects = list(sorted(np.unique([obj[task] for obj in self.objects])))
            self.task_object_map = {}

            for obj in self.objects:
                if obj[task] not in self.task_object_map.keys():
                    self.task_object_map[obj[task]] = [obj]
                else:
                    self.task_object_map[obj[task]] += [obj]

    def _dump_experiment_details(self):
        experiment_info = {
            "actions": self.actions,
            "objects": self.objects,
            "environment": self.environment,
            "sensing_resolution": self.sensing_resolution,
            "samples_per_pdf": self.samples_per_pdf,
            "motion_stack": self.motion_stack,
            "skin_shape": self.body.skin_shape
        }
        save_dict_to_file(experiment_info,
                          path=self.body.data_folder,
                          filename="experiment_info")

    def initialize_save(self, action, exp_folder):
        if self.learning:
            obj = action[0]
            action_idx = action[2][0]
            sample_no = action[3]
            exp_name = "{}_{}_{}".format(obj, action_idx, sample_no)

            self.body.writer = SkinWriter(
                shape=(0,) + self.body.skin_shape,
                name=exp_name,
                format="h5",
                folder=exp_folder
            )
        return True

    # While learning, sense and save --- this should work provided the frame rate is high enough!!! or slow-down robot.
    def record_sense(self, step):
        if self.current_time_level is None:
            self.sense_time_step = self.body.palpation_duration/self.sensing_resolution
            self.time_levels = list(np.arange(self.body.palpation_start_time,
                                              self.body.palpation_end_time,
                                              self.sense_time_step)[::-1])
            self.current_time_level = self.time_levels.pop()

        if self.body.current_time >= self.current_time_level:
            try:
                self.skin_buffer[self.skin_buffer_idx, :] = self.body.skin_snapshot[-1, :]
            except:

                # DEBUG
                if self.verbose:
                    print("currently {} sec into palpation. {} time levels left. last time level at {}".format(
                        self.body.current_time-self.body.palpation_start_time,
                        len(self.time_levels),
                        self.current_time_level-self.body.palpation_start_time)
                    )

            self.skin_buffer_idx += 1
            if len(self.time_levels) > 0:
                self.current_time_level = self.time_levels.pop()
                self.idx_count += 1

    # save current stacked motion, and clear
    def save(self):
        if self.learning and self.body.writer is not None:
            self.body.writer.write(self.skin_buffer)
            self.body.writer.release()
        self.skin_buffer = np.zeros((self.sensing_resolution,) + self.body.skin_shape)
        self.skin_buffer_idx = 0

    def discard(self, keep_motion=True):
        if self.learning and self.body.writer is not None:
            self.body.writer.discard()
        self.skin_buffer = np.zeros((self.sensing_resolution,) + self.body.skin_shape)
        self.skin_buffer_idx = 0
        if self.learning and keep_motion:
            self.motion_stack += [self.current_motion]

        if not self.learning:

            plot_robot_inference1d(
                pdfs=self.pdfs[self.body.current_motion[2][0]],
                X_train=self.training_data[self.body.current_motion[2][0]],
                y_train=self.training_labels[self.body.current_motion[2][0]],
                X_test=self.test_data[self.body.current_motion[2][0]][:self.inference_progress, :],
                y_test=self.location_labels[:self.inference_progress],
                predictions=self.location_inference,
                show=True
            )

            # save conf matrix
            plot_conf_matrix(
                targets=self.location_labels,
                outputs=self.location_inference,
                ordered_labels=self.data_objects,
                folder=self.body.results_folder,
                filename="conf_matrix"
            )


    def _create_location_stack(self):
        location_stack = []
        location_labels = []
        for key in self.environment.keys():
            for loc in self.environment[key]:
                location_stack += [loc]
                location_labels += [key]
        return location_stack, location_labels

    def _load_experiment_details(self):
        try:
            json_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith(".json")]))
            json_filename = self.body.data_folder + json_files.pop()
            with open(json_filename, 'r') as f:
                experiment_info = json.load(f)
            self.actions = experiment_info["actions"]
            self.objects = experiment_info["objects"]
            self.samples_per_pdf = experiment_info["samples_per_pdf"]
            self.sensing_resolution = experiment_info["sensing_resolution"]
            self.number_of_actions = len(self.actions)
            self.number_of_objects = len(self.objects)
            self.loaded_skin_shape = experiment_info["skin_shape"]
        except FileNotFoundError as e:
            print("\nERROR: It seems like no experiments have been performed, so it is impossible to 'continue' from stored data.\n"
                  "Please, set 'resume_previous_experiment' to False\n")
            raise e

    def _load_data(self):
        # ------------ Load Data from files ----------

        self.train_samples_per_pdf = int(np.floor((self.samples_per_pdf)*self.training_ratio))
        self.test_samples_per_pdf = self.samples_per_pdf - self.train_samples_per_pdf

        # skin_data filenames
        self.skin_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith(".h5")]))
        # indexes to find initial and test data in data_folder
        sample_belief_idxs = {}
        sample_test_idxs = {}
        self.moving_idxes = {}  # need this to know which file to retrieve at the "next" iteration during run

        self.prior = 1. / self.number_of_objects
        self.original_belief_data = {}
        self.original_test_data = {}

        # indeces to remember where in the original_data[act] are the datapoints belonging to specific objects
        self.pdfs_belief_indeces = {}
        self.pdfs_test_indeces = {}

        # number of times the actions have been explored
        self.action_counts = {}

        self.missing_actions = []
        self.missing_objects = {}
        self.pcas = {}
        self.normalize_max = {}
        self.normalize_min = {}

        for act, _, _, in self.actions:
            if act not in sample_belief_idxs.keys() or act not in sample_test_idxs.keys():
                sample_belief_idxs[act] = {}
                sample_test_idxs[act] = {}
                self.moving_idxes[act] = list(range(self.samples_per_pdf))
                self.missing_objects[act] = []
            for obj in self.objects:
                obj_act_indexes = list(range(self.samples_per_pdf))
                for sample_no in list(range(self.samples_per_pdf)):
                    # get filename and check if exists...
                    skin_filename = self.body.data_folder + "{}_{}_{}.h5".format(obj, act, sample_no)
                    if skin_filename.split("/")[-1] not in self.skin_files:
                        obj_act_indexes.remove(sample_no)
                        if sample_no in self.moving_idxes[act]: self.moving_idxes[act].remove(sample_no)
                if self.train_samples_per_pdf + self.test_samples_per_pdf > len(obj_act_indexes):
                    if len(obj_act_indexes) == 0:
                        self.missing_actions += [act]
                        self.missing_objects[act] += [obj]
                    else:
                        raise SystemError("there isn't enough data to perform experiments in folder '{}'"
                                          .format(self.body.data_folder))
                else:
                    sample_belief_idxs[act][obj] = obj_act_indexes[0:self.train_samples_per_pdf]
                    sample_test_idxs[act][obj] = obj_act_indexes[self.samples_per_pdf - self.test_samples_per_pdf:]
                    [self.moving_idxes[act].remove(x) for x in sample_belief_idxs[act][obj] + sample_test_idxs[act][obj]
                     if x in self.moving_idxes[act]]

        # ------------ Dimensionality Reduction --------
        # fit pca to all gathered data - i.e. choose eigenvector directions in hyperspace
        # also, save indeces to create pdfs after
        self.belief_idx = {}
        self.test_idx = {}
        for act, _, _, in self.actions:
            if act not in self.missing_actions:
                if act not in self.original_belief_data.keys() or act not in self.original_test_data.keys():
                    self.belief_idx[act] = 0
                    self.test_idx[act] = 0
                    self.original_belief_data[act] = np.zeros((
                        self.number_of_objects * (self.train_samples_per_pdf + self.test_samples_per_pdf),
                        (self.sensing_resolution-1) * np.prod(list(self.loaded_skin_shape))
                    ))
                    self.action_counts[act] = 0
                    self.pdfs_belief_indeces[act] = {}
                    self.pdfs_test_indeces[act] = {}

                for obj in self.objects:
                    idx_belief_sample_start = self.belief_idx[act]
                    idx_test_sample_start = self.test_idx[act]
                    if obj not in self.missing_objects[act]:
                        # randomize sample numbers, so non-biased distribution for training/testing
                        # random_samples = np.random.choice(self.samples_per_pdf, self.samples_per_pdf, replace=False)
                        for sample_no in sample_belief_idxs[act][obj] + sample_test_idxs[act][obj]:

                            # get filename and check if exists...
                            skin_filename = self.body.data_folder + "{}_{}_{}.h5".format(obj, act, sample_no)
                            if skin_filename.split("/")[-1] in self.skin_files:

                                self.skin_files.remove(skin_filename.split("/")[-1])  # pop data so we don't re-load it

                                skin_file = tables.open_file(skin_filename, mode='r')
                                skin_data = skin_file.root.data[:]
                                skin_file.close()

                                skin_data = skin_data - skin_data[0]

                                self.original_belief_data[act][self.belief_idx[act], :] = skin_data[1:].flatten()
                                self.belief_idx[act] += 1

                            else:
                                raise SystemError(
                                    "there is missing data in the folder! I can't find '{}'.".format(skin_filename))

                    self.pdfs_belief_indeces[act][obj] = [(idx_belief_sample_start, self.belief_idx[act])]
                    self.pdfs_test_indeces[act][obj] = [(idx_test_sample_start, self.test_idx[act])]

                self.action_counts[act] += 1

        # normalize_min = np.min(self.original_belief_data, axis=0)
        # normalize_max = np.max(self.original_belief_data, axis=0)
        # reduced_act_belief_data = (self.original_belief_data - normalize_min) / (
        #         normalize_max - normalize_min)  # normalize
        # self.pca = PCA(n_components=self.dimensionality_reduction)
        # self.pca.fit(reduced_act_belief_data)
        return self.original_belief_data.copy()

    def _fit_pdfs(self, belief_data, act=None, reduced_belief_data=None, reduced_test_data=None):

        # ------------ Fit Likelihoods --------------------
        if self.pdfs is None:
            self.pdfs = dict()

        if act is not None and reduced_test_data is not None and reduced_belief_data is not None:
            iteration_actions = [(act, None, None)]
        else:
            iteration_actions = self.actions.copy()
            reduced_belief_data = dict()
            reduced_test_data = dict()

        # n_cols = all_data.shape[1]
        for act, _, _ in iteration_actions:
            if act not in self.missing_actions:
                if act not in self.pdfs.keys():
                    self.pdfs[act] = {}
                if act not in reduced_test_data.keys():
                    reduced_belief_data[act] = {}
                    reduced_test_data[act] = {}
                    self.original_diagnosis_data = {}
                    self.reduced_diagnosis_data = {}

                mat_obj_belief_copy = {}
                for obj in self.objects:
                    if obj not in self.missing_objects[act]:
                        # TRAIN DATA - multiple set of indexes added at different times beacuse robot keeps training
                        for i in range(len(self.pdfs_belief_indeces[act][obj])):
                            if obj not in mat_obj_belief_copy.keys():
                                mat_obj_belief_copy[obj] = belief_data[act][
                                                           self.pdfs_belief_indeces[act][obj][0][0]:
                                                           self.pdfs_belief_indeces[act][obj][0][1],
                                                           :
                                                           ].copy()
                            else:
                                mat_obj_belief_copy[obj] = np.append(arr=mat_obj_belief_copy[obj],
                                                                     values=belief_data[act][
                                                                            self.pdfs_belief_indeces[act][obj][i][0]:
                                                                            self.pdfs_belief_indeces[act][obj][i][1],
                                                                            :].copy(),
                                                                     axis=0)

                self.normalize_min[act] = np.min(self.original_belief_data[act].copy(), axis=0)
                self.normalize_max[act] = np.max(self.original_belief_data[act].copy(), axis=0)
                self.normalize_max[act] = [maxx if maxx-minx != 0 else maxx+1
                                           for minx, maxx in zip(self.normalize_min[act], self.normalize_max[act])]
                normalized_belief_data = (self.original_belief_data[act].copy() - self.normalize_min[act]) / (
                        self.normalize_max[act] - self.normalize_min[act])  # normalize
                self.pcas[act] = PCA(n_components=self.dimensionality_reduction)
                self.pcas[act].fit(normalized_belief_data)

                for obj in self.objects:
                    mat_obj_belief_copy[obj] = (mat_obj_belief_copy[obj] - self.normalize_min[act]) / (
                            self.normalize_max[act] - self.normalize_min[act])
                    reduced_belief_mat = self.pcas[act].transform(mat_obj_belief_copy[obj])
                    mean = np.mean(reduced_belief_mat, axis=0)
                    cov = np.cov(reduced_belief_mat, rowvar=False)
                    # if len(cov.shape) > 0:
                    #     np.identity(2) * .000001*cov
                    # else:
                    #     cov += .000001
                    self.pdfs[act][obj] = (mean, cov)
                    reduced_belief_data[act][obj] = reduced_belief_mat

        return reduced_belief_data

    def _fit_pdfs_online(self, belief_data):

        self.pcas = {}
        self.normalize_max = {}
        self.normalize_min = {}
        self.prior = 1. / len(self.task_objects)

        # ------------ Fit Likelihoods --------------------
        if self.pdfs is None:
            self.pdfs = dict()

        reduced_belief_data = dict()

        # n_cols = all_data.shape[1]
        for act in self.belief_data.keys():
            if act not in self.pdfs.keys():
                self.pdfs[act] = {}
            if act not in reduced_belief_data.keys():
                reduced_belief_data[act] = {}
                mat_all_objs = None
                for task_obj in belief_data[act].keys():
                    # this belief data is different than when loading from folder, it's ordered in dict
                    if mat_all_objs is None:
                        mat_all_objs = self.belief_data[act][task_obj]
                    else:
                        mat_all_objs = np.append(arr=mat_all_objs,
                                                 values=self.belief_data[act][task_obj],
                                                 axis=0)

                self.normalize_min[act] = np.min(mat_all_objs, axis=0)
                self.normalize_max[act] = np.max(mat_all_objs, axis=0)
                self.normalize_max[act] = [maxx if maxx-minx != 0 else maxx+1
                                           for minx, maxx in zip(self.normalize_min[act], self.normalize_max[act])]
                normalized_belief_data = (mat_all_objs - self.normalize_min[act]) / (
                        self.normalize_max[act] - self.normalize_min[act])  # normalize

                self.pcas[act] = PCA(n_components=self.dimensionality_reduction)

                if self.dimensionality_reduction > normalized_belief_data.shape[0]:
                    # trick fit repeating data, but only until enough data to properly do pca
                    self.pcas[act].fit(np.tile(normalized_belief_data,
                            (int(np.ceil(self.dimensionality_reduction/normalized_belief_data.shape[0])) , 1)))
                else:
                    self.pcas[act].fit(normalized_belief_data)

                for obj in sorted(self.belief_data[act].keys()):
                    normal_belief_obj_copy = (belief_data[act][obj] - self.normalize_min[act]) / (
                            self.normalize_max[act] - self.normalize_min[act])
                    reduced_belief_mat = self.pcas[act].transform(normal_belief_obj_copy)
                    mean = np.mean(reduced_belief_mat, axis=0)
                    # if reduced_belief_mat.shape[0] == 1:
                    #     cov = np.var(reduced_belief_mat)
                    # else:
                    cov = np.cov(reduced_belief_mat, rowvar=False)
                    self.pdfs[act][obj] = (mean, cov)
                    reduced_belief_data[act][obj] = reduced_belief_mat

        return reduced_belief_data


    def _create_datasets(self, act=None, reduced_train_data=None):
        self._compute_cpms()

        self.training_data = {}
        self.test_data = {}
        self.training_labels = {}
        self.test_labels = {}

        # retrieve data for best motion
        action_benefit = self.get_perceived_benefit()
        self.data_objects = sorted(reduced_train_data[self.actions[0][0]].keys())
        label_dict = {}
        for i, obj in enumerate(self.data_objects):
            label_dict[obj] = i

        obj_keys = self.data_objects

        if act == None:
            actions = [act[0] for act in self.actions]
        else:
            actions = [act]

        for act in actions:
            if act not in self.training_labels.keys():
                self.training_data[act] = np.zeros(
                    (self.samples_per_pdf * len(self.data_objects), self.dimensionality_reduction))
                self.test_data[act] = np.zeros(
                    (len(self.location_stack) * len(self.data_objects), self.dimensionality_reduction))
                self.training_labels[act] = []
                self.test_labels[act] = []
            for j, obj in enumerate(obj_keys):
                if obj not in self.missing_objects[act]:
                    # TRAINING DATA (currently not used)
                    self.training_data[act][j * self.samples_per_pdf:j * self.samples_per_pdf + self.samples_per_pdf, :] \
                        = reduced_train_data[act][obj]
                    self.training_labels[act] += [label_dict[obj]] * self.samples_per_pdf
        return True

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
                folder=self.body.results_folder,
                windows_name="",
                show=show,
                save=True
            )

        if reduced_train_data is not None and action_benefit is not None and self.dimensionality_reduction <= 2:
            # UNCOMMENT TO SEE MOTION
            fig_best, fig_worst = plot_normals(
                self.pdfs, reduced_train_data, action_benefit,
                folder=self.body.results_folder,
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
        for action in self.pdfs.keys():
            div_mat = np.zeros((self.number_of_objects, self.number_of_objects))
            for i, obj1 in enumerate(sorted(self.pdfs[action].keys())):
                for j, obj2 in enumerate(sorted(self.pdfs[action].keys())):
                    div_mat[i, j] = compute_distance_coefficient(self.pdfs[action][obj1], self.pdfs[action][obj2])
            self.cpms[action] = div_mat
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

    # function to pop a motion from stack or make one
    def get_motion(self):

        global TIME_LEFT

        if self.learning:
            # --------- PROGRESS LEVEL -----------

            TIME_LEFT = (time.clock() - self.clock) * len(self.motion_stack)
            print("\nExecuted number of palpations: {} out of {}. Progress at {}%.   Avg. Frame Rate: {}   --- Time Left: {}".format(
                self.total_number_of_experiments-len(self.motion_stack),
                self.total_number_of_experiments,
                format((self.total_number_of_experiments-len(self.motion_stack))*100/self.total_number_of_experiments, '.2f'),
                format(FRAME_RATE, '.2f'),
                get_time(TIME_LEFT)
            ))
            # -------------------------------------

            # -- re-set variables
            self.clock = time.clock()
            self.current_time_level = None
            self.time_levels = None

            self.data_idx = 0
            self.idx_count = 0

            # -- set variables
            if len(self.motion_stack) == 0:
                return None
            motion = self.motion_stack.pop()
            self.current_motion = motion

            # initialize stack for motion sensing
            # note "self.current_motion[2]" is the action, and self.current_motion[2][0] is the action identifier (int)
            if not self.current_motion[2][0] in self.idx_dict.keys():
                self.idx_dict[self.current_motion[2][0]] = {}
            if not self.current_motion[0] in self.idx_dict[self.current_motion[2][0]].keys():
                self.idx_dict[self.current_motion[2][0]][self.current_motion[0]] = \
                    [np.zeros(((self.sensing_resolution,) + self.body.skin_snapshot.shape))]
            if len(self.idx_dict[self.current_motion[2][0]][self.current_motion[0]][0]) != 0:
                self.idx_dict[self.current_motion[2][0]][self.current_motion[0]] += \
                    [np.zeros(((self.sensing_resolution,) + self.body.skin_snapshot.shape))]
        else:
            # --------- PROGRESS LEVEL -----------

            TIME_LEFT = (time.clock() - self.clock) * (len(self.location_stack) - self.inference_progress)
            print("\nLocations Palpated: {} out of {}. Progress at {}%.   Avg. Frame Rate: {}   --- Time Left: {}".format(
                self.inference_progress,
                len(self.location_stack),
                format(self.inference_progress*100/len(self.location_stack), '.2f'),
                format(FRAME_RATE, '.2f'),
                get_time(TIME_LEFT)
            ))
            # -------------------------------------

            if self.inference_progress >= len(self.location_stack):
                return None
            location = self.location_stack[self.inference_progress]
            self.inference_progress += 1
            act_identifier = self.get_best_actions()[1][0]
            act = [motion for motion in self.actions if motion[0] == act_identifier][0]
            motion = (None, location, act, None)
        return motion

    def _execute_motion(self, moving, step):
        # logic behind what to do during palpation (once contact is established this function handles the robot motion)
        if not self.body.end_palpate:
            if not self.body.touched:
                if self.body.writer is None:
                    res = self.initialize_save(self.body.current_motion, self.body.data_folder)
                    if res is False:
                        return False
                if self.verbose:
                    print("Descending...")
                self.body.descending = True
                if not moving:
                    # Handle morph filter - self.current_motion[2][-1] is the filter action
                    # if self.current_motion[2][-1] < 0:
                    #     self.body.filter.vacuum(np.abs(self.current_motion[2][-1]))
                    # else:
                    #     self.body.filter.pump(self.current_motion[2][-1])

                    self.body.robot.set_speed(vel=-.001, acc=.05, t=10)  # go down
                else:
                    if self.verbose:
                        print("Checking for contact...")
                    # Check Skin and trigger palpation if felt contact -- only if you're not in the middle of palpation
                    skin_diff = self.body.skin_snapshot[-1, :] - self.body.skin_snapshot[0, :]
                    # print(skin_diff)
                    # if np.count_nonzero(np.array(skin_diff > 30)) >= 4 and self.body.state['actual_TCP_pose'][2] < -.048:
                    if np.count_nonzero(np.array(skin_diff > 8)) >= 1 and self.body.state['actual_TCP_pose'][2] < 0.068:
                        # check if contact has happened 3 consecutive times, otherwise it may be a false positive
                        self.body.detected_contact_number += 1
                        if self.body.detected_contact_number >= 2 \
                                and time.clock() - self.body.skin_contact_time > self.body.skin_time_lag:
                            self.body.detected_contact_number = 0
                            self.body.skin_contact_time = time.clock()
                            # if touched, start palpation!
                            self.body.touched = True
                            self.body.robot.joint_stop(acc=3.)
                            self.body.robot_stop = True
                            self.body.palpation_start_pose = self.body.state['actual_TCP_pose']
                            if self.verbose:
                                print("TOUCHED!!!")
                        # print(self.body.palpation_start_pose)
            else:
                # clock start of palpation, so can stop after a specified time
                self.body.current_time = time.clock()
                if self.body.palpation_start_time is None:
                    self.body.palpation_start_time = time.clock()
                    self.body.palpation_end_time = self.body.palpation_start_time + self.body.palpation_duration

                # if out of time save all, otherwise handle move
                if self.body.current_time >= self.body.palpation_end_time:
                    self.body.end_palpate = True
                    self.body.robot.joint_stop(acc=3.)
                    self.robot_stop = True
                    # self.body.filter.stop()
                else:
                    # compute speed profiles given motion, at current time step, and execute on robot.
                    etas = np.array(self.body.current_motion[2][1])
                    alphas = np.array(self.body.current_motion[2][2])
                    self.body.velocities = -alphas*etas*np.sin(etas*(self.body.current_time-self.body.palpation_start_time))
                    self.body.velocities = [0 if math.isnan(elem) or math.isinf(elem) else elem for elem in self.body.velocities]
                    if self.verbose:
                        print(self.body.velocities)
                    self.body.robot.set_speed(axis='all', vel=self.body.velocities, acc=2., t=10)
                    self.record_sense(step=step)

        else:
            if self.verbose:
                print("Ascending...")
            if not moving:
                # self.body.robot.cart_go_to(pose=self.body.descent_start_pose, acc=.3, vel=.3)
                self.body.robot.set_speed(vel=.03, acc=.05, t=10)    # go back
            else:
                if self.body.descent_start_pose[2] - self.body.state['actual_TCP_pose'][2] <= 0.0001:
                    # stop robot
                    if self.verbose:
                        print("All done! on to another bead")
                    self.body.robot.joint_stop(acc=5.)
                    self.robot_stop = True

                    # save locally for learning
                    if self.learning:
                        if self.body.current_motion[2][0] not in self.belief_data.keys():
                            self.belief_data[self.body.current_motion[2][0]] = {}
                        if self.body.current_motion[0][self.task] not in self.belief_data[self.body.current_motion[2][0]]:
                            self.belief_data[self.body.current_motion[2][0]][self.current_motion[0][self.task]] = \
                                (self.skin_buffer - self.skin_buffer[0])[1:].reshape(1, -1)
                        else:
                            self.belief_data[self.body.current_motion[2][0]][self.current_motion[0][self.task]] = np.append(
                                arr=self.belief_data[self.body.current_motion[2][0]][self.current_motion[0][self.task]],
                                values=(self.skin_buffer - self.skin_buffer[0])[1:].reshape(1, -1),
                                axis=0
                            )

                        self.reduced_belief_data = self._fit_pdfs_online(self.belief_data)
                        X_train = None
                        y_train = []

                        for obj in self.reduced_belief_data[self.body.current_motion[2][0]].keys():
                            if X_train is None:
                                X_train = self.reduced_belief_data[self.body.current_motion[2][0]][obj]
                            else:
                                X_train = np.append(arr=X_train,
                                                    values=self.reduced_belief_data[self.body.current_motion[2][0]][obj],
                                                    axis=0)
                            y_train += [self.task_objects.index(obj)] * \
                                       self.reduced_belief_data[self.body.current_motion[2][0]][obj].shape[0]


                        self._compute_cpms()
                        # plot_robot_inference1d(
                        #     pdfs=self.pdfs[self.body.current_motion[2][0]],
                        #     X_train=X_train,
                        #     y_train=y_train,
                        #     show=True,
                        #     save=True,
                        # )
                        plot_robot_inference2d(
                            pdfs=self.pdfs[self.body.current_motion[2][0]],
                            X_train=X_train,
                            y_train=y_train,
                            act=self.body.current_motion[2][0],
                            filter=self.body.current_motion[2][-1],
                            objects=self.task_objects,
                            task=self.task,
                            show=True,
                            save=True,
                            folder=self.body.results_folder
                        )

                    else:
                        # -------- PREDICT -----------
                        # -- normalized skin data
                        skin_data = self.skin_buffer - self.skin_buffer[0]
                        normalized_skin_buffer = (skin_data[1:].reshape(1, -1) - self.normalize_min[self.body.current_motion[2][0]]) / \
                                                 self.normalize_max[self.body.current_motion[2][0]] - self.normalize_min[self.body.current_motion[2][0]]
                        normalized_skin_buffer = np.array([x if not math.isinf(x) and not math.isnan(x) else 0 for x in normalized_skin_buffer.flatten()]).reshape(1, -1)

                        # -- PCA reduce normalized data
                        reduced_buffer = self.pcas[self.body.current_motion[2][0]].transform(normalized_skin_buffer)

                        # -- Bayesian predict from PCA projection & save
                        buffer_prediction = bayesian_predict(self.pdfs[self.body.current_motion[2][0]], reduced_buffer)
                        self.test_data[self.body.current_motion[2][0]][self.inference_progress - 1, :] = reduced_buffer
                        self.location_inference += [buffer_prediction[0]]

                        # -- check prediction
                        if self.location_inference[self.inference_progress - 1] == self.location_labels[self.inference_progress - 1]:
                            print("Correct prediction {}!", self.location_inference[self.inference_progress - 1])
                        else:
                            print("Incorrect prediction. The doctor thought it was '{}' while instead it was '{}'".format(
                                self.location_inference[self.inference_progress - 1],
                                self.location_labels[self.inference_progress - 1]
                            ))

                        plot_robot_inference1d(
                            pdfs=self.pdfs[self.body.current_motion[2][0]],
                            X_train=self.training_data[self.body.current_motion[2][0]],
                            y_train=self.training_labels[self.body.current_motion[2][0]],
                            X_test=self.test_data[self.body.current_motion[2][0]][:self.inference_progress, :],
                            y_test=self.location_labels[:self.inference_progress],
                            predictions=self.location_inference,
                            show=True
                        )

                    # save and move on
                    self.save()
                    self.body._reset_palpation()

        return True


def bayesian_predict(pdfs, data):
    labels = []
    for i in range(data.shape[0]):
        label = 0
        max_prob = 0
        for j, obj in enumerate(sorted(pdfs.keys())):
            mu, sig = pdfs[obj]
            prob = gaussian(data[i], mu, sig)
            if prob > max_prob:
                max_prob = prob
                label = j
        labels += [label]
    return np.array(labels)


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