try:
    import skin_run
except:
    print("Could not import library 'skin_run'")
import numpy as np
import tables
from simple_io import *
try:
    import xlwt
except:
    print("could not import library 'xlwt'")
import os

class SkinData:
    data = None
    skin_base = None

    def __init__(self, live=True, filename=None, calibrate=True):
        self.live = live
        if self.live:
            print("Initializing Skin...")
            self.skin = skin_run.PySkinRun()
            print("Skin ok!")
            print("Starting acquisition...")
            self.skin.start_acquisition()
            self.t_num = self.skin.skin_length()
            self.layout = self._get_layout()
            self.shape, self.taxels_to_pad = self._get_shape()

            self._calibrate_skin()
            print("Skin calibrated!")

        else:
            if filename:
                self.file = tables.open_file(filename, mode='r')
                self.idx = -1
                self.shape = self.file.root.data.shape
            else:
                raise ValueError("Please, provide a 'filename' containing recorded skin data.")

    def _calibrate_skin(self):
        print("Calibrating skin...")
        _, skin_base = self.read()
        skin_base.astype(np.int64)
        for i in range(19):
            next, new_vals = self.read()
            skin_base += new_vals
        self.skin_base = skin_base / 20

    def _get_layout(self):
        skin_layout = {
            'ihb_layout': np.zeros(self.t_num).astype(np.int),
            'module_layout': np.zeros(self.t_num).astype(np.int),
            'sensor_layout': np.zeros(self.t_num).astype(np.int)
        }
        self.skin.get_layout(
            skin_layout['ihb_layout'],
            skin_layout['module_layout'],
            skin_layout['sensor_layout']
        )
        return skin_layout

    # internal! only use internally to figure out shape, otherwise use the next function
    def _get_shape(self):
        # tests, to check for conformity, otherwise cannot do reshape
        print("checking for conformity of skin...")
        taxels_to_pad = 0  # number of taxels to pad so can reshape
        _, ihb_counts = np.unique(self.layout['ihb_layout'], return_counts=True)
        if len(set(ihb_counts)) != 1:
            raise ValueError("Not all IHBs have the same number of taxels, so can't properly reshape in numpy!")

        _, module_counts = np.unique(self.layout['module_layout'], return_counts=True)
        if len(set(module_counts)) != 1:
            print("Not all Modules have the same number of taxels, so can't properly reshape in numpy!")
            if len(set(module_counts[:-1])) == 1:
                print("NOTE: I'm padding some taxels to be able to reshape, look out for taxels with value '-1'!")
                taxels_to_pad = module_counts[0] - module_counts[-1]

        return (ihb_counts.shape[0], module_counts.shape[0], -1), taxels_to_pad

    def get_shape(self):
        _, skin_read = self.read()
        return skin_read.shape

    def read(self):
        if self.live:
            skin_reading = np.ones(self.t_num + self.taxels_to_pad).astype(np.int) * -1
            self.skin.read_skin(skin_reading)  # load skin reading, in-place!
            res = skin_reading.reshape(self.shape)
            if self.skin_base is not None:
                return True, res - self.skin_base
            return True, res
        else:
            self.idx += 1
            return self.idx < self.shape[0]-1, self.file.root.data[self.idx, :]

    def skin_contact(self):
        skin, _ = self.read()
        if np.any(np.abs(skin) > 50):
            return True
        return False

    def close(self):
        self.skin.start_acquisition()
        return True


class SkinWriter:

    def __init__(self, shape=None, name="skin_out", format="h5", folder="./../data/"):
        self.name = name
        self.format = format
        self.folder = folder
        self.shape = shape

        # name_base = ''.join([char for char in self.name if not char.isdigit()])
        # num = ''.join([char for char in self.name if char.isdigit()])
        # if len(num) == 0:
        #     num = 0
        # else:
        #     num = int(num)
        # self.filename = "{}{}.{}".format(self.folder, name_base + str(num), self.format)

        self.filename = "{}{}.{}".format(self.folder, self.name, self.format)

        num = 0
        while file_exists(self.filename):
            self.filename = "{}{}-({}).{}".format(self.folder, self.name, num, self.format)
            num += 1
        self.file = tables.open_file(self.filename, mode='w')
        self.file.create_earray(self.file.root, 'data', tables.Int32Atom(), self.shape)

    def write(self, new_data):
        # self.file.root.data.append(new_data.reshape((1,) + new_data.shape))
        self.file.root.data.append(new_data)

    def discard(self):
        try:
            if self.file.isopen == 1:
                self.file.close()
                os.remove(self.filename)
        except Exception as e:
            print(e)

    def release(self):
        self.file.close()


class RobotWriter:

    def __init__(self, name="robot_out", folder="./../data/"):
        self.name = name
        self.folder = folder
        self.filename = "{}{}.{}".format(self.folder, self.name, self.format)

        self.book = xlwt.Workbook()
        self.sheet = self.book.add_sheet("1")
        self.sheet.write(0, 0, "length")
        self.sheet.write(1, 0, "time")
        self.sheet.write(3, 0, "force")
        self.sheet.write(3, 1, "height")

        if file_exist_query(self.filename):
            pass
            # okay to (over)write
            # self.file = tables.open_file(self.filename, mode='w') todo: overwrite xls
        else:
            # todo: look up available filename in directory
            raise NotImplementedError("I should be looking for the next available filename, but I can't right now.")


