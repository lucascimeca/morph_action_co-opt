import serial
import time

class MorphingFilter(object):

    def __init__(self, COM=7, baud_rate=9600, verbose=False):

        print("Enstablishing connection to Arduino...")
        self.arduino = serial.Serial(
            "COM{}".format(COM),
            baud_rate,
            timeout=.1
        )
        time.sleep(2)
        print("Connection enstablished.")

        self.verbose = verbose

    def send(self, cmd):
        cmd += "\n"
        self.arduino.write(cmd.encode())
        res = int(self.arduino.readline().decode())
        if self.verbose:
            if res:
                print("command '{}' sent with success.".format(cmd[:-1]))
            else:
                print("command '{}' not processed, there may be a mistake in the syntax, please check."
                      .format(cmd[:-1]))
            print(res)
        return res

    def pump(self, level):
        if level > 255:
            level = 255
        elif level < 0:
            level = 0
        cmd = "b:{}".format(level)
        return self.send(cmd)

    def vacuum(self, level):
        if level > 255:
            level = 255
        elif level < 0:
            level = 0
        cmd = "f:{}".format(level)
        return self.send(cmd)

    def stop(self):
        return self.send('s')
