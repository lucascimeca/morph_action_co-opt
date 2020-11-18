import serial  # import serial library
import time
from morph_filter import MorphingFilter

if __name__ == "__main__":

    filter = MorphingFilter(
        COM=4,
        baud_rate=9600,
        verbose=False
    )

    # t = time.time()
    # filter.pump(25)
    # while time.time() - t < 15:
    #     pass
    #
    # t = time.time()
    # filter.pump(50)
    # while time.time() - t < 15:
    #     pass

    print('trying 100')
    t = time.time()
    filter.pump(255)
    while time.time() - t < 15:
        pass

    # print('trying 130')
    # t = time.time()
    # filter.pump(130)
    # while time.time() - t < 15:
    #     pass

    filter.stop()
    time.sleep(2)

    print('trying reverse')
    t = time.time()
    filter.vacuum(255)
    while time.time() - t < 10:
        pass


    # for i in range(255):
    #     filter.pump(i)
    #
    # for i in range(255, 0, -1):
    #     filter.pump(i)
    #
    # for i in range(255):
    #     filter.vacuum(i)
    #
    # for i in range(255, 0, -1):
    #     filter.vacuum(i)

    filter.stop()
