import numpy as np
import cv2
import time

video_capture_0 = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
video_capture_1 = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

frame_width = video_capture_0.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture_0.get(cv2.CAP_PROP_FRAME_HEIGHT)

# video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

camera0 = cv2.VideoWriter(
    './../manuscript/media/out_cam_{}.avi'.format(0),
    cv2.VideoWriter_fourcc('M','J','P','G'),   #*'XVID'
    30,  # todo: parameter must be based on time interval
    (np.int32(frame_width), np.int32(frame_height)),
    True
)
camera1 = cv2.VideoWriter(
    './../manuscript/media/out_cam_{}.avi'.format(1),
    cv2.VideoWriter_fourcc('M','J','P','G'),   #*'XVID'
    30,  # todo: parameter must be based on time interval
    (np.int32(frame_width), np.int32(frame_height)),
    True
)


initial_time = time.time()
prev_time = time.time()
while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        cv2.imshow('Cam 0', frame0)
    # camera0.write(frame0)
    #
    if (ret1):
        cv2.imshow('Cam 1', frame1)
    # camera1.write(frame1)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or time.time() - initial_time > 60*.1:
        break
    else:
        print("time left {}sec".format(60*6 - (time.time() - initial_time)))

    print("Frame rate: {}fps".format(1/(time.time()-prev_time)))
    prev_time = time.time()

camera0.release()
camera1.release()