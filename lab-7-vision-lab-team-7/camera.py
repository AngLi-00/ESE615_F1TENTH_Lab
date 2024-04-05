import cv2
import time

# Open the camera using GStreamer pipeline with v4l2src to capture at 60Hz
cap = cv2.VideoCapture("v4l2src device=/dev/video4 extra-controls=\"c,exposure_auto=3\" ! "
                       "video/x-raw, width=960, height=540, framerate=60/1 ! "
                       "videoconvert ! video/x-raw, format=BGR ! appsink")

# Check if the camera was successfully opened
if cap.isOpened():
    cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)

    # Target time per frame to achieve close to 60Hz frame rate
    target_time = 1.0 / 60.0

    while True:
        # Capture the current time
        start_time = time.time()

        # Read a frame
        ret_val, img = cap.read()
        if not ret_val:
            print("Failed to grab a frame")
            break

        # Display the frame
        cv2.imshow('demo', img)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate the actual time taken to grab and show the frame
        elapsed = time.time() - start_time

        # Wait to maintain the target frame rate
        wait_time = max(target_time - elapsed, 0)
        time.sleep(wait_time)
        print(1/(time.time() - start_time), 'Hz')

else:
    print("Camera open failed")

cv2.destroyAllWindows()

# import cv2
# import time

# cap = cv2.VideoCapture("v4l2src device=/dev/video4 extra-controls=\"c,exposure_auto=3\" ! "
#                        "video/x-raw, width=960, height=540, framerate=60/1 ! "
#                        "videoconvert ! video/x-raw, format=BGR ! appsink")

# time_old = time.time()
# if cap.isOpened():
#     cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
#     while True:
#         time_now = time.time()
#         ret_val, img = cap.read()
#         print(1/(time_now - time_old), 'Hz')
#         time_old = time_now
#         cv2.imshow('demo', img)
#         cv2.waitKey(1)
# else:
#     print("camera open failed")

# cv2.destroyAllWindows()
