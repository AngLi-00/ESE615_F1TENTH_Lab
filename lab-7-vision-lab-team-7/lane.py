import cv2
import numpy as np

def detect_lanes(image):
    # convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range
    lower_yellow = np.array([20, 65, 80])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # find countours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw green borders around lane markers
    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

    return image

if __name__ == "__main__":
    image = cv2.imread('/home/nvidia/lab-7-vision-lab-team-7/resource/lane.png')

    result = detect_lanes(image)

    cv2.imshow('Lane Detection Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
