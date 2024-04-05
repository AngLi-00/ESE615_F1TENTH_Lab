import distance
import lane
import detection
import numpy as np
import cv2 as cv

def get_distance(center_x, center_y):
    """
    calculate the distance between the object and the camera
    """
    camera_matrix = np.array([[694.71543087,   0,         449.37540769],
                            [0,         695.5496121,  258.64705735],
                            [0,           0,           1,        ]])
    H_mount = 0.13477281192899426

    f_x = camera_matrix[0][0]
    f_y = camera_matrix[1][1]
    x_0 = camera_matrix[0][2]
    y_0 = camera_matrix[1][2]
    x_car = f_y*(H_mount)/(center_y - y_0)
    y_car = (center_x - x_0)/f_x*x_car
    return x_car, y_car

if __name__ == '__main__':
    # Get picture from camera
    # camera_capture = cv.VideoCapture(0)
    # camera_capture = 'resource/test_car_x60cm.png'
    camera_capture = 'detection.jpg'
    
    # lane detection
    # lane.detect_lanes(camera_capture)
    # cv.imshow('Lane Detection Result', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    

    # object and distance detection
    model_path = 'model_100.trt'
    center_x, center_y = detection.detect_and_display(camera_capture, model_path)
    print(center_x, center_y)
    center_x = center_x/320*960
    center_y = center_y/180*540
    print(center_x, center_y)

    object_x, object_y = get_distance(center_x, center_y)
    print(object_x, object_y)