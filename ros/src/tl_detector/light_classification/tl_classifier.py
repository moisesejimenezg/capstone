import os

import enum
import numpy as np
import cv2
import pathlib
from keras.models import load_model
from labeler import Labeler
from styx_msgs.msg import TrafficLight


MODEL_LOCATION = str(pathlib.Path(os.getcwd()).joinpath("model.h5"))


def resize_image(image):
    new_height = int(image.shape[0] * 0.25)
    new_width = int(image.shape[1] * 0.25)
    dim = (new_width, new_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class TLClassifier(object):
    def __init__(self, mode="rb"):
        # TODO load classifier
        if mode == "wb":
            self.__labeler = Labeler(mode)
        self.__model = None

        if os.path.isfile(MODEL_LOCATION):
            self.__model = load_model(MODEL_LOCATION)
        else:
            print("Could not init TLClassifier!")

    @property
    def model(self):
        assert self.__model is not None
        return self.__model

    def save_image(self, image, label):
        cv2.imwrite("test.png", image)
        return self.__labeler.label_image(image, label)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            uint8 UNKNOWN=4
            uint8 GREEN=2
            uint8 YELLOW=1
            uint8 RED=0
        """
        # TODO implement light color prediction
        if self.__model is None:
            return TrafficLight.UNKNOWN

        image_array = np.array(resize_image(image))
        traffic_light = int(
            self.model.predict(image_array[None, :, :, :], batch_size=1)
        )
        prob = self.model.predict_proba(image_array[None, :, :, :], batch_size=1)

        if prob < 0.5:
            print("Using hough due to low probability: " + str(prob))
            return self.__hough_stop_light_detector(image_array)

        if traffic_light == 0:
            return TrafficLight.RED
        elif traffic_light == 1:
            return TrafficLight.YELLOW
        elif traffic_light == 2:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN

    def __hough_stop_light_detector(self, img):
        gray = np.array(img)[:, :, 2]
        cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=5,
                                   param1=100, param2=15, minRadius=3, maxRadius=10)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            center_dots = []
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                center_dots.append(img[i[1], i[0]])

            median = np.median(center_dots, axis=0)
            is_red = median[0] < 10 and median[1] < 10 and median[2] > 200
            is_green = median[0] < 10 and median[1] > 200 and median[2] < 10
            if is_red:
                return TrafficLight.RED
            elif is_green:
                return TrafficLight.GREEN
            # TODO: orange case

        return TrafficLight.UNKNOWN
