import os

import numpy as np
import cv2
from keras.models import load_model
from styx_msgs.msg import TrafficLight
from labeler import Labeler

MODEL_LOCATION = "/capstone/ros/src/tl_detector/light_classification/model.h5"


def resize_image(image):
    new_height = int(image.shape[0] * 0.25)
    new_width = int(image.shape[1] * 0.25)
    dim = (new_width, new_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.__labeler = Labeler("wb")
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
        if traffic_light == 0:
            return TrafficLight.RED
        elif traffic_light == 1:
            return TrafficLight.YELLOW
        elif traffic_light == 2:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
