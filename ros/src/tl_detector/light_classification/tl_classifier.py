import os

import numpy as np
from keras.models import load_model
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.__model = None

        if os.path.isfile("model.h5"):
            self.__model = load_model("model.h5")
        else:
            print("Could not init TLClassifier!")

    @property
    def model(self):
        assert self.__model is not None
        return self.__model

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

        image_array = np.asarray(image)
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

