from styx_msgs.msg import TrafficLight
from labeler import Labeler
import cv2


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.__labeler = Labeler()

    def save_image(self, image, label):
        self.__labeler.label_image(image, label)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        return TrafficLight.UNKNOWN
