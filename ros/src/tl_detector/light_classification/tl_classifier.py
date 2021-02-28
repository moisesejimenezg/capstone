from styx_msgs.msg import TrafficLight
import cv2

IMG_PREFIX = "/capstone/images/image_"
IMG_SUFFIX = ".png"
LABEL_FILE = "/capstone/images/labels.csv"


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.__image_id = 0
        self.__image_array = []
        self.__image_size = (800, 600)
        self.__file_object = open(LABEL_FILE, "w+")

    def save_image(self, image, label):
        if self.__image_id < 1000:
            cv2.imwrite(IMG_PREFIX + str(self.__image_id) + IMG_SUFFIX, image)
            self.__image_id += 1
            self.__file_object.write(str(label) + "\n")
        elif not self.__file_object.closed:
            self.__file_object.close()
            return True
        return False

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        return TrafficLight.UNKNOWN
