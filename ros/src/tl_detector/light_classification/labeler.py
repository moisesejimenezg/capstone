import cv2

IMG_PREFIX = "/capstone/images/image_"
IMG_SUFFIX = ".png"
LABEL_FILE = "/capstone/images/labels.csv"


class Labeler:
    def __init__(self):
        self.__image_id = 0
        self.__image_array = []
        self.__image_size = (800, 600)
        self.__file_object = open(LABEL_FILE, "w+")

    def label_image(self, image, label):
        if self.__image_id < 1000:
            cv2.imwrite(IMG_PREFIX + str(self.__image_id) + IMG_SUFFIX, image)
            self.__image_id += 1
            self.__file_object.write(str(label) + "\n")
        elif not self.__file_object.closed:
            self.__file_object.close()
            return True
        return False
