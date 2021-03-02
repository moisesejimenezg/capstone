import cv2
import pickle
import numpy as np
import cv2

PICKLE_FILE = "/capstone/images/pickle_rick.p"
PAIR_PER_PICKLE = 50
MAX_PICKLE_N = 50


def resize_image(image):
    new_height = int(image.shape[0] * 0.25)
    new_width = int(image.shape[1] * 0.25)
    dim = (new_width, new_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class Labeler:
    def __init__(self, mode):
        self.__pickles = 0
        self.__pickle_file = open(PICKLE_FILE, mode)
        self.__header = False
        self.__reset()

    def __reset(self):
        self.__features = []
        self.__labels = np.array([])

    def label_image(self, image, label):
        if not self.__header:
            pickle.dump(MAX_PICKLE_N, self.__pickle_file)
            self.__header = True
        print(
            "Pickle #: " + str(self.__pickles) + " image #: " + str(self.__labels.size)
        )
        if self.__pickles < MAX_PICKLE_N:
            if self.__labels.size < PAIR_PER_PICKLE:
                image = resize_image(image)
                array = np.array(image)
                self.__features.append(image)
                self.__labels = np.append(self.__labels, label)
            else:
                data = {"features": np.array(self.__features), "labels": self.__labels}
                pickle.dump(data, self.__pickle_file)
                self.__reset()
                self.__pickles += 1
            return False
        self.__pickle_file.close()
        return True

    def __load(self):
        if self.__pickles == 0:
            self.__pickles = pickle.load(self.__pickle_file)
        if self.__pickles > 0:
            self.__pickles -= 1
            return pickle.load(self.__pickle_file)
        return None

    def load(self):
        data = self.__load()
        while self.__pickles > 0:
            new_pickle = self.__load()
            data["features"] = np.append(data["features"], new_pickle["features"])
            data["labels"] = np.append(data["labels"], new_pickle["labels"])
        return data
