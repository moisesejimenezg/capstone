import cv2
import numpy as np
from sklearn.utils import shuffle
from tl_classifier import TLClassifier

from labeler import Labeler


def load_data(tmp_data):
    images = []
    labels = []
    for pair in tmp_data:
        images.append(pair["features"])
        labels.append(pair["labels"])
    return np.array(images), np.array(labels)


def generator(samples, batch_sz=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_sz):
            batch_samples = samples[offset: offset + batch_sz]

            tmp_features, tmp_labels = load_data(batch_samples)
            yield shuffle(tmp_features, tmp_labels)


# ======== MAIN ========
feature_shape = [150, 200, 3]

labeler = Labeler("rb")
data = labeler.load()
data["features"] = data["features"].reshape([-1] + feature_shape)

print("Labels: " + str(data["labels"].size))
print("Features: " + str(data["features"].size))
print("Shape: " + str(data["features"].shape))

for img, label in zip(data["features"], data["labels"]):
    classifier = TLClassifier()
    result = classifier.get_classification(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        print(str(is_red) + "=" + str(label))

    cv2.imshow('detected circles', img)
    cv2.waitKey(1)
    # assert int(label) == int(result)

