from labeler import Labeler
import numpy as np
import cv2

labeler = Labeler("rb")

data = labeler.load()
cv2.imwrite("test.png", data["features"][0])

print("Features: " + str(len(data["features"])))
print("Labels: " + str(len(data["labels"])))
