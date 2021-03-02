from labeler import Labeler
import numpy as np
import cv2

labeler = Labeler("rb")

data = labeler.load()

data["features"] = data["features"].reshape(-1, 150, 200, 3)
cv2.imwrite("test.png", data["features"][1000])

print("Features: " + str(len(data["features"])))
print("Labels: " + str(len(data["labels"])))
