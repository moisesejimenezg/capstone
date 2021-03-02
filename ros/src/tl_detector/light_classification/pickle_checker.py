from labeler import Labeler
import numpy as np
import cv2

labeler = Labeler("rb")

data = labeler.load()

# Resize data
feature_shape = [600, 800, 3]
data["features"] = data["features"].reshape([-1] + feature_shape)

cv2.imwrite("test.png", data["features"][0])

print("Features: " + str(len(data["features"])))
print("Labels: " + str(len(data["labels"])))
