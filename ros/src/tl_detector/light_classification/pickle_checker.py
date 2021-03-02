from labeler import Labeler
import numpy as np

labeler = Labeler("rb")

data = labeler.load()

print("Features: " + str(len(data["features"])))
print("Labels: " + str(len(data["labels"])))
