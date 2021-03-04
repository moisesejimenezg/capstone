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

for image, label in zip(data["features"], data["labels"]):
    classifier = TLClassifier()
    result = classifier.get_classification(image)
    assert int(label) == int(result)

