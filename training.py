import cv2
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

for (i, imagePath) in enumerate(cropping_path):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)


    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # scale the input image pixels to the range [0, 1], then transform
    # the labels into vectors in the range [0, num_classes] -- this
    # generates a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    data = np.array(data) / 255.0
    labels = np_utils.to_categorical(labels, 2)


    pca = PCA(n_components=20)
    pca.fit_transform(data)


    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2, random_state=42)


    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


