# will genereate a labeled dataset with
# images from the malware and notmalware 
# directories. 

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

data_dir = "./images"
catagories = ["notmalware", "malware"]
training_data = []

# iterate over the training data
# and create a dataset from the images
def create_training_data():
    for category in catagories:
        path = os.path.join(data_dir, category)

        class_num = catagories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

# shuffle the data
random.shuffle(training_data)

# X will be the image
# y will be the label
X = []
y = []

# add the images and labels to the X and Y lists
for features, label in training_data:
    X.append(features)
    y.append(label)
    print(label)

# convert X into a numpy array
X = np.array(X).reshape(-1, 256, 256 ,1)

# save the dataset
pickle_out = open("./data/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("./data/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# # load data
# # pickle_in = open("X.pickle", "rb")
# # X = pickle.load(pickle_in)
