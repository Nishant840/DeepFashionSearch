import pickle
import numpy as np

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))

# print(np.array(feature_list).shape) # (44441,2048)

filenames = pickle.load(open('filenames.pkl','rb'))

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# ResNet model
model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/sneaker.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalised_result = result / norm(result)

# Using KNN algorithm to predict closest five recommendation

from sklearn.neighbors import NearestNeighbors

neighbours = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbours.fit(feature_list)

distances, indices = neighbours.kneighbors([normalised_result])

# print(indices)
for file in indices[0]:
    print(filenames[file])

# import cv2

# for file in indices[0]:
#     temp_img = cv2.imread(filenames[file])
#     cv2.imshow('output',cv2.resize(temp_img,(512,512)))
