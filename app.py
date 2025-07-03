import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm

model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  
])

# print(model.summary())

def extract_feature(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalised_result = result / norm(result)
    
    return normalised_result

import os
# print(os.listdir('images'))

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# print(len(filenames))

feature_list = []

from tqdm import tqdm

for file in tqdm(filenames):
    feature_list.append(extract_feature(file,model))

import pickle
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))