# Loading a NN for digit recognition, made using Google Colab here:
# https://colab.research.google.com/drive/1zDvKFQe4caMdBS8g4Olzacp2VlYjKqf5#scrollTo=BVwn-S3vUHIg


import keras
import matplotlib.pyplot as plt
import numpy as np

def makeModel():
    return keras.models.load_model('nn.h5')

model = makeModel()
model.summary()