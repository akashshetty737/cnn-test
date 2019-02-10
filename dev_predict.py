import scipy
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
x = imread('data/dev/test/0/9512.png', mode='L')
#compute a bit-wise inversion so black becomes white and vice versa
x = np.invert(x)
#make it the right size
x = imresize(x,(32,32))
#convert to a 4D tensor to feed into our model
x = x.reshape(1,32,32,1)
x = x.astype('float32')
x /= 255

#perform the prediction
import keras
from keras.models import load_model
model = load_model('data/dev_modelV1.h5')
out = model.predict(x)
print(np.argmax(out))