
import cv2
import keras
import random as rn
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import SVG
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, save_img
from tensorflow.python.keras.layers import Dense, Flatten,MaxPooling2D, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from keras.utils.vis_utils import plot_model
import os
#%%


benign_train_dir = '../input/skin-cancer-malignant-vs-benign/train/benign'
malign_train_dir = '../input/skin-cancer-malignant-vs-benign/train/malignant'

benign_val_dir = '../input/skin-cancer-malignant-vs-benign/test/benign'
malign_val_dir = '../input/skin-cancer-malignant-vs-benign/test/malignant'


name_malignant = os.listdir(malign_train_dir)
name_benign = os.listdir(benign_train_dir)

name_test_malignant= os.listdir(malign_val_dir)
name_test_benign= os.listdir(benign_val_dir)

new_malignant = []
new_benign = []
for i in name_malignant:
    new_malignant.append('../input/skin-cancer-malignant-vs-benign/train/malignant/'+i)
for j in name_benign:
    new_benign.append('../input/skin-cancer-malignant-vs-benign/train/benign/'+j)
    
    
new_test_malignant = []
new_test_benign = []    
for k in name_test_benign:
    new_test_benign.append('../input/skin-cancer-malignant-vs-benign/test/benign/'+k)
for l in name_test_malignant:
    new_test_malignant.append('../input/skin-cancer-malignant-vs-benign/test/benign/'+l)

import pandas as pd
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3= pd.DataFrame()
df4= pd.DataFrame()
df_test= pd.DataFrame()
df = pd.DataFrame()


df1['Skin_Type'] = new_malignant
df2['Skin_Type'] = new_benign
df1['Target'] = 'malignant'
df2['Target'] = 'benign'

df= pd.concat([df1, df2], axis = 0)

df3['Skin_Type'] = new_test_malignant
df4['Skin_Type'] = new_test_benign
df3['Target'] = 'malignant'
df4['Target'] = 'benign'

df_test= pd.concat([df3, df4], axis = 0)    
df = df.sample(frac=1).reset_index(drop=True)

#%%

img_height= 224
img_width =224

datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.1,
                                   validation_split=0.3)
train_datagen_flow = datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col='Skin_Type',
    y_col='Target',
    target_size=(img_height, img_width),
    batch_size=32,
    subset='training',
    shuffle = True,
    class_mode='binary'
)
valid_datagen_flow = datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col='Skin_Type',
    y_col='Target',
    target_size=(img_height, img_width),
    batch_size=32,
    subset='validation',
    class_mode = 'binary',
    shuffle = True
)

test_datagen_flow= datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col='Skin_Type',
    y_col='Target',
    target_size=(img_height, img_width),
    batch_size=32,
    subset='validation',
    class_mode = 'binary',
    shuffle = True
)

test_datagen_flow.class_indices


test_X, test_Y = next(test_datagen_flow)

#%%

from tensorflow.keras.applications.resnet import ResNet50

from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
import tensorflow

base_model = ResNet50(weights='imagenet', include_top = True,input_shape = (224,224,3))
outputs = base_model.layers[-2].output

# Finetune Layer
fine_tune_layer = Dense(128)(outputs)
fine_tune_layer = Dropout(0.2)(fine_tune_layer) #usually .2
fine_tune_layer = Dense(1, activation='softmax')(fine_tune_layer)

# Final Model
model = Model(inputs=base_model.input, outputs=fine_tune_layer)


# Compile Model
model.compile(optimizer = tensorflow.keras.optimizers.SGD(learning_rate=.001, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy','AUC'])




batch_size=128
n_epoch_beforeSaving= 50
history = tensorflow.keras.callbacks.History()

model.fit_generator(train_datagen_flow,
                    steps_per_epoch= train_datagen_flow.n // batch_size,
                    epochs=n_epoch_beforeSaving,
                    validation_data=valid_datagen_flow,
                    validation_steps= valid_datagen_flow.n // batch_size,  callbacks=[history])



#%%

from art.estimators.classification import KerasClassifier

classifier = KerasClassifier(model=model, use_logits=False, clip_values=(0, 1))

#%%

from art.attacks.evasion import ProjectedGradientDescent

PGD_images=[]
PGD_ACC=[]

PGD_ATT= ProjectedGradientDescent(classifier,eps=.3,max_iter=10)
PGD_data= PGD_ATT.generate(x=test_X)



predPGD= classifier.predict(PGD_data)
PGDaccuracy = np.sum(np.argmax(predPGD, axis=1) == np.argmax(test_Y, axis=1)) / len(test_Y)
PGD_ACC.append(PGDaccuracy*100)
print("Accuracy on adversarial examples (test set): {}%".format(PGDaccuracy * 100))
print('PGD is DONE!')


#%%
from art.attacks.evasion import AdversarialPatchNumpy

at_images=[]
at_ACC=[]

AT_ATT= AdversarialPatchNumpy(classifier, max_iter=5, scale_max=.6 )
AT_data= AT_ATT.generate(x=test_X, y=test_Y)



predat= classifier.predict(AT_data)
ataccuracy = np.sum(np.argmax(predat, axis=1) == np.argmax(test_Y, axis=1)) / len(test_Y)
at_ACC.append(ataccuracy*100)
print("Accuracy on adversarial examples (test set): {}%".format(ataccuracy * 100))
print('PA is DONE!')












