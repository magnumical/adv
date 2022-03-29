
import numpy as np
import pandas as pd 
import os

#%%
base_image_dir = os.path.join('..', '/input/diabetic-retinopathy-resized/')
df = pd.read_csv(os.path.join( '../input/diabetic-retinopathy-resized/trainLabels_cropped.csv'))
df['path'] = df['image'].map(lambda x: os.path.join('../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped/','{}.jpeg'.format(x)))
df = df.drop(columns=['image']) 
df = df.sample(frac=1).reset_index(drop=True) 
df.head(10)

labels = df.loc[:,"level"]
one_hot = []
for i in labels:
    if i == 2:
        one_hot.append("Grade2")
    elif i== 3:
        one_hot.append("Worse")
    elif i== 4:
        one_hot.append("Worse")
    else:
        one_hot.append(False)
one_hot_series = pd.Series(one_hot)
one_hot_series.value_counts()

df['label'] = pd.Series(one_hot_series, index=df.index)
df.drop(['level'], axis=1,inplace = True)
df = df[df.label != False]
    

#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

    
data_gen = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              rotation_range=45, 
                              shear_range = 0.2,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              preprocessing_function=preprocess_input,
                              fill_mode = 'reflect',
                              zoom_range=0.2)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    df_gen = img_data_gen.flow_from_dataframe(in_df,
                                              x_col=path_col,
                                              y_col=y_col,
                                    **dflow_args)
    return df_gen


#%%

from sklearn.model_selection import train_test_split
train_df, test_valid_df = train_test_split(df, 
                                   test_size = 0.20, 
                                   random_state = 0,
                                   stratify = df['label'])
valid_df, test_df = train_test_split(test_valid_df, 
                                   test_size = 0.40, 
                                   random_state = 0,
                                   stratify = test_valid_df['label'])
train_df["label"].value_counts()

#%%
image_size = (224, 224) # image re-sizing target

train_gen = flow_from_dataframe(data_gen, train_df, path_col = 'path', y_col = 'label', 
                                target_size = image_size, 
                                color_mode = 'rgb',
                                class_mode='categorical',
                                batch_size = 64)

valid_gen = flow_from_dataframe(data_gen, valid_df, path_col = 'path', y_col = 'label', 
                                target_size = image_size, 
                                color_mode = 'rgb', 
                                class_mode='categorical',
                                batch_size = 128)

test_X, test_Y = next(flow_from_dataframe(data_gen, valid_df, path_col = 'path', y_col = 'label', 
                                          target_size = image_size, 
                                          color_mode = 'rgb', 
                                          class_mode='categorical',
                                          batch_size = 512))

#%%
import matplotlib.pyplot as plt 

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0])
    c_ax.axis('off')


#%%

from tensorflow.keras.applications.resnet import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
import tensorflow
tensorflow.keras.callbacks.History()



base_model = ResNet50(weights='imagenet', include_top = True,input_shape = (224,224,3))
outputs = base_model.layers[-2].output

fine_tune_layer = Dense(128)(outputs)
fine_tune_layer = Dropout(0.2)(fine_tune_layer) 
fine_tune_layer = Dense(2, activation='softmax')(fine_tune_layer)


model = Model(inputs=base_model.input, outputs=fine_tune_layer)

for layer in model.layers[:25]:
    layer.trainable = False

n_data= (train_gen.n, valid_gen.n)


# Compile Model
model.compile(optimizer = tensorflow.keras.optimizers.SGD(learning_rate=.001, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy','AUC'])




batch_size=128
n_epoch_beforeSaving= 50
history = tensorflow.keras.callbacks.History()

model.fit_generator(train_gen,
                    steps_per_epoch= n_data[0] // batch_size,
                    epochs=n_epoch_beforeSaving,
                    validation_data=valid_gen,
                    validation_steps= n_data[1] // batch_size,  callbacks=[history])



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






