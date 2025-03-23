# pip install np_utils

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow import ImageDataGenerator
from tensorflow import keras
from sklearn.datasets import load_files
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img

# """## 2. Importing the NEU Metal Surface Defect Dataset"""

# from google.colab import drive
# drive.mount('/content/drive')

# import os
# print(os.listdir('/content/drive/MyDrive/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master'))

# print(os.listdir('/content/drive/MyDrive/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master/NEU Metal Surface Defects Data'))

# !cp -r "/content/drive/MyDrive/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master/NEU Metal Surface Defects Data" "/content/neu-metal-surface-defects-data"

# import os
# print(os.listdir('/content/target_folder_name'))
# /content/neu-metal-surface-defects-data

train_dir = './Data/train'
val_dir = './Data/valid'
test_dir='./Data/test'
# print("Path: ",os.listdir("/content/neu-metal-surface-defects-data/NEU Metal Surface Defects Data"))
# print("Train: ",os.listdir("/content/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/train"))
# print("Test: ",os.listdir("/content/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test"))
# print("Validation: ",os.listdir("/content/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/valid"))

# /content/sample_data

# from google.colab import drive
# drive.mount('/content/drive')
# """#### For each Class, the Training Data includes 276 Images, Validation & Test sets have 12 images each."""

# print("Inclusion Defect")
# print("Training Images:",len(os.listdir(train_dir+'/'+'Inclusion')))
# print("Testing Images:",len(os.listdir(test_dir+'/'+'Inclusion')))
# print("Validation Images:",len(os.listdir(val_dir+'/'+'Inclusion')))

# """## 3. Data Pre-processing"""

# Rescaling all Images by 1./255
# Tạo một ImageDataGenerator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Training images are put in batches of 10
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

# Validation images are put in batches of 10
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

# """#### Setting upper Limit of Max 98% training accuracy"""

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

# """## 4. Defining the CNN Architecture"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# tf.keras.utils.plot_model(
#     model,
#     to_file='cnn_architecture.png',
#     show_shapes=True)

"""## 5. Training the Defined CNN Model"""

callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 32,
        epochs=2,
        validation_data=validation_generator,
        callbacks=[callbacks],
        verbose=1, shuffle=True)

# """## 6. Analysing the Accuracy & the Loss Curves"""

# sns.set_style("whitegrid")
# plt.subplot(211)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# sns.set_style("whitegrid")
# plt.subplot(212)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

"""## 7. Test Result visualization"""

# Loading file names & their respective target labels into numpy array
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
x_test, y_test,target_labels = load_dataset(test_dir)
no_of_classes = len(np.unique(y_test))
no_of_classes

#y_test = np_utils.to_categorical(y_test,no_of_classes)
y_test = to_categorical(y_test,no_of_classes)

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_test = np.array(convert_image_to_array(x_test))
# print('Test set shape : ',x_test.shape)

x_test = x_test.astype('float32')/255

# """## Results of Hot-Rolled Steel Strips Surface Defect Detection"""

# Plotting Random Sample of test images, their predicted labels, and ground truth
y_pred = model.predict(x_test)
fig = plt.figure(figsize=(10, 10))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))