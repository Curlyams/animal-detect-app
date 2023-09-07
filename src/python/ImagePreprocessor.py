import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer



class ImagePreprocessor:
    def __init__(self, root_dir, desired_size=(128, 128)):
        self.root_dir = root_dir
        self.desired_size = desired_size
        ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

        self.datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        
    def _load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, self.desired_size)
        return image
    
    def process_images(self):
        images = []
        labels = []

        for label in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, label)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    filepath = os.path.join(class_path, filename)
                    if filepath.endswith('.jpg'):
                        image = self._load_image(filepath)
                        images.append(image)
                        labels.append(label)
        
        # Convert images and labels to arrays
        images = np.array(images) / 255.0
        labels = np.array(labels)

        # One-hot encode labels
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        return images, labels

    def get_train_val_test(self, test_size=0.2, val_size=0.1):
        images, labels = self.process_images()
        trainX, testX, trainY, testY = train_test_split(images, labels, test_size=test_size)
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=val_size / (1 - test_size))
        return (trainX, trainY), (valX, valY), (testX, testY)
