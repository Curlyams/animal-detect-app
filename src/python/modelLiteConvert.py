import tensorflow as tf


#Load the model

loaded_model = tf.keras.models.load_model("C:\\Users\\bcurl\\Desktop\\AnimalDetect\\models\\animal_detection_model_1.keras")

#convert the model to tf lite

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

with open("animal_detection_model_1.tflite", "wb") as f:
    f.write(tflite_model)
