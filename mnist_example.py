import os

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

# Boto3 S3 library
import boto3

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

np.save('/tmp/predictions.npy', predictions)

# Upload result to Minio
s3 = boto3.resource('s3', 
                    endpoint_url=os.environ.get('MINIO_ENDPOINT'), 
                    aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('MINIO_SECRET_ACCESS_KEY'))

data = open('/tmp/predictions.npy', 'rb')
s3.Bucket('default').put_object(Key='predictions.npy', Body=data)