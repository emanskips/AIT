import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp
import sklearn.metrics as sk
from random import randint

#Normalize the dataset, split it into training and validation sets, and apply data augmentation techniques
fm = tf.keras.datasets.fashion_mnist
(trX, trY), (teX, teY) = fm.load_data()

teX = teX/255
trX = trX/255

#reshape input to flatten images
teX = teX.reshape(-1, 28*28)
trX = trX.reshape(-1, 28*28)

#trY.shape
#teY.shape
#trX.shape
#teY.shape

labels = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

#mp.figure()
#mp.imshow(trX[9990])
#mp.colorbar()

#build and train the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(trX, trY, epochs=20)

#display the confusion matrix and the accuracy
prediction = np.argmax(model.predict(teX), axis=-1)
cm = sk.confusion_matrix(teY, prediction)
accuracy = sk.accuracy_score(teY, prediction)
#loss = sk.log_loss(teY, prediction)
print(f"Accuracy: \n {accuracy}")
#print(f"Loss: \n {loss}")
print(f"Confusion Maxtrix: \n {cm}")

#randomly select 10 images and have the model predict, then compare to the actual
count = 0
while count < 10:
    i = randint(0, 9999)
    mp.figure()
    mp.imshow(teX[i].reshape(28, 28))
    mp.colorbar()
    mp.show()
    print(f"Predicted: {labels[prediction[i]]}")
    print(f"Actual: {labels[teY[i]]}")
    count += 1