# libraries
import numpy as np
import pandas as pd
import os 
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Dense, Flatten, Dropout, SeparableConv1D
import matplotlib.pyplot as plt
import librosa

# file addresses 

addresses=[]
label=[]
for i in os.listdir(path of the file' /recordings'):
  addresses.append(path of the file' /recordings/'+i)
  label.append(i[0])

print(len(addresses),len(label))

# feature exteraction
data=[]
for i in addresses:
  sound, sample_rate = librosa.load(i,sr=4800)
  stft = np.abs(librosa.stft(sound))  
  mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=128),axis=1)
  data.append(mfccs)
  
# convert list to numpy array
data=np.asarray(data)
data=data.reshape((2500,128,1))

# make one hot labels
label=keras.utils.to_categorical(label,num_classes=10)
print(label.shape,data.shape)

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data, label, test_size=0.2, random_state=42)


#1D CNN model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(128, 1)))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2)) 
model.add(Dropout(0.5))

model.add(SeparableConv1D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2)) 
model.add(Dropout(0.5))

model.add(SeparableConv1D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2)) 
model.add(Dropout(0.5))

model.add(SeparableConv1D(512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2)) 

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(1024, activation='relu'))   
model.add(Dense(10, activation='softmax'))
model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train ,validation_data=(x_test,y_test), epochs=100, batch_size=8,verbose=1)

# ploting the accuracy and loss function

acc=history.history['acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
val_acc=history.history['val_acc']

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
