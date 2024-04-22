import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import matplotlib.pyplot as plt
x_train=np.loadtxt('C:/Users/Lenovo/Downloads/Data Set/data set/input.csv',delimiter=',')
y_train=np.loadtxt('C:/Users/Lenovo/Downloads/Data Set/data set/labels.csv',delimiter=',')
x_test=np.loadtxt('C:/Users/Lenovo/Downloads/Data Set/data set/input_test.csv',delimiter=',')
y_test=np.loadtxt('C:/Users/Lenovo/Downloads/Data Set/data set/labels_test.csv',delimiter=',')
x_train=x_train.reshape(len(x_train),100,100,3)
y_train=y_train.reshape(len(y_train),1)
x_test=x_test.reshape(len(x_test),100,100,3)
y_test=y_test.reshape(len(y_test),1)
x_train=x_train/255.0
x_test=x_test/255.0
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train[1,:]
idx=random.randint(0,len(x_train))
plt.imshow(x_train[idx,:])
plt.show()
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=64)
model.evaluate(x_test,y_test)
idx2=random.randint(0,len(y_test))
plt.imshow(x_test[idx2,:])
plt.show()
y_pred=model.predict(x_test[idx2,:].reshape(1,100,100,3))
y_pred=y_pred>0.5
if(y_pred==0):
    pred='dog'
else:
    pred='cat'
print('Our model says it is a :',pred)    


