import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
np.random.seed(202505)

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
print(train.columns)
Y_train = train["y"]

X_train = train.drop(labels = ["y"],axis = 1) 
X_train = X_train.set_index('id')
id_test = test['id']
print(id_test)
test = test.set_index('id')


print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)
X_train = X_train.values.reshape(-1,32,32,3)
test = test.values.reshape(-1,32,32,3)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

from keras import utils                                   # tools for creating one-hot encoding
num_classes = 4
Y_train = utils.to_categorical(Y_train, num_classes = num_classes)


train_index = np.random.choice(range(1,len(X_train)), size=len(X_train)//2)
# print(train_index)
val_index = []
for i in range(1,len(X_train)):
    if not i in train_index:
        val_index.append(i)


train_X = X_train[train_index]
valid_X = X_train[val_index]
train_label = Y_train[train_index]
valid_label = Y_train[val_index]



print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU

batch_size = 64
epochs = 20

plants_model = Sequential()
plants_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(32,32,3),padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))
plants_model.add(MaxPooling2D((2, 2),padding='same'))
plants_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))
plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
plants_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))                  
plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
plants_model.add(Flatten())
plants_model.add(Dense(128, activation='linear'))
plants_model.add(LeakyReLU(alpha=0.1))                  
plants_model.add(Dropout(0.3))
plants_model.add(Dense(num_classes, activation='softmax'))

plants_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

print(plants_model.summary())

plants_train = plants_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


predicted_classes = plants_model.predict(test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

# print(predicted_classes)

# print(test)
response = pd.DataFrame(
    data={
        'id': id_test,
        'y': predicted_classes
        }
)

response.to_csv('response.csv', index=False)