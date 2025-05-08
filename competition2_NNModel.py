import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
import json
warnings.filterwarnings('ignore')
np.random.seed(202505)

# Read data from files
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
# print(train.columns)

# Split Training into X and y
Y_train = train["y"]
X_train = train.drop(labels = ["y"],axis = 1) 
X_train = X_train.set_index('id')
id_test = test['id']
# print(id_test)
test = test.set_index('id')
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# Reshape into 3D array
X_train = X_train.values.reshape(-1,32,32,3)
test = test.values.reshape(-1,32,32,3)
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)

from keras import utils                                   # tools for creating one-hot encoding
# Turn category into seperate columns
num_classes = 4
Y_train = utils.to_categorical(Y_train, num_classes = num_classes)

# Split into training and validation data
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

# Setup hyper parameters
batch_size = 200
epochs = 100

available_functions = ['linear', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign','tanh','selu','elu','exponential']
function_count = 3

best_accuracy = 0
best_activations = []
best_node_count = []
best_kernel = []
for i in range(1,100):
    # Set Hyper Parameters
    activations = np.random.choice(available_functions, size=function_count)
    node_count = []
    kernel_list = []
    for _ in range(function_count):
        node_count.append(random.randint(40, 500))
        k = 2*random.randint(0, 3)+1
        kernel_list.append((k,k))
    activations = np.append(activations, "tanh")
    node_count.append(70)
    kernel_list.append((1,1))
    
    
        
    # Build model
    plants_model = Sequential()
    plants_model.add(Conv2D(node_count[0], kernel_size=kernel_list[0],activation=activations[0],input_shape=(32,32,3),padding='same'))
    plants_model.add(LeakyReLU(alpha=0.1))
    plants_model.add(MaxPooling2D((2, 2),padding='same'))
    plants_model.add(Conv2D(node_count[1], kernel_list[1], activation=activations[1],padding='same'))
    plants_model.add(LeakyReLU(alpha=0.1))
    plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    plants_model.add(Conv2D(node_count[2], kernel_list[2], activation=activations[2],padding='same'))
    plants_model.add(LeakyReLU(alpha=0.1))                  
    plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    plants_model.add(Flatten())
    plants_model.add(Dense(node_count[3], activation=activations[3]))
    plants_model.add(LeakyReLU(alpha=0.1))                  
    plants_model.add(Dropout(0.3))
    plants_model.add(Dense(num_classes, activation='softmax'))

    plants_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    # print(plants_model.summary())
    # Train using training data and validate with validation data
    plants_train = plants_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
    # print(plants_train.history.keys())
    # print(plants_train.history['accuracy'])
    
    # Check if current model is the best and store parameters if it is
    if  plants_train.history['val_accuracy'][len(plants_train.history['val_accuracy'])-1] > best_accuracy:
        best_accuracy = plants_train.history['accuracy'][len(plants_train.history['val_accuracy'])-1]
        best_activations = activations
        best_node_count = node_count
        best_kernel = kernel_list
        
        current_json = {}
        try:
            with open('results_description.json', 'r') as f:
                current_json = json.load(f)
        except FileNotFoundError:
            print("File doesn't exist")
        
        with open('results_description.json', 'w') as f:
            description_dict = {
                "val_accuracy": best_accuracy,
                "summary": str(plants_model.summary()),
                "activation_functions": list(activations),
                "nodes": list(node_count),
                "kernels": list(kernel_list)
            }
            current_json[str(i)] = description_dict
            json.dump(current_json,f, indent=4)
            # print(plants_model.summary())
            # f.write(str(plants_model.summary()))
            # print("Activation functions:", activations)
            # f.write(' '.join(["Activation functions:", str(activations)]))
            # f.write('\n')
            # print("Nodes:",node_count)
            # f.write(' '.join(["Nodes:", str(node_count)]))
            # f.write('\n')
            # print("Kernels:",kernel_list)
            # f.write(' '.join(["Kernels:", str(kernel_list)]))
            # f.write('\n')
    if (1-best_accuracy) < 0.01:
        break

activations = best_activations
node_count = best_node_count
kernel_list = best_kernel
print(best_accuracy)

# Run model again with best parameters
plants_model = Sequential()
plants_model.add(Conv2D(node_count[0], kernel_size=kernel_list[0],activation=activations[0],input_shape=(32,32,3),padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))
plants_model.add(MaxPooling2D((2, 2),padding='same'))
plants_model.add(Conv2D(node_count[1], kernel_list[1], activation=activations[1],padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))
plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
plants_model.add(Conv2D(node_count[2], kernel_list[2], activation=activations[2],padding='same'))
plants_model.add(LeakyReLU(alpha=0.1))                  
plants_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
plants_model.add(Flatten())
plants_model.add(Dense(node_count[3], activation=activations[3]))
plants_model.add(LeakyReLU(alpha=0.1))                  
plants_model.add(Dropout(0.3))
plants_model.add(Dense(num_classes, activation='softmax'))

plants_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# Print results of the best model
with open('results_description.txt', 'w') as f:
    print(plants_model.summary())
    f.write(str(plants_model.summary()))
    print("Activation functions:", activations)
    f.write(' '.join(["Activation functions:", str(activations)]))
    f.write('\n')
    print("Nodes:",node_count)
    f.write(' '.join(["Nodes:", str(node_count)]))
    f.write('\n')
    print("Kernels:",kernel_list)
    f.write(' '.join(["Kernels:", str(kernel_list)]))
    f.write('\n')

plants_train = plants_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# Predict the classification of the test results
predicted_classes = plants_model.predict(test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

# print(predicted_classes)

# print(test)
# Create the output table and store as a csv
response = pd.DataFrame(
    data={
        'id': id_test,
        'y': predicted_classes
        }
)

response.to_csv('response.csv', index=False)