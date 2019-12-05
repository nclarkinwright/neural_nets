# Nicholas Clarkin-Wright
# nc819094@wcupa.edu
# CSC 481 Neural Networks Assignment
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def get_model():
    # load wine dataset
    dataset = loadtxt('wine.csv', delimiter = ',')
    X = dataset[:,1:]
    y = dataset[:,0]

    # define the keras model
    model = Sequential()
    model.add(Dense(21, input_dim=13, activation='relu'))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=200, batch_size=10)

    return model

# Below used for testing
# model = get_model()
# evaluate the keras model
# dataset = loadtxt('wine.csv', delimiter = ',')
# X = dataset[:, 1:]
# y = dataset[:, 0]
# _, accuracy = model.evaluate(X, y)
# print('Accuracy: %.2f' %(accuracy*100))