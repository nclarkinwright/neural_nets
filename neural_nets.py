# Nicholas Clarkin-Wright
# nc819094@wcupa.edu
# CSC 481 Neural Networks Assignment
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def get_model():
    # load wine dataset
    dataset = loadtxt('wine.csv', delimiter = ',')
    X = dataset[:, 1:]
    y = dataset[:, 0]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)

    return model

model = get_model()

# evaluate the keras model
dataset = loadtxt('wine.csv', delimiter = ',')
X = dataset[:, 1:]
y = dataset[:, 0]
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' %(accuracy*100))