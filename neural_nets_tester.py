import neural_nets
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

model = get_model()

# evaluate the keras model
dataset = loadtxt('wine.csv', delimiter = ',')
X = dataset[:, 1:]
y = dataset[:, 0]
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' %(accuracy*100))