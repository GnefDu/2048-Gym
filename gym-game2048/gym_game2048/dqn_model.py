import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np

class DQNModel:
  NUM_ACTIONS = 4

  def __init__(self, learning_rate=0.0001, size=(4,4)):
    # self.model = Sequential()
    self.size=size
    
    # model = self.build_model()

# this is a poor model, replace it.
  def build_model(self):
    # define convolutional layers
    input_shape = (self.size[0], self.size[1], 13)

    self.model = Sequential([
      Conv2D(128, kernel_size=(2, 2), activation='relu', input_shape=input_shape),
      BatchNormalization(),
      Conv2D(128, kernel_size=(2, 2), activation='relu'),
      BatchNormalization(),
      Flatten(),
      Dense(256, activation='relu'),
      Dense(128, activation='relu'),
      Dense(self.NUM_ACTIONS, activation='linear')  # Output Q-values for each action
      ])

    self.model.compile(loss='mse', optimizer='adam')
    return self.model
    # a different kind of model uses the following values:
    # 2 depths of convolutional layers with depths 128 for the first and 256 for the second

  def build_model2(self):
    input_shape = (self.size[0], self.size[1], 13)
    # I'm gonna change this and use 2 conv layers only with max pooling to see how it performs
    # I'll start with a simpler but working version of a model and increase complexity if necessary

    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu') (input_layer)
    conv1_1 = Conv2D(filters=128, kernel_size=(3, 2), activation='relu')(conv1)
    conv1_2 = Conv2D(filters=128, kernel_size=(2, 3), activation='relu')(conv1)

    conv2 = Conv2D(filters=256, kernel_size=(2, 2), activation='relu')(input_layer)
    conv2_1 = Conv2D(filters=256, kernel_size=(2, 1), activation='relu')(conv2)
    conv2_2 = Conv2D(filters=256, kernel_size=(1, 2), activation='relu')(conv2)

    flatten = [Flatten()(x) for x in [conv1_1, conv1_2, conv2_1, conv2_2]]
    print(flatten[0])
    dense1 = Dense(256, activation='relu')(flatten)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    output = Dense(self.NUM_ACTIONS, activation='linear')(dense2)
    self.model2 = Model(inputs=input_layer, outputs=output)

    self.model2.compile(loss='mse', optimizer=tf.keras.optimizers.ExponentialDecay(float(LEARNING_RATE, 50, 0.90, staircase=True)),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return self.model2

  def build_model3(self):
    input = Input(shape=(self.size[0], self.size[1], 13))
    model = Sequential([
      Input(shape=(self.size[0], self.size[1], 13) ),
      Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
      Conv2D(filters=256, kernel_size=(3, 2), activation='relu', padding='same'),
      Flatten(),
      Dense(256, activation='relu'),
      Dropout(0.2),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(self.NUM_ACTIONS, activation='linear')
    ])
    model.compile(loss='mse',metrics=['accuracy'], optimizer='adam') # try stochastic gradient descent as well for the optimizer
    return model

  def getLogEncoding(self, board):
    # how about the dynamic obstacles of log_2 1 = 0?
    # if it's 1, it'll have a log_encoding of 0. Which will be the same as empty tile.
    # Maybe we will want to turn it into a 2 instead. We'll see.
    new_board = np.where(board > 0, np.log2(board), 0)
    return new_board

  def getModel(self):
    return self.model3