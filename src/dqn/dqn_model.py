import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

class dqn_model:
  NUM_ACTIONS = 4
  LEARNING_RATE = 0.0001
  def __init__(self, size=(4,4)):
    self.size=size

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

    self.model2.compile(loss='mse', optimizer=tf.keras.optimizers.ExponentialDecay(float(0.001, 50, 0.90, staircase=True)),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return self.model2

  def build_model3(self):
    input = Input(shape=(self.size[0], self.size[1], 1))
    model = Sequential([
      Input(shape=(self.size[0], self.size[1], 1) ),
      Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'),
      BatchNormalization(),
      Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(512, activation='relu'),
      Dropout(0.3),
      Dense(256, activation='relu'),
      Dropout(0.2),
      Dense(self.NUM_ACTIONS, activation='linear')
    ])
    model.compile(loss='mse',metrics=['accuracy'], optimizer=Adam(learning_rate=self.LEARNING_RATE)) # try stochastic gradient descent as well for the optimizer
    return model

  def getLogEncoding(self, board):
    new_board = np.where(board > 0, np.log2(board)+1, 0).astype(int)
    return new_board

  def getModel(self):
    return self.model3