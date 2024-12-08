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

  
  def build_model3(self):
    # model = Sequential([
    #   Input(shape=(self.size[0], self.size[1], 1) ),
    #   Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'),
    #   BatchNormalization(),
    #   Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    #   MaxPooling2D(pool_size=(2, 2)),
    #   Flatten(),
    #   Dense(128, activation='relu'),
    #   Dropout(0.3),
    #   Dense(256, activation='relu'),
    #   Dropout(0.2),
    #   Dense(self.NUM_ACTIONS, activation='linear')
    # ])


    # Define the model
    model = Sequential([
        Flatten(input_shape=(4, 4, 1)),  # Flatten the 4x4x16 state
        Dense(256, activation='relu'),    # First dense layer
        Dense(128, activation='relu'),    # Optional second dense layer
        Dense(64, activation='relu'),     # Optional third dense layer
        Dense(4, activation='linear')     # Output layer for Q-values
    ])
    model.compile(loss='mse',metrics=['accuracy'], optimizer=Adam(learning_rate=self.LEARNING_RATE)) 
    return model

  def getLogEncoding(self, board):
    new_board = np.where(board > 0, np.log2(board)+1, 0).astype(int)
    return new_board

  def getModel(self):
    return self.model3