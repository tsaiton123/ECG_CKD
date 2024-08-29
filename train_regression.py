from test_model import get_model
import pandas as pd
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import resample
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
import h5py
import os
import matplotlib.pyplot as plt
# import wandb
# from wandb.integration.keras import WandbCallback
from ordinal import ordinal_entropy

tf.config.run_functions_eagerly(True)


class PredictionCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Predict the first batch of the validation set
        val_predict = (np.asarray(self.model.predict(val_x[:32]))).round()
        val_targ = val_y[:32]
        print(f'\nValidation predictions for epoch {epoch}: {val_predict}')
        print(f'Validation actuals for epoch {epoch}: {val_targ}')

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)


def self_defined_loss(y_true, y_pred):
    # MSE loss + ordinal loss
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    ordinal_loss = ordinal_entropy(y_pred, y_true)
    # ordinal_loss = 0
    return mse_loss + ordinal_loss
        
        
def train(train_gen, val_gen, epochs, input_shape, output_classes, model_path, checkpoint, class_weight=None):

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=0.0001
    )
     
    optim = Adam(learning_rate=0.001)

    with tf.device('/gpu:0'):
        model = get_model(input_shape, output_classes, last_layer='linear')
    # model.compile(optimizer=optim, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.compile(optimizer=optim, loss=self_defined_loss, metrics=['mean_squared_error']) 
    print(model.summary()) 
    # his = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, lr_callback, WandbCallback()], shuffle=True, class_weight=class_weight)
    his = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, lr_callback], shuffle=True)

    fig , ax= plt.subplots(1,2,figsize=(10,3))
    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(his.history[met])
        ax[i].plot(his.history['val_' + met])
        ax[i].set_title('model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train' , 'val'])
    plt.savefig('regression_report/training.png')

    
if __name__ == '__main__':

    ########################################## Config ############################################
    batch_size = 128
    num_epoch = 500
    segment_length = 1000
    num_classes = 7
    ##############################################################################################



    if os.path.exists('tmp/train.h5') and os.path.exists('tmp/val.h5'):
        print('loading datasets...')
        with h5py.File('tmp/train.h5', 'r') as hf:
            train_x = hf['train_x'][:]
            train_y = hf['train_y'][:]
        with h5py.File('tmp/val.h5', 'r') as hf:
            val_x = hf['val_x'][:]
            val_y = hf['val_y'][:]
        print('datasets loaded')




    checkpoint_path = "checkpoints/model_regression.keras"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
    

    
    train_x = np.nan_to_num(train_x)
    val_x = np.nan_to_num(val_x)

    train_x = np.transpose(train_x, (0, 2, 1))
    val_x = np.transpose(val_x, (0, 2, 1))


    #####################################################################################################
    # Experiment : 
    # The input shape is now (n, 12, 1000). Segment into (n*k, 12, 1000/k)
    

    print('original: ', train_x.shape, train_y.shape)
    new_train_x = segment_array(train_x, segment_length=segment_length)
    new_val_x = segment_array(val_x, segment_length=segment_length)

    expanded_train_y = [item for item in train_y for _ in range(int(1000/segment_length))]
    expanded_train_y = np.array(expanded_train_y)
    expanded_val_y = [item for item in val_y for _ in range(int(1000/segment_length))]
    expanded_val_y = np.array(expanded_val_y)
    print('Segmented: ', new_train_x.shape, expanded_train_y.shape)
    #####################################################################################################


    train_gen = DataGenerator(new_train_x, expanded_train_y, batch_size)
    val_gen = DataGenerator(new_val_x, expanded_val_y, batch_size)

    
    
    train(train_gen, val_gen, num_epoch,(12, segment_length), num_classes, 'model', checkpoint, class_weight=None)
