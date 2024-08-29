from Resnet import *
import pandas as pd
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.optimizers import SGD,Adam

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
import h5py
import os
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbCallback




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

        
        
def train(train_gen, val_gen, epochs, input_shape, output_classes, model_path, checkpoint):

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=0.0001
    )
     
    optim = Adam(learning_rate=0.001)

    with tf.device('/gpu:0'):

        model = get_model(input_shape, output_classes)


    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    # his = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, lr_callback, WandbCallback()], shuffle=True)
    his = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, lr_callback], shuffle=True)
    # model.save(model_path)

    fig , ax= plt.subplots(1,2,figsize=(10,3))
    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(his.history[met])
        ax[i].plot(his.history['val_' + met])
        ax[i].set_title('model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train' , 'val'])
    plt.savefig('classification_report/training.png')


    
if __name__ == '__main__':

    ########################################## Config ############################################
    batch_size = 128
    num_epoch = 200
    segment_length = 1000
    num_classes = 7
    ##############################################################################################

    # wandb.init(
    # project='ECG CKD experiment', 
    # name='Resnet (modified)',)
    # config = wandb.config
    # config.batch_size = batch_size
    # config.num_epoch = num_epoch
    # config.segment_length = segment_length
    # config.num_classes = num_classes


    if os.path.exists('tmp/train_.h5') and os.path.exists('tmp/val_.h5') and os.path.exists('tmp/test_.h5'):
        print('loading datasets...')
        with h5py.File('tmp/train_.h5', 'r') as hf:
            train_x = hf['train_x'][:]
            train_y = hf['train_y'][:]
        with h5py.File('tmp/val_.h5', 'r') as hf:
            val_x = hf['val_x'][:]
            val_y = hf['val_y'][:]
        with h5py.File('tmp/test_.h5', 'r') as hf:
            test_x = hf['test_x'][:]
            test_y = hf['test_y'][:]
        print('datasets loaded')
    # else:


        # # read csv file
        # df = []
        # # read csv file
        # df.append(segment_data(load_dataset('normal'), 1000))
        # df.append(segment_data(load_dataset('stage_1'), 1000))
        # df.append(segment_data(load_dataset('stage_2'), 1000))
        # df.append(segment_data(load_dataset('stage_3'), 1000))
        # df.append(segment_data(load_dataset('stage_4'), 1000))
        # df.append(segment_data(load_dataset('stage_5'), 1000))
        # df.append(segment_data(load_dataset('end'), 1000))
        
        # train_data, test_data, val_data = [], [], []
        # train_labels, test_labels, val_labels = [], [], []


        # for i, d in enumerate(df):
        #     print(f'{i}: {d.shape}')
        #     label = to_categorical([i]*d.shape[0], num_classes=7)
        #     train_x, test_x, train_y, test_y = train_test_split(d, label, test_size=0.2, random_state=42)
        #     train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
            
        #     train_data.append(train_x)
        #     test_data.append(test_x)
        #     val_data.append(val_x)
        #     train_labels.append(train_y)
        #     test_labels.append(test_y)
        #     val_labels.append(val_y)
            
        # train_data = np.concatenate(train_data)
        # test_data = np.concatenate(test_data)
        # val_data = np.concatenate(val_data)
        # train_labels = np.concatenate(train_labels)
        # test_labels = np.concatenate(test_labels)
        # val_labels = np.concatenate(val_labels)
        
            
        # # save datasets
        # with h5py.File('tmp/train_.h5', 'w') as hf:
        #     hf.create_dataset("train_x",  data=train_data)
        #     hf.create_dataset("train_y",  data=train_labels)

        # with h5py.File('tmp/val_.h5', 'w') as hf:
        #     hf.create_dataset("val_x",  data=val_data)
        #     hf.create_dataset("val_y",  data=val_labels)

        # with h5py.File('tmp/test_.h5', 'w') as hf:
        #     hf.create_dataset("test_x",  data=test_data)
        #     hf.create_dataset("test_y",  data=test_labels)



    checkpoint_path = "checkpoints/model_classification.keras"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                monitor='val_accuracy',
                                verbose=0,
                                save_best_only=True,
                                mode='max')

    # check nan and inf
    # print(np.isnan(train_x).any())
    # print(np.isnan(val_x).any())

    train_x = np.nan_to_num(train_x)
    val_x = np.nan_to_num(val_x)
    train_x = np.transpose(train_x, (0, 2, 1))
    val_x = np.transpose(val_x, (0, 2, 1))

    #####################################################################################################
    # Experiment : 
    # The input shape is now (n, 12, 1000). Segment into (n*k, 12, 1000/k)
    

    # print('original: ', train_x.shape, train_y.shape)
    # new_train_x = segment_array(train_x, segment_length=segment_length)
    # new_val_x = segment_array(val_x, segment_length=segment_length)

    # expanded_train_y = [item for item in train_y for _ in range(int(1000/segment_length))]
    # expanded_train_y = np.array(expanded_train_y)
    # expanded_val_y = [item for item in val_y for _ in range(int(1000/segment_length))]
    # expanded_val_y = np.array(expanded_val_y)
    # print('Segmented: ', new_train_x.shape, expanded_train_y.shape)




    train_gen = DataGenerator(train_x, train_y, batch_size)
    val_gen = DataGenerator(val_x, val_y, batch_size)


    # train_gen = DataGenerator(new_train_x, expanded_train_y, batch_size)
    # val_gen = DataGenerator(new_val_x, expanded_val_y, batch_size)

    print('segment_length:', segment_length)

    
    train(train_gen, val_gen, num_epoch, (12, segment_length), 7, 'model', checkpoint)