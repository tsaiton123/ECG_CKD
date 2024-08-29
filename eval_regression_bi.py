import pandas as pd
import numpy as np
import h5py
import os
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

tf.config.run_functions_eagerly(True)

def parse_args():
    """
    mode : 
    1. any (binary classification)
    2. mild (binary classification)
    3. severe (binary classification)
    4. multi (multi-class classification)
    """
    parser = argparse.ArgumentParser(description='Evaluate the model on the test set.')
    parser.add_argument('--mode', type=str, default='any', help='Mode of classification')
    return parser.parse_args()


def compute_gradcam(model, input_data, layer_name, pred_index=None):
    """
    Compute Grad-CAM for a given input and a specific layer.

    Args:
        model: The trained model.
        input_data: The input data for which to compute the Grad-CAM.
        layer_name: The name of the layer for which to compute the Grad-CAM.
        pred_index: The index of the prediction for which to compute the Grad-CAM.

    Returns:
        gradcam: The Grad-CAM heatmap.
    """
    # Get the model's output for the given input
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        print('conv_outputs:', conv_outputs.shape)   #
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        loss = predictions[0]

    # Compute the gradients of the top predicted class with respect to the output feature map
    grads = tape.gradient(loss, conv_outputs)


    # Pool the gradients across the feature map
    pooled_grads = tf.reduce_mean(grads, axis=1)


    # Multiply each channel by the corresponding pooled gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    
    pooled_grads = tf.expand_dims(pooled_grads, axis=0)
    conv_outputs = conv_outputs * pooled_grads

    return conv_outputs


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = [cls for i, cls in enumerate(classes) if not np.all(np.isnan(cm[i]))]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)




if __name__ == '__main__':

    args = parse_args()
    mode = args.mode

    # load dataset
    print('loading datasets...')

    with h5py.File('tmp/test.h5', 'r') as hf:
        test_x = hf['test_x'][:]
        test_y = hf['test_y'][:]
    print('datasets loaded')

    # load model
    checkpoint_path = 'checkpoints/model_regression.keras'
    model = tf.keras.models.load_model(checkpoint_path, compile = False)


    test_x = np.transpose(test_x, (0, 2, 1))


    #####################################################################################################
    # Experiment : 
    # The input shape is now (n, 12, 1000). Segment into (n*k, 12, 1000/k)
    # segment_length = 500
    

    # print('original: ', test_x.shape, test_y.shape)
    # test_x = segment_array(test_x, segment_length=segment_length)


    # test_y = [item for item in test_y for _ in range(int(1000/segment_length))]
    # test_y = np.array(test_y)

    # print('Segmented: ', test_x.shape, test_y.shape)
    #####################################################################################################

    y_pred = model.predict(test_x).flatten()
    # round predictions
    test_y = test_y.astype(int)
    #######################
    # test_y = np.argmax(test_y, axis=1)
    #######################
    y_pred = np.nan_to_num(y_pred)
    y_pred = np.round(y_pred).astype(int)
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = 0



    # Compute Grad-CAM

    # index = 50
    # gradcam = compute_gradcam(model, test_x[index:index+1], 'conv2d_1', pred_index=0)


    # ticks = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # # plot gradcam for each lead

    # for i in range(12):
    #     plt.figure(figsize=(10, 10))
    #     plt.plot(test_x[index][i] * 10, color='black', alpha=0.5)
    #     # normalize gradcam[i] from 0 to 10
    #     tmp = (gradcam[i] - np.min(gradcam[i])) / (np.max(gradcam[i]) - np.min(gradcam[i])) * 10
    #     plt.imshow([tmp], cmap='hot', alpha=0.5, extent=(0, 1000, 0, 10), aspect='auto')
    #     plt.title(ticks[i])
    #     plt.savefig(f'gradcam/gradcam_{ticks[i]}.png')
    #################################################
    # Experoment 1: combine into binary classification
    # Normal (0) : 0
    # Abnormal (1) : 1, 2, 3, 4, 5, 6
    if mode == 'any':
        y_pred[y_pred > 0] = 1
        test_y[test_y > 0] = 1
    
    # Mild stage (1) : 1, 2
    if mode == 'mild':
        y_pred[y_pred == 1] = 1
        y_pred[y_pred == 2] = 1
        y_pred[y_pred == 3] = 0
        y_pred[y_pred == 4] = 0
        y_pred[y_pred == 5] = 0
        y_pred[y_pred == 6] = 0
        
        test_y[test_y == 1] = 1
        test_y[test_y == 2] = 1
        test_y[test_y == 3] = 0
        test_y[test_y == 4] = 0
        test_y[test_y == 5] = 0
        test_y[test_y == 6] = 0
    # Severe stage (2) : 3, 4, 5, 6
    if mode == 'severe':
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == 2] = 0
        y_pred[y_pred == 3] = 1
        y_pred[y_pred == 4] = 1
        y_pred[y_pred == 5] = 1
        y_pred[y_pred == 6] = 1
        
        test_y[test_y == 1] = 0
        test_y[test_y == 2] = 0
        test_y[test_y == 3] = 1
        test_y[test_y == 4] = 1
        test_y[test_y == 5] = 1
        test_y[test_y == 6] = 1
    
    
    
    #################################################

    accuracy = np.sum(test_y == y_pred) / len(test_y)
    print('accuracy:', accuracy)


    # # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, y_pred)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10, 10))
    print('mode:', mode)
    if mode == 'multi':
        plot_confusion_matrix(cnf_matrix, classes=['normal', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5', 'end']) 
        plt.savefig(f'regression_report/confusion_matrix_{mode}.png')
        exit()
    elif mode == 'any':
        plot_confusion_matrix(cnf_matrix, classes=['normal', 'abnormal']) 
    elif mode == 'severe':
        plot_confusion_matrix(cnf_matrix, classes=['others', 'severe'])
    elif mode == 'mild':
        plot_confusion_matrix(cnf_matrix, classes=['others', 'mild'])

    plt.savefig(f'regression_report/confusion_matrix_{mode}.png')
    
    
    # Calculate area under the curve
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(test_y, y_pred)
    print('roc_auc:', roc_auc)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cnf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('sensitivity:', sensitivity)
    print('specificity:', specificity)
    
    # Calculate PPV and NPV
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print('ppv:', ppv)
    print('npv:', npv)
    


       


    



    


