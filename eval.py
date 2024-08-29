import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
import os
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from utils import segment_array


def create_tsne_plot(y_pred, testY, features, classes, initial_dims=20):
    """
    Constructs a t-SNE plot given predicted labels (y_pred), true labels (testY), and features.
    Assumes the labels are provided in one-hot encoded form.

    :param y_pred: The predicted class labels for the data points, one-hot encoded.
    :param testY: The true class labels for the data points, one-hot encoded.
    :param features: The feature set to be used for t-SNE.
    """
    # Convert one-hot encoded labels to single digit class labels if necessary
    if y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if testY.shape[1] > 1:
        testY = np.argmax(testY, axis=1)

    num_samples, height, width_channels = features.shape
    features = features.reshape(num_samples, height * width_channels)
    features = StandardScaler().fit_transform(features)

    if features.shape[1] > initial_dims:
        # Use PCA to reduce the dimensionality of the features
        pca = PCA(n_components=initial_dims)
        features = pca.fit_transform(features)
        

    # Instantiate a t-SNE object
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000)
    
    # Perform t-SNE on the features
    tsne_results = tsne.fit_transform(features)
    
    # Plotting the results
    plt.figure(figsize=(16,10))
    
    # Scatter plot for each true class label
    
    for class_id in np.unique(testY):
        plt.scatter(tsne_results[testY == class_id, 0], tsne_results[testY == class_id, 1],
                    c=[plt.cm.tab10.colors[class_id % 10]], label=classes[class_id])
        
    plt.title('t-SNE plot of predicted labels')
    plt.legend()
    plt.savefig('tsne_plot.png')




def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':
    # load dataset
    print('loading datasets...')
    # with h5py.File('tmp/train.h5', 'r') as hf:
    #     train_x = hf['train_x'][:]
    #     train_y = hf['train_y'][:]
    # with h5py.File('tmp/val.h5', 'r') as hf:
    #     val_x = hf['val_x'][:]
    #     val_y = hf['val_y'][:]
    with h5py.File('tmp/test_.h5', 'r') as hf:
        test_x = hf['test_x'][:]
        test_y = hf['test_y'][:]
    print('datasets loaded')

    #load model
    checkpoint_path = 'checkpoints/model_classification.keras'
    model = tf.keras.models.load_model(checkpoint_path)
    # loss , acc = model.evaluate(test_x , test_y)
    # print('loss:' , loss)
    # print('accuracy: ' , acc)

    test_x = np.transpose(test_x, (0, 2, 1))
    #####################################################################################################
    # Experiment : 
    # The input shape is now (n, 12, 1000). Segment into (n*k, 12, 1000/k)
    # segment_length = 200
    

    # print('original: ', test_x.shape, test_y.shape)
    # test_x = segment_array(test_x, segment_length=segment_length)


    # test_y = [item for item in test_y for _ in range(int(1000/segment_length))]
    # test_y = np.array(test_y)

    # print('Segmented: ', test_x.shape, test_y.shape)
    #####################################################################################################

    # if not os.path.exists('tmp/y_pred.h5'):
            
    #     y_pred = model.predict(test_x)
    #     y_pred = np.nan_to_num(y_pred)

    # else:
    #     with h5py.File('tmp/y_pred.h5', 'r') as hf:
    #         y_pred = hf['y_pred'][:]

    # # save y_pred in h5 file
    # with h5py.File('tmp/y_pred.h5', 'w') as hf:
    #     hf.create_dataset("y_pred",  data=y_pred)

    y_pred = model.predict(test_x)
    y_pred = np.nan_to_num(y_pred)

    accuracy = metrics.accuracy_score(test_y.argmax(axis=1), y_pred.argmax(axis=1))
    print('Accuracy:', accuracy)
  

    print('F1 score:', f1_score(test_y.argmax(axis=1), y_pred.argmax(axis=1), average='weighted'))

    # # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
    np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix, classes=['stage_noemal', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5', 'end'])
    plt.savefig('classification_report/confusion_matrix.png')

    test_x = np.nan_to_num(test_x)
    test_x = np.array(test_x)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])


    # Calculate area under the curve
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(test_y, y_pred)
    print('roc_auc:', roc_auc)
    
    # # Calculate sensitivity and specificity
    # tn, fp, fn, tp = cnf_matrix.ravel()
    # sensitivity = tp / (tp + fn)
    # specificity = tn / (tn + fp)
    # print('sensitivity:', sensitivity)
    # print('specificity:', specificity)
    
    # # Calculate PPV and NPV
    # ppv = tp / (tp + fp)
    # npv = tn / (tn + fn)
    # print('ppv:', ppv)
    # print('npv:', npv)