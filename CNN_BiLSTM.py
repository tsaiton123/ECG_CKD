import tensorflow as tf
from tensorflow.keras import layers, models

def build_ecg_model(input_shape=(12, 5000, 1), num_classes=7, last_layer='softmax'):
    inputs = tf.keras.Input(shape=input_shape)

    # Temporal Convolution (Reduced filters and layers)
    x = layers.Conv2D(filters=8, kernel_size=(1, 32), strides=(1, 4), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=16, kernel_size=(1, 64), strides=(1, 4), padding='same', activation='relu')(x)
    
    # Spatial Convolution (Reduced filters)
    x = layers.Conv2D(filters=4, kernel_size=(3, 1), padding='same', activation='relu')(x)
    
    # Feature Convolutions (Reduced filters and depth multiplier)
    x = layers.SeparableConv2D(filters=8, kernel_size=(1, 8), depth_multiplier=1, strides=(1, 2), padding='same', activation='relu')(x)
    
    # Reshape the output to fit LSTM input requirements
    x = layers.Reshape((x.shape[1], -1))(x)  # Combine spatial dimensions with channels
    
    # LSTM (Reduced units)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    
    # Batch Normalization
    x = layers.BatchNormalization()(x)
    
    # Dense Output (Reduced number of units)
    if last_layer == 'softmax':
        outputs = layers.Dense(num_classes, activation=last_layer)(x)
    elif last_layer == 'linear':
        outputs = layers.Dense(1)(x)
        outputs = layers.Activation('sigmoid')(outputs) * num_classes - 1

    # Create the model
    model = models.Model(inputs, outputs)
    
    return model

if __name__ == '__main__':

    # Model creation
    input_shape = (12, 5000, 1)  # ECG input shape
    num_classes = 71  # Number of output classes
    model = build_ecg_model(input_shape, num_classes)

    # Model summary
    model.summary()
