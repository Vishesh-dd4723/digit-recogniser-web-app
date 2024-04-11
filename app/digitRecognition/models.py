from keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.initializers import random_uniform, glorot_uniform


class ResNet50:
    def __init__(self, input_shape, classes) -> None:
        self.input_shape = input_shape
        self.classes = classes

    def identity_block(self, X, f, filters, training=True, initializer=random_uniform):
        # Retrieve Filters
        F1, F2, F3 = filters

        # Saving the input
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(
            X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(
            X)
        X = BatchNormalization(axis=3)(X, training=training)

        # Final step: Add shortcut value to main path
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def convolutional_block(self, X, f, filters, s=2, training=True, initializer=glorot_uniform):
        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(
            X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(
            X)
        X = BatchNormalization(axis=3)(X, training=training)

        # SHORTCUT PATH
        X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid',
                            kernel_initializer=initializer(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

        # Final step
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def build(self) -> Model:
        # Defining the input as a tensor with shape input_shape
        X_input = Input(self.input_shape)

        # Zero-Padding
        X = ZeroPadding2D((21, 21))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.identity_block(X, 3, [64, 64, 256])

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = AveragePooling2D(pool_size=(2, 2))(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(self.classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X)

        return model


class SimpleNN:
    def __init__(self, input_shape, classes) -> None:
        self.input_shape = input_shape
        self.classes = classes

    def conv_block(self, X, filter, kernel, s, training=True, initializer=glorot_uniform):
        X = Conv2D(filters=filter, kernel_size=kernel, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(
            X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)
        return X

    def build(self):
        X_input = Input(self.input_shape)
        X = ZeroPadding2D((2, 2))(X_input)

        X = self.conv_block(X, 2, 1, 1)
        X = self.conv_block(X, 4, 1, 1)
        X = self.conv_block(X, 8, 3, 1)
        X = MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = self.conv_block(X, 16, 3, 1)
        X = MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = Flatten()(X)
        X = Dense(512, activation='leaky_relu')(X)
        X = Dense(256, activation='leaky_relu')(X)
        X = Dense(self.classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X)

        return model