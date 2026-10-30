import pandas as pd
import numpy as np
import os
os.environ['USE_PYGEOS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.callbacks import TerminateOnNaN


########################## MLP training ############################### 

# def MLP_training(param_dict, X_train, y_train, n_outputs, validation_data=None, validation_split= None):

#     dr = param_dict['dropout_rate']
#     nb_neurones = param_dict['nb_neurones_by_layer']
#     hidden_activation= param_dict['hidden_activation']
#     lr = param_dict['lr']
#     output_activation = param_dict['output_activation' ]
#     batch_size = param_dict['batch_size']
#     epochs = param_dict['epochs']

#     input_layer = Input(shape=(X_train.shape[1],))
#     dropout= input_layer


#     for nb_n in nb_neurones:    
#         hidden_layer = Dense(nb_n, activation=hidden_activation)(dropout)
#         dropout = Dropout(dr)(hidden_layer)
#     output_layer = Dense(n_outputs, activation=output_activation)(dropout)

#     model = Model(inputs=input_layer, outputs=output_layer)

#     optimizer = Adam(learning_rate=lr)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

#     es = EarlyStopping(monitor='val_loss', patience= 15, restore_best_weights=True)

#     mc = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True) 

#     #model.summary()

#     history = model.fit(X_train, y_train, 
#                         epochs=epochs, 
#                         batch_size=batch_size, 
#                         validation_data=validation_data,
#                         validation_split=validation_split,
#                         verbose=0,
#                         callbacks=[es, mc])
    
#     return model, history.history


def MLP_training(
    param_dict,
    X_train,
    y_train,
    n_outputs,
    validation_data=None,
    validation_split=None
):

    dr = param_dict['dropout_rate']
    nb_neurones = param_dict['nb_neurones_by_layer']
    hidden_activation = param_dict['hidden_activation']
    lr = param_dict['lr']
    output_activation = param_dict['output_activation']
    batch_size = param_dict['batch_size']
    epochs = param_dict['epochs']

    # -----------------------------
    # Model architecture
    # -----------------------------

    input_layer = Input(shape=(X_train.shape[1],))
    x = input_layer

    for nb_n in nb_neurones:
        x = Dense(
            nb_n,
            activation=hidden_activation
        )(x)

        x = Dropout(dr)(x)

    output_layer = Dense(
        n_outputs,
        activation=output_activation
    )(x)

    model = Model(
        inputs=input_layer,
        outputs=output_layer
    )

    # -----------------------------
    # Compile
    # -----------------------------

    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # -----------------------------
    # Callbacks
    # -----------------------------

    callbacks = []

    if validation_data is not None or validation_split is not None:

        es = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        callbacks.append(es)

    # -----------------------------
    # Training
    # -----------------------------

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        verbose=0,
        callbacks=callbacks
    )

    return model, history.history