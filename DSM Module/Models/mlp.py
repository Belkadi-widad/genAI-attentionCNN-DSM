from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def MLP_training(
    param_dict,
    X_train,
    y_train,
    n_outputs,
    validation_data=None,
    validation_split=None
):

    dropout_rate = param_dict["dropout_rate"]
    hidden_units = param_dict["nb_neurones_by_layer"]
    hidden_activation = param_dict["hidden_activation"]
    learning_rate = param_dict["lr"]
    output_activation = param_dict["output_activation"]
    batch_size = param_dict["batch_size"]
    epochs = param_dict["epochs"]

    input_layer = Input(shape=(X_train.shape[1],))
    x = input_layer

    for units in hidden_units:
        x = Dense(units, activation=hidden_activation)(x)
        x = Dropout(dropout_rate)(x)

    output_layer = Dense(n_outputs, activation=output_activation)(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    callbacks = []
    if validation_data is not None or validation_split is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        verbose=0,
        callbacks=callbacks,
    )

    return model, history.history