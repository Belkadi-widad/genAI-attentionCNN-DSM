import os
import multiprocessing
from optuna.integration import TFKerasPruningCallback
import sys
# Set CPU threads BEFORE importing TensorFlow
num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cores)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
os.environ['USE_PYGEOS'] = '0'
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(True)
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW, Adadelta, Adagrad, Nadam, SGD
from optuna.integration import TFKerasPruningCallback
from CVs.cv_utils import build_folds_by_method


from tensorflow.keras.utils import register_keras_serializable
@register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation='sigmoid', use_bias=False)

    def call(self, inputs):
        # mean and max across channels
        avg_pool = K.mean(inputs, axis=-1, keepdims=True)
        max_pool = K.max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = self.conv(concat)
        return inputs * cbam_feature


# 1. Channel Attention
# -----------------------------
def channel_attention(input_feature, ratio=8, name=None):
    channel = input_feature.shape[-1]
    # Shared MLP layers
    shared_dense_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    name=f'{name}_mlp_0' if name else None)
    shared_dense_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    name=f'{name}_mlp_1' if name else None)
    # Global average pooling
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))
    # Global max pooling
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))
    # Combine and scale
    scale = layers.Activation('sigmoid')(avg_pool + max_pool)
    return layers.Multiply()([input_feature, scale])




def cbam_block(input_feature, ratio=12, kernel_size=7, name=None):
    x = channel_attention(input_feature, ratio, name=f'{name}_ch' if name else None)
    x = SpatialAttention()(x)
    return x

def create_cnn_optuna(trial, input_shape):
    """
    Create a CNN dynamically based on hyperparameters suggested by Optuna.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # --- Number of Conv blocks ---
    n_blocks = trial.suggest_int("n_blocks", 1, 2)

    for i in range(n_blocks):
        bp = f"block{i+1}"

        # --- Conv type ---
        conv_type = trial.suggest_categorical(
        f"{bp}_type",
        ["conv", "separable"]
        )

        filters = trial.suggest_categorical(f"filters_{bp}", [8, 16, 32,128])

        kernel_size = trial.suggest_categorical(f"kernel_{bp}", [3, 7])

        activation = trial.suggest_categorical(
        f"activation_{bp}",
        ["elu" , "relu", "gelu"]
        )

        dilation = trial.suggest_categorical(
        f"dilation_{bp}",
        [(1,1), (2,2)]
        )

        use_batchnorm = trial.suggest_categorical(f"batchnorm_{bp}", [True, False])

        use_cbam = trial.suggest_categorical(f"cbam_{bp}", [True, False])

        dropout_rate = trial.suggest_float(f"dropout_{bp}", 0.0, 0.3)

        # ---------- Build Conv ----------
        if conv_type == "conv":
            x = layers.Conv2D(filters, kernel_size,
                          padding="same",
                          activation=None,
                          dilation_rate=dilation,
                          strides=(1,1))(x)
        else:
            x = layers.SeparableConv2D(filters, kernel_size,
                                   padding="same",
                                   activation=None,
                                   dilation_rate=dilation,
                                   strides=(1,1))(x)

        # BatchNorm
        if use_batchnorm:
            x = layers.BatchNormalization()(x)

        # Activation
        x = layers.Activation(activation)(x)

        # Optional CBAM
        if use_cbam:
            x = cbam_block(x, name=f"cbam_{bp}")

        # ---------- Optional Pooling (SAFE) ----------
        h = x.shape[1]
        w = x.shape[2]

        use_pooling = trial.suggest_categorical(f"pooling_{bp}", [True, False])

        if use_pooling and h >= 4 and w >= 4:
            # Safe pooling, reduces 7â†’4 or 5â†’3 or 4â†’2
            x = layers.MaxPooling2D(pool_size=(2,2), padding="same")(x)

        # Dropout
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    # --- Dense layers ---
    x = layers.GlobalAveragePooling2D()(x)

    n_dense = trial.suggest_int("n_dense", 1, 2)
    for i in range(n_dense):
        units = trial.suggest_categorical(f"dense_units_{i+1}", [50, 100, 200])
        activation = trial.suggest_categorical(f"dense_activation_{i+1}", ["relu", "tanh", "elu"])
        x = layers.Dense(units, activation=activation)(x)
        dropout_rate = trial.suggest_float(f"dense_dropout_{i+1}", 0.0, 0.5)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs, outputs)
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience= 10,          # number of epochs with no improvement before stopping
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,     # reduce LR by half
    patience=5,     # after 5 epochs with no improvement
    min_lr=1e-6
)


callbacks = [early_stop, reduce_lr]

