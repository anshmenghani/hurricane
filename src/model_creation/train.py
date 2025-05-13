from tensorflow.keras.layers import Input, Dense, LayerNormalization, GaussianDropout, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredLogarithmicError as MSLE
from sklearn.model_selection import train_test_split
import pandas as pd

def HurricanePathModel(shape):
    input_layer = Input(shape=shape)
    norm_input = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(input_layer)

    hidden1 = Dense(32, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(norm_input)
    hidden1 = PReLU()(hidden1)
    hidden1 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden1)
    hidden1 = GaussianDropout(0.6)(hidden1)

    hidden2 = Dense(64, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(hidden1)
    hidden2 = PReLU()(hidden2)
    hidden2 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden2)
    hidden2 = GaussianDropout(0.4)(hidden2)

    hidden3 = Dense(32, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(hidden2)
    hidden3 = PReLU()(hidden3)
    hidden3 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden3)

    output = Dense(1)(hidden3)

    closing_price_model = Model(inputs=input_layer, outputs=output)
    return closing_price_model


x_train, x_test, y_train, y_test = loadData(path)
hpm = HurricanePathModel((100, 3,)) # change shape later 
hpm.compile(optimizer=Adam(learning_rate=0.001), loss=MSLE(), metrics=["accuracy"], run_eagerly=False, jit_compile=False, steps_per_execution=1)
hpm.fit(x=x_train, y=y_train, validation_split=0.20, epochs=100, batch_size=32, shuffle=False, validation_batch_size=16, validation_freq=2) # Make sure shuffle=False so that hurricanes dont get ungrouped
hpm.save("src/model/hpm.keras")
