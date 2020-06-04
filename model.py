from keras.models import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, Dense , Flatten
from keras import optimizers
import pickle
import keras_metrics as km
from keras.regularizers import l2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


BASELINE=0.9

def get_class_weights(y_encoded):

    weights = dict()

    for y in y_encoded:
        pos = np.argmax(y)
        try:
            weights[pos] += 1
        except KeyError:
            weights[pos] = 1

    for w in weights:
        weights[w] = y_encoded.shape[0] / weights[w]
    return weights


def generate_model(n_steps, n_features, n_categories):
    model = Sequential()

    ks = 5
    filters = 250

    # convolutional part
    model.add(Conv1D(filters=filters, kernel_size=ks, activation='relu',
                     strides=1, input_shape=(n_steps, n_features), kernel_regularizer=l2()))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=filters, kernel_size=ks, activation='relu', strides=1, kernel_regularizer=l2()))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    # deep part
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories, activation="softmax"))

    opt = optimizers.Adam(lr=1e-4)

    f1 = km.categorical_f1_score()

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy", f1])

    return model


def send_to_model(X, y, epochs, station):

    batch_size = 25
    cw = get_class_weights(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    n_steps = X_train.shape[1]

    n_features = X_train.shape[2]
    n_categories = y.shape[1]

    early_stop = EarlyStopping(monitor='val_f1_score', patience=0, min_delta=0.001, mode='min',
                               baseline=BASELINE, verbose=1)


    # design network
    model = generate_model(n_steps, n_features, n_categories)
    # fit network
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.3,
                        verbose=2,
                        shuffle=True,
                        class_weight=cw,
                        callbacks=[early_stop])

    output = model.evaluate(X_test, y_test, batch_size=batch_size)


    # try:
    #     df = pd.read_excel("data/models/results.xlsx")
    # except:
    #     df = None
    #
    # tmp = pd.DataFrame(columns=["Station", "Accuracy score", "f1 score"])
    # tmp["Station"] = [station]
    # tmp["Accuracy score"] = [output[1]]
    # tmp["f1 score"] = [output[2]]
    # if df is None:
    #     df = tmp
    # else:
    #     df = pd.concat((df, tmp), ignore_index=True)
    # df.to_excel("data/models/results.xlsx", index=False)
    print('Score: %.2f, Acc: %.2f, f1: %.2f') % (output[0], output[1], output[2])


    # model.save("data/models/model_%s.hdf" % station)


def encoder(y_list):
    encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y_list)
    y_encoded = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

    return y_encoded


def main(epochs=500, station="DEDWA"):

    data_dict = pickle.load(open('data/stations/%s_labeled.pkl' % station, "rb"))

    X = data_dict["x"]
    y = data_dict["y"]

    y = encoder(y)

    send_to_model(X, y, epochs, station)



if __name__ == "__main__":
    main()
