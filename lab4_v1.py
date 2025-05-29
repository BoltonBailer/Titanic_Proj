import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

# regression vs classification
if __name__ == '__main__':

    # load trial info
    output = r''
    path = r''
    trials = pd.read_csv(path)

    for i, row in trials.iterrows():

        # ------------------------------------------------------------------------
        # load data
        # see if we even need to do trial
        if not np.isnan(row['mae']):
            continue

        # grab start time
        start_time = time.time()
        output = r'C:\Users\devlu\Downloads\DPL\DPL\labs\output\week4_output'
        path = r'C:\Users\devlu\Downloads\DPL\DPL\labs\data\auto.csv'
        seed = 123857

        df = pd.read_csv(path)

        # grab only numerical columns
        numerical_df = df.select_dtypes(include=['int64', 'float64'])

        # pick features (drop the output)
        in_feat = numerical_df.drop(columns=['MPG_City']).columns
        out_feat = 'MPG_City'

        # drop rows with missing data
        df = df.dropna(subset=in_feat.tolist() + [out_feat])

        
        n_nuro = int(row['n_nuro'])
        n_layers = int(row['n_layers'])
        activation = row['activation']
        n_epochs = 20

        X = df[in_feat]
        y = df[out_feat]

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

        # create the model
        model = Sequential()
        model.add(Input(shape=(len(in_feat),)))  # input layer

        # add multiple hidden layers
        for _ in range(n_layers):
            model.add(Dense(n_nuro, activation=activation))

        model.add(Dense(1))

        # finalize model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # start training
        model.fit(X_train, y_train, epochs=n_epochs, verbose=1)

        # validate model
        metric = mean_absolute_error(y_test, model.predict(X_test))
        print(f'Trial {i} MAE: {metric:.4f}')

        # save out metrics
        trials.loc[i, 'mae'] = metric
        trials.loc[i, 'trial_time'] = time.time() - start_time

    # save updated trial file
    path = r'C:\Users\devlu\Downloads\DPL\DPL\labs\output\week4_output\trials.csv'
    trials.to_csv(path, index=False)
