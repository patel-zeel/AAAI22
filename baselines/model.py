"""
Disclaimer:

This is our own implementation of "A Neural Attention Model for Urban Air Quality Inference: Learning the Weights of Monitoring Stations" (ADAIN) paper 
produced after consulting with the original authors. Please note that this is not the original code used in the ADAIN paper, so if this code is 
used to reproduce the ADAIN paper, it may not give the exact same results but expected to give similar results.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Lambda, Multiply, Reshape, Input, Dropout, Attention, Flatten
import tensorflow.keras.backend as K

def ADAIN(met, dist, aq, time_window, dropout):
    """
    met = meteorological data shape
    aq = air quality data shape
    stat = number of stations
    """

    
    local_inputs2d = Input(shape=(met, time_window, ))                              # (-1, met, time_window)
    all_station_fc_input = Input(shape=(None, dist, ))                              # (-1, stat, dist)
    all_station_lstm_input = Input(shape=(None, met+aq, time_window))               # (-1, stat, met+aq, time_window)


    local_lstm_layer = LSTM(300,
                            return_sequences=True)(local_inputs2d)                  # (-1, 300, time_window)
    local_lstm_layer2 = LSTM(300)(local_lstm_layer)              # (-1, 300)
    

    local_dense_layer1 = Dense(200, activation='relu')(local_lstm_layer2)               # (-1, 200)
    local_dense_layer1 = Dropout(dropout)(local_dense_layer1)               
    local_dense_layer2 = Dense(200, activation='relu')(local_dense_layer1)          # (-1, 200)
    local_dense_layer2 = Dropout(dropout)(local_dense_layer2)

    fmul_list = []
    no_stations = 2
    for n in range(no_stations):

        # station_fc_input = Lambda(lambda x: x[:, n, :])(all_station_fc_input)       # (-1, dist)
        # station_lstm_input = Lambda(
            # lambda x: x[:, n, :])(all_station_lstm_input)                           # (-1, met+aq, time_window)

        station_fc_input = all_station_fc_input[:,n,:]
        station_lstm_input = all_station_lstm_input[:,n,:]

        station_fc_layer = Dense(100, activation='relu')(station_fc_input)          # (-1, 100)
        station_fc_layer = Dropout(dropout)(station_fc_layer)

        station_lstm_layer = LSTM(
            300, return_sequences=True)(station_lstm_input)      # (-1, 300, time_window)
        
        station_lstm_layer2 = LSTM(300)(station_lstm_layer)      # (-1, 300)

        # station_fc_layer = Flatten()(station_fc_layer)
        station_output1 = Concatenate()(
            [station_fc_layer, station_lstm_layer2])                                # (-1, 400)

        station_dense_layer1 = Dense(200, activation='relu')(station_output1)       # (-1, 200)
        station_dense_layer1 = Dropout(dropout)(station_dense_layer1)
        station_dense_layer2 = Dense(
            200, activation='relu')(station_dense_layer1)                           # (-1, 200)
        station_dense_layer2 = Dropout(dropout)(station_dense_layer2)

        Attn_1 = Dense(200, activation='relu')(
            Concatenate()([local_dense_layer2, station_dense_layer2]))              # (-1, 200)
        Attn_2 = Dense(200)(Attn_1)                                                 # (-1, 200)
        Attn_3 = Dense(1, activation='softmax')(Attn_2)                             # (-1, 1)

        fmul_element = Multiply()([station_dense_layer2, Attn_3])                   # (-1, 200)
        fmul_element = Lambda(lambda x: K.reshape(x, (-1,1,200)))(fmul_element)
        fmul_list.append(fmul_element)

    fmul = Concatenate(axis=1)(fmul_list)                                           # (-1, stat, 200)

    fA = Lambda(lambda x: K.sum(x, axis=1))(fmul)                                   # (-1, 200)    

    final_input = Concatenate()([local_dense_layer2, fA])                           # (-1, 400)
    final_fc_layer = Dense(200, activation='relu')(final_input)                     # (-1, 200)
    final_fc_layer = Dropout(dropout)(final_fc_layer)                               

    output = Dense(1)(final_fc_layer)                                               # (-1, 1)

    model = Model(
        inputs=[local_inputs2d, all_station_fc_input, all_station_lstm_input], outputs=output)

    return model
