from typing import Sequence
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam

from .config import Config


def build_lstm(input_shape, cfg: Config) -> Model:
    seq = Sequential()
    seq.add(Input(shape=input_shape))
    seq.add(LSTM(cfg.lstm_units[0], return_sequences=True))
    seq.add(Dropout(cfg.dropout))
    seq.add(LSTM(cfg.lstm_units[1], return_sequences=False))
    seq.add(Dropout(cfg.dropout))
    seq.add(Dense(1))
    seq.compile(optimizer=Adam(learning_rate=cfg.learning_rate), loss="mse")
    return seq


def build_gru(input_shape, cfg: Config) -> Model:
    seq = Sequential()
    seq.add(Input(shape=input_shape))
    seq.add(GRU(cfg.gru_units[0], return_sequences=True))
    seq.add(Dropout(cfg.dropout))
    seq.add(GRU(cfg.gru_units[1], return_sequences=False))
    seq.add(Dropout(cfg.dropout))
    seq.add(Dense(1))
    seq.compile(optimizer=Adam(learning_rate=cfg.learning_rate), loss="mse")
    return seq
