import numpy as np
from pickle import load, dump
from load import load_default
from data_format import data_processing
from keras.models import Input, Model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot


batch_size = 64
epochs = 50
num_samples = 128


def df_analysis(df):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for _, review in df.iterrows():
        input_text = review['reviewText']
        summary = review['summary']
        target_text = summary
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            decoder_target_data[i, t:, target_token_index[' ']] = 1.


    return num_encoder_tokens, num_decoder_tokens, \
           max_encoder_seq_length, max_decoder_seq_length, \
           encoder_input_data, decoder_input_data, decoder_target_data


def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # return all models
    return model, encoder_model, decoder_model


def load_model(name):
    with open(name, "r") as f:
        return load(f)


def save_model(name, obj):
    with open(name, "wb") as f:
        dump(obj, f)


def train_model():
    df = data_processing((load_default(limit=num_samples)))
    num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, encoder_input_data, decoder_input_data, decoder_target_data = df_analysis(df)
    model, encoder_model, decoder_model = define_models(num_encoder_tokens, num_decoder_tokens, max_decoder_seq_length)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    decoder_model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[es])

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    save_model("../models/deeplearning_enc_dec_trained_128.pckl", history)


if __name__ == "__main__":
    model = load_model("../models/deeplearning_enc_dec_trained_128.pckl")