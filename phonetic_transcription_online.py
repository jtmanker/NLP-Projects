# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:27:32 2021

@author: jtman
"""


import numpy as np
from keras.models import Model, model_from_json
from keras.layers import Input

#takes the numerical sequence representing a spelled word and trains it on the sequence-to-sequence model
def decode_sequence(input_seq):
    thought = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, len(phone_idx)))
    stop_condition = False
    generated_sequence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + thought)
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = num_phone_idx[generated_token_idx]
        generated_sequence += generated_char
        if (generated_char == stop_token or len(generated_sequence) > trans_max_len):
            stop_condition = True
        target_seq = np.zeros((1, 1, len(phone_idx)))
        target_seq[0, 0, generated_token_idx] = 1.
        thought = [h, c]
    return generated_sequence


#takes a spelled word and converts it to numbers, which it feeds into the model
def response(input_text):
    input_seq = np.zeros((1, max_len, len(letter_idx)), dtype = 'float32')
    for t, char in enumerate(input_text):
        input_seq[0, t, letter_idx[char]] = 1.
    decoded_sentence = decode_sequence(input_seq)
    print ('Transcription:', decoded_sentence)

#reading in file
file = "D:/librispeech-lexicon.txt"
transcription_file = open(file, "r")
word_transcriptions = transcription_file.readlines()
words = [word for line in word_transcriptions for word in line.split()]


#reading in word transcriptions
words = []
transcriptions = []
for line in word_transcriptions:
    words.append(line.split()[0])
    transcriptions.append(line.split()[1:])

new_words = []
new_trans = []
start_token = '\n'
stop_token = '\t'


for word_trans in zip(words, transcriptions):  #removing words over a certain length
    if len(word_trans[0]) <= 18:
        new_words.append(word_trans[0])
        new_trans.append([start_token] + word_trans[1] + [stop_token])
        
   
#gathering all the possible phones and letters
phones = sorted(set([sound for line in new_trans for sound in line]))
letters = sorted(set([letter for line in new_words for letter in line]))

max_len = max([len(word) for word in new_words])
trans_max_len = max([len(word) for word in new_trans])

#dictionaries for character -> number conversions
letter_idx = {char:i for i, char in enumerate(letters)}
num_letter_idx = {i:char for i, char in enumerate(letters)}
phone_idx = {char: i for i, char in enumerate(phones)}
num_phone_idx = {i:char for i, char in enumerate(phones)}


#one hot encoding each word with 1s for each letter
encoder_one_hot_input = np.zeros((len(new_words), max_len, len(letter_idx)), dtype = 'float32')
decoder_one_hot_input = np.zeros((len(new_trans), trans_max_len, len(phone_idx)), dtype = 'float32')
decoder_one_hot_output = np.zeros((len(new_trans), trans_max_len, len(phone_idx)), dtype = 'float32')

for i, word in enumerate(new_words):
    for j, letter in enumerate(word):
        encoder_one_hot_input[i, j, letter_idx[letter]] = 1
        
for i, word in enumerate(new_trans):
    for j, phone in enumerate(word):
        decoder_one_hot_input[i, j, phone_idx[phone]] = 1
        if j > 0:
            decoder_one_hot_output[i, j-1, phone_idx[phone]] = 1
        

#hyperparameters
num_neurons = 256

start_token = '\t'
stop_token = '\n'

with open("phonetic_transcription.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights('phonetic_transcription_weights.h5')


#building the sequence-to-sequence model
encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.layers[2].output
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_state_input_h = Input(shape=(num_neurons,))
decoder_state_input_c = Input(shape=(num_neurons,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

#tryingout out some fake words to see what their predicted pronunciation is
response('QUAQUE')
response('PHAYTHOSE')
response('PNEUCHO')
