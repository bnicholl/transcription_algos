#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:41:13 2022

@author: bennicholl
"""


from transformers import *
import torch
import soundfile as sf
#https://www.thepythoncode.com/article/speech-recognition-using-huggingface-transformers-in-python


# import librosa
import os
import torchaudio

#model_name = "facebook/wav2vec2-base-960h"
model_name = "facebook/hubert-large-ls960-ft"
#model_name = "facebook/wav2vec2-large-960h" # 360MB
#model_name = "facebook/wav2vec2-large-960h-lv60-self" # 1.18GB

processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#model = Wav2Vec2ForCTC.from_pretrained(model_name)

#model = Wav2Vec2ForCTC.from_pretrained(model_name)
model = HubertForCTC.from_pretrained(model_name)
#model = torch.load('/Users/bennicholl/Desktop/call_center_algos/models/wavbase')



#audio_url = "/Users/bennicholl/Desktop/o4157234000 by  @ 3_52_44 AM_Main recording-stereo.wav"
audio_url = "/Users/bennicholl/Desktop/call_center_algos/transcripts/two/9182302279 by 70471 @ 20201216084616.wav"
# load our wav file

#sr, or The sampling rate refers to the number of samples of audio recorded every second.
#It is measured in samples per second or Hertz (abbreviated as Hz or kHz, with one kHz 
#being 1000 Hz). An audio sample is just a number representing the measured acoustic wave 
#value at a specific point in time
speech, sr = torchaudio.load(audio_url)
speech = speech.squeeze()

# torchaudio can be used for transforms
# resample from whatever the audio sampling rate to 16000
resampler = torchaudio.transforms.Resample(sr, 16000)


#### below is logic for our while loop
break_next_loop = False
transcription_list = []

check_this_many_frequencies = 200000
first_indice = 0
second_indice = check_this_many_frequencies

max_amount_second_indice = speech.shape[0]

while True:
    
    speech1 = speech[first_indice:second_indice]
    #speech1 = speech1[0:400000]
    

    speech1 = resampler(speech1)
    
    
    # tokenize our wav
    input_values = processor(speech1, return_tensors="pt", sampling_rate=16000)["input_values"]
    
    
    # perform inference
    logits = model(input_values)["logits"]
    #logits = model(speech.unsqueeze(0))["logits"]
    
    
    
    # use argmax to get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    
    
    
    #predicted_ids[0] is a list of letters
    # for example 21 is g, 8 is o, and 14 is d notice below list
    # [21,  8,  0,  0,  8,8, 14], it looks like its spelling goood, but the algorithm 
    # is able to recognize it should be good
    transcription = tokenizer.decode(predicted_ids[0])
    transcription_list.append(transcription)
    
    if second_indice < max_amount_second_indice:
        first_indice += check_this_many_frequencies
        second_indice += check_this_many_frequencies
    
    elif break_next_loop == True:
        break
        
    else:
        break
        
        first_indice += check_this_many_frequencies
        second_indice = second_indice + (max_amount_second_indice - second_indice)
        break_next_loop = True








