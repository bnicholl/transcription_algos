#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:48:59 2022

@author: bennicholl
"""
from transformers import Wav2Vec2Processor, HubertForCTC, AdamW, get_linear_schedule_with_warmup, Wav2Vec2ForCTC
import torch
import pandas as pd
import torchaudio
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import copy

"""gets 16,000 frames per second"""
our_sampling_rate_value = 16000


#processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
#model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", ctc_loss_reduction="mean")


#data= pd.read_csv('/Users/bennicholl/Desktop/call_center_algos/youtube_data/talk_show_confrontations.txt')

#audio_url = '/Users/bennicholl/Desktop/call_center_algos/youtube_data/top-10-most-confrontational-talk-show-moments.wav'

data= pd.read_csv('/Users/bennicholl/Desktop/call_center_algos/youtube_data/talk_show_awkward/talk_show_awkwardy.txt')

audio_url = '/Users/bennicholl/Desktop/call_center_algos/youtube_data/talk_show_awkward/when-talk-show-hosts-make-celebrities-uncomfortable.wav'



speech, sr = torchaudio.load(audio_url)

if len(speech) == 2:
    speech = speech.mean(0)
    
    
# resmaple at 16,000 frames oper second    
resampler = torchaudio.transforms.Resample(sr, our_sampling_rate_value)
speech = resampler(speech)


# tokenize our wav
input_values = processor(speech, return_tensors="pt", sampling_rate=our_sampling_rate_value)["input_values"]
input_values = input_values.tolist()

for e,i in enumerate(speech):
    if i !=0:
        print('first non zero frequency is at ', e/our_sampling_rate_value, ' seconds')
        break



def reengineer(data, how_much_data_per_seconds = 1):
    amount_of_seconds = 0.0
    #################
    new_data = []
    for index, row in data.iterrows():
        """only get the even rows, because that is where our times are """
        if index % 2 == 0:
            """below three lines turns 0:3 into 3 seconds """
            current_time = datetime.datetime.strptime(row.data, "%M:%S")
            timedelta = current_time - datetime.datetime(1900, 1, 1)
            seconds = timedelta.total_seconds()
            
            if seconds - amount_of_seconds >= how_much_data_per_seconds:
                new_data.append(seconds)
                amount_of_seconds = seconds
        
        else:
            new_data.append(row.data)
            
    return new_data
            
reengineered_data = reengineer(data)


def reengineer_data_with_raw_audio(reengineered_data, input_values):
    raw_waveform_training_examples = []
    transcript_training_examples_labels = []
    
    transcript_training_example = []
    
    first_indice_for_raw_data = 0
    for data in reengineered_data:
        """checks if our value is how many seconds in we are in the video """
        if type(data) == float:
            """this multiplies the amount of seconds we are iterated on and multiples that by our sampling rate
            this will act as the second indice for our raw wave data for trainng examples """
            second_indice_for_raw_data = int(our_sampling_rate_value * data)
            """here we get our raw waveform data with our first and second indice """
            raw_waveform_training_examples.append(input_values[0][first_indice_for_raw_data:second_indice_for_raw_data])
            """gets our text/transcript training examples """
            transcript_training_examples_labels.append(" ".join(transcript_training_example))
            transcript_training_example = []
            
            first_indice_for_raw_data = second_indice_for_raw_data
        else:
            transcript_training_example.append(data)
        

    return raw_waveform_training_examples, transcript_training_examples_labels

raw_waveform_training_examples, transcript_training_example_labels = reengineer_data_with_raw_audio(reengineered_data, input_values)
    
#################### I only take some data here so It doenst creash my memory
raw_waveform_training_examples = raw_waveform_training_examples[40:42]
transcript_training_example_labels = transcript_training_example_labels[40:42]

############################# Delete when you put this on sagemaker

def get_mask_and_pad_inputs(raw_waveform_training_examples):
    training_mask = []
    
    max_amount_of_waveform_vals = max([len(i) for i in raw_waveform_training_examples])
    
    for training_example in raw_waveform_training_examples:
        amount_of_waveform_vals = len(training_example)
        ones = [1] * amount_of_waveform_vals
        
        zeros = [0] * (max_amount_of_waveform_vals - amount_of_waveform_vals)
        
        training_example.extend(zeros)
        ones.extend(zeros)
        
        training_mask.append(ones)
    
    #  might be cleaner if I return raw_waveform_training_examples
    return training_mask

#training_mask = get_mask_and_pad_inputs(raw_waveform_training_examples)




def create_labels_and_pad():
    ##### this below code creates our labels and pads them
    labels = []
    with processor.as_target_processor():
        for sentence in transcript_training_example_labels:
            labels.append(processor(sentence.upper(), return_tensors="pt").input_ids.tolist()[0])
            
    
    return labels

labels = create_labels_and_pad()



def run_hubert(inputs, labels, epochs = 30):
    #inputs = torch.tensor(inputs)
    #labels = torch.tensor(labels)
    
    
    model.freeze_feature_extractor()
    
    lowest_loss = 100000000
    loss_list = []
    
    #processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    #model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    inputs, validation_inputs, labels, validation_labels = train_test_split(inputs, labels, test_size = 0.2)


    optimizer = AdamW(model.parameters(),
                      lr = 5e-5,
                      eps = 1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = epochs)


    for batch in range(epochs):
        
        #inputs, labels = shuffle(inputs, labels)
        for input_, label_ in zip(inputs, labels):
        
            
            model.train()
            optimizer.zero_grad()
            
            outputs = model(torch.tensor(input_).unsqueeze(0), labels=torch.tensor(label_).unsqueeze(0))
            
            loss = outputs.loss
            print('first',loss)
            loss.backward
            optimizer.step()
            scheduler.step()
        
        
        
        ######################### now we evaluate
        summed_loss = 0
        for val_input_, val_label_ in zip(validation_inputs, validation_labels):
            model.eval()
            #with torch.no_grad():
            validation_output = model(torch.tensor(val_input_).unsqueeze(0), labels = torch.tensor(val_label_).unsqueeze(0))
            validation_loss = validation_output.loss
            print('second',validation_loss)
            if validation_loss < lowest_loss:
                lowest_loss = validation_loss
                #lowest_loss_model = copy.deepcopy(model)
                lowest_loss_model = model
            
            summed_loss += validation_loss
            loss_list.append(summed_loss)
            
    return model, lowest_loss, loss_list    
        
    
#lowest_loss_model, lowest_loss, loss_list  = run_hubert(raw_waveform_training_examples, labels)



#torch.save(lowest_loss_model, '/Users/bennicholl/Desktop/call_center_algos/models/wavbase')






        










