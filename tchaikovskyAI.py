#Essential
print('\t \t Welome to Tchaikovsky.AI \t \t\n')
import numpy as np
#MIDI Libraries
import mido
from mido import MidiFile, MidiTrack, Message
#ML Libraries
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
print(" Libraries Loaded!\n")
# To use of Bach's composition
#midi_input = MidiFile('bach_846_format0.mid')

# To use all of Tchaikovksy's compositions
import os
DATA_DIR = "."
MIDI_DATA_FOLDER = os.path.join(DATA_DIR,'tschai')
all_midi = []
for fname in os.listdir(MIDI_DATA_FOLDER):
    if fname[0] =='.':
        continue
    current_midi_name = os.path.join(MIDI_DATA_FOLDER,str(fname))
    curr_midi = MidiFile(current_midi_name)
    all_midi+=curr_midi
midi_input = all_midi
print("Loaded MIDI Data!\n")

#PVTdata = [Pitch Velocity Time]
PVTdata = []
temp=[]
time = float(0)
last_time = float(0)
for msg in midi_input:
    time+=msg.time
    if (not(msg.is_meta)) and msg.channel==0 and msg.type=='note_on':
        data = msg.bytes()
        data = data[1:3]
#         data.append(float(msg.time))
#         temp.append(data)
        data.append(time-last_time)
        temp.append(time-last_time)
        last_time = time
        PVTdata.append(data)
print("MIDI Data Processed\n")

#Scale all values to  0-1 range
max_time_note = max(temp)
for note in PVTdata:
    #Piano has 88 keys
    note[0] = (note[0])/88
    #Velocity has a range of 0-127
    note[1] = (note[1])/127
    #Divide Time by the time of the longest note
    note[2] = (note[2])/max_time_note
print("MIDI Data Scaled Down")

# Prepare your data for the network

#seq_len gives indicates how many notes in the past you should look
seq_len = 500
num_of_seq = len(PVTdata) - seq_len


X = np.zeros((num_of_seq,seq_len,3),dtype=float)
y = np.zeros((num_of_seq,3),dtype=float)

for i in range(num_of_seq):
    X[i,:,:]=PVTdata[i:i+seq_len]
    y[i,:] = PVTdata[i+seq_len]
print("Training Matrices Prepared\n")

# LSTM Architecture

print('\n\tBuild Model...\n')

model = Sequential()
model.add(LSTM(128,input_shape=(seq_len,3),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128,input_shape=(seq_len,3),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128,input_shape=(seq_len,3),return_sequences=False))
model.add(Dense(3, activation='linear'))

rms = RMSprop(lr=1e-3)
model.compile(loss='mse',optimizer=rms)
print("\t Model Compiled!\n")

print("\n Training Begun ...\t \n")
model.fit(X,y,batch_size=50,epochs=30,verbose=1)
print("\t Training Complete!..\t\n")

print(" Generating Predicted Tune!\n")
predicted_tune = []
x=[]
#start_tune_index = np.random.randint(num_of_seq)
start_tune_index = 0
x = PVTdata[start_tune_index:start_tune_index+seq_len]

x=np.expand_dims(x,axis=0)

# Generate 300 notes from our model(excluding start_tune notes)
predicted_length = 300

for i in range(predicted_length):
    curr_pred_note = model.predict(x)
    print(curr_pred_note)
    x = np.squeeze(x)
    x = np.concatenate((x,curr_pred_note))
    x = x[1:]
    x = np.expand_dims(x,axis=0)
    curr_pred_note = np.squeeze(curr_pred_note)
    predicted_tune.append(curr_pred_note)
print("Predicted Tune Generated!\n")

print("Scaling up Generated MIDI tune!\n")
# Scale back the midi outputs
for note in predicted_tune:
    note[0] = note[0]*88
    note[1] = note[1]*127
    note[2] = note[2]*max_time_note

    if(note[0]<0):
        note[0] = 0
    if(note[0]>88):
        note[0] = 88
    if(note[1]<0):
        note[1] = 0
    if(note[1]>127):
        note[1] = 127
    if(note[2]<0):
        note[2] = 0



## Recreate a Midi Track from your predicted_tune
print("Creating the .MIDI file generated")
midi_output = MidiFile()
track = MidiTrack()

for note in predicted_tune:
    # 144 is the note_on for Channel 1
    note = np.insert(note,0,144)
    bytes = note.astype(int)
    print(note)
    message = Message.from_bytes(bytes[0:3])
    t = int(note[3]/0.001025)
    message.time = t
    track.append(message)

midi_output.tracks.append(track)
midi_output.save('tchAIkovsky.mid')

print("\n  MIDI file generated through AI! :) ")
