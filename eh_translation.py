# -*- coding: utf-8 -*-

! apt-get install git

from google.colab import drive
drive.mount('/content/drive')


"""**handeyetogether** takes a hand and eye file, and take a certain time window. Then, it splits the entire series of time and splits it up into these windows. After, it looks at what position the most recent fixations for the eyes/hands are and fills in the interval using this."""

import pandas as pd
import numpy as np

def handeyetogether(urlhand, urleye, windowlength):
  eye_data=pd.read_excel(urleye)
  hand_data=pd.read_excel(urlhand)
  eye=eye_data.copy()
  hand=hand_data.copy()
  np.array(eye)
  np.array(hand)
  starttime= min(np.min(hand['start']), np.min(eye['start']))
  hand['start']+=-starttime
  hand['end']+=-starttime
  eye['start']+=-starttime
  eye['end']+=-starttime
  endtime=max(np.max(hand['start']), np.max(eye['start']))
  intervals=int(endtime//windowlength)
  currenttime=0
  eyearray=np.zeros((intervals, 2))
  handarray=np.zeros((intervals, 2))
  for i in range(eye.shape[0]-1):
    for j in range(int(eye.loc[i].start//windowlength), int(eye.loc[i+1].start//windowlength)):
      eyearray[j][0]=eye.loc[i].x
      eyearray[j][1]=eye.loc[i].y
  for k in range(int(eye.loc[eye.shape[0]-1].start//windowlength), intervals):
    eyearray[k][0]=eye.loc[eye.shape[0]-1].x
    eyearray[k][1]=eye.loc[eye.shape[0]-1].y
  for l in range(hand.shape[0]-1):
    for m in range(int(hand.loc[l].start//windowlength), int(hand.loc[l+1].start//windowlength)):
      handarray[m][0]=hand.loc[l].x
      handarray[m][1]=hand.loc[l].y
  for n in range(int(hand.loc[hand.shape[0]-1].start//windowlength), intervals):
    handarray[n][0]=hand.loc[hand.shape[0]-1].x
    handarray[n][1]=hand.loc[hand.shape[0]-1].y
  return eyearray, handarray




"""**generate_sequences_together** combines the data from all files to be used as training data for the LSTM

"""

import pandas as pd
import numpy as np

def generate_sequences_together(user_type,win_len):
  resulteye=[]
  resulthand=[]
  maxlen=0
  eye_basic_url='/content/drive/My Drive/data_set/Eye_'
  hand_basic_url='/content/drive/My Drive/data_set/Hand_'
  if (user_type=='ASD'):
    numOfUser=9
    eye_basic_url+="ASD_"
    hand_basic_url+='ASD_'
  else:
    eye_basic_url+="TD_"
    hand_basic_url+='TD_'
    numOfUser=17
  for i in range(1, numOfUser+1):
    for j in range(0,2):
      c_eye_url=eye_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      c_hand_url=hand_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      #asd_eye_data=pd.DataFrame()
      try:
        dataeye, datahand =handeyetogether(c_hand_url, c_eye_url, win_len)
        resulteye.append(dataeye)
        resulthand.append(datahand)
        if len(dataeye)>maxlen or len(datahand)>maxlen:
          maxlen=max(len(dataeye), len(datahand))
      except IOError:
        print("")
  return resulteye, resulthand, maxlen

"""simply putting the x and y into the LSTM"""

import pandas as pd
import numpy as np

def plainsequences(user_type):
  resulteye=[]
  resulthand=[]
  maxlen=0
  eye_basic_url='/content/drive/My Drive/data_set/Eye_'
  hand_basic_url='/content/drive/My Drive/data_set/Hand_'
  if (user_type=='ASD'):
    numOfUser=9
    eye_basic_url+="ASD_"
    hand_basic_url+='ASD_'
  else:
    eye_basic_url+="TD_"
    hand_basic_url+='TD_'
    numOfUser=17
  for i in range(1, numOfUser+1):
    for j in range(0,2):
      c_eye_url=eye_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      c_hand_url=hand_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      #asd_eye_data=pd.DataFrame()
      try:
        asd_eye_data=pd.read_excel(c_eye_url)
        asd_hand_data=pd.read_excel(c_hand_url)
        resulteye.append(asd_eye_data[['x','y']].to_numpy())
        resulthand.append(asd_hand_data[['x','y']].to_numpy())
        c_max_length=max(asd_eye_data.shape[0], asd_hand_data.shape[0])
        if c_max_length>maxlen:
          maxlen=c_max_length
      except IOError:
        print("")
  return resulteye, resulthand, maxlen

"""this code is used to make a graph of the errors for certain window sizes"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

X=[]
Ytd=[]
Yasd=[]

# Show plot

for i in range(1, 20):
  windowlength=1000*i
  inputtd, outputtd, max_lentd=generate_sequences_together('TD',windowlength)
  tderrors=[]
  asderrors=[]
  for i in range(5):
    modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
    tderrors.append(final_losstd)
  inputasd, outputasd, max_lenasd=generate_sequences_together('ASD',windowlength)
  for i in range(5):
    modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
    asderrors.append(final_lossasd)
  X.append(windowlength)
  Ytd.append(sum(tderrors)/5)
  Yasd.append(sum(asderrors)/5)
# Create DataFrame

dataTD = pd.DataFrame({'x': X, 'y': Ytd})
sns.scatterplot(data=dataTD, x='x', y='y', color='blue')
dataASD = pd.DataFrame({'x': X, 'y': Yasd})
sns.scatterplot(data=dataASD, x='x', y='y', color='red')

# Add labels, title, and legend
plt.xlabel('Windowlength')
plt.ylabel('Prediction error')
plt.title('Windowlength vs Error')
plt.legend()
plt.show()

"""**eyehandoverlaps** only looks at where the eye and hand fixations overlap"""

def eyehandoverlaps(urleye, urlhand):
  eye_data=pd.read_excel(urleye)
  hand_data=pd.read_excel(urlhand)
  eye=eye_data.copy()
  hand=hand_data.copy()
  np.array(eye)
  np.array(hand)
  starttime= min(np.min(hand['start']), np.min(eye['start']))
  hand['start']+=-starttime
  hand['end']+=-starttime
  eye['start']+=-starttime
  eye['end']+=-starttime
  handarray=np.zeros((0,2))
  eyearray=np.zeros((0,2))
  for i in range(eye.shape[0]):
    for j in range(hand.shape[0]):
      if max(hand.loc[j].start, eye.loc[i].start)<=min(hand.loc[j].end, eye.loc[i].end):
 #       handeye=np.append(handeye, [[eye.loc[i].x, eye.loc[i].y, hand.loc[j].x, hand.loc[j].y, max(hand.loc[j].start, eye.loc[i].start), min(hand.loc[j].end, eye.loc[i].end) ]], axis=0)
        eyearray=np.append(eyearray, [[eye.loc[i].x, eye.loc[i].y]], axis=0)
        handarray=np.append(handarray, [[hand.loc[j].x, hand.loc[j].y]], axis=0)
  return eyearray, handarray
eyehandoverlaps('/content/drive/My Drive/data_set/Eye_TD_U1_Active_1.xlsx', '/content/drive/My Drive/data_set/Hand_TD_U1_Active_1.xlsx')

"""**generateeyehandoverlaps** puts all the overlap data into one array to feed into the LSTM"""

import pandas as pd
import numpy as np

def geneyehandoverlaps(user_type):
  resulteye=[]
  resulthand=[]
  maxlen=0
  eye_basic_url='/content/drive/My Drive/data_set/Eye_'
  hand_basic_url='/content/drive/My Drive/data_set/Hand_'
  if (user_type=='ASD'):
    numOfUser=9
    eye_basic_url+="ASD_"
    hand_basic_url+='ASD_'
  else:
    eye_basic_url+="TD_"
    hand_basic_url+='TD_'
    numOfUser=17
  for i in range(1, numOfUser+1):
    for j in range(0,2):
      c_eye_url=eye_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      c_hand_url=hand_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      #asd_eye_data=pd.DataFrame()
      try:
        dataeye, datahand = eyehandoverlaps(c_eye_url, c_hand_url)
        resulteye.append(dataeye)
        resulthand.append(datahand)
        if len(dataeye)>maxlen or len(datahand)>maxlen:
          maxlen=max(len(dataeye), len(datahand))
      except IOError:
        print("")
  return resulteye, resulthand, maxlen


"""**randeyehandoverlaps** takes random windows of 1/2 the length to be used to train"""
import scipy.stats as stats

ASD=[]
TD=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=plainsequences('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=plainsequences('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")

import scipy.stats as stats

ASD=[]
TD=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=generate_sequences_together('TD', 12000)
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=generate_sequences_together('ASD', 18000)
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
meanASD = statistics.mean(ASD)
meanTD = statistics.mean(TD)
print(meanASD, meanTD)

import statistics

import scipy.stats as stats

ASD=[]
TD=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=generate_sequences_together('TD', 12000)
  modeltd, final_losstd = GRU_model(inputtd, outputtd, max_lentd, 2)
  inputasd, outputasd, max_lenasd=generate_sequences_together('ASD', 17000)
  modelasd, final_lossasd=GRU_model(inputasd, outputasd, max_lenasd, 2)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
meanASD = statistics.mean(ASD)
meanTD = statistics.mean(TD)
print(meanASD, meanTD)

std_asd = statistics.pstdev(ASD)
std_td = statistics.pstdev(TD)
print(std_asd, std_td)

import scipy.stats as stats

ASD=[]
TD=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=generate_sequences_together('TD', 12000)
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=generate_sequences_together('ASD', 12000)
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")

import scipy.stats as stats

ASD=[]
TD=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=plainsequences('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=plainsequences('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")

import statistics
meanASD = statistics.mean(ASD)
meanTD = statistics.mean(TD)
print(meanASD, meanTD)

std_asd = statistics.pstdev(ASD)
std_td = statistics.pstdev(TD)
print(std_asd, std_td)

print(ASD, TD)

import scipy.stats as stats

ASD1=[]
TD1=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=geneyehandoverlaps('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=geneyehandoverlaps('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD1.append(final_lossasd)
  TD1.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print(ASD1, TD1)
import statistics
meanASD1 = statistics.mean(ASD1)
meanTD1 = statistics.mean(TD1)
print(meanASD1, meanTD1)

std_asd1 = statistics.pstdev(ASD1)
std_td1 = statistics.pstdev(TD1)
print(std_asd1, std_td1)

import scipy.stats as stats

ASD1=[]
TD1=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=geneyehandoverlaps('TD')
  modeltd, final_losstd = GRU_model(inputtd, outputtd, max_lentd, 2)
  inputasd, outputasd, max_lenasd=geneyehandoverlaps('ASD')
  modelasd, final_lossasd=GRU_model(inputasd, outputasd, max_lenasd, 2)
  ASD1.append(final_lossasd)
  TD1.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print(ASD1, TD1)
import statistics
meanASD1 = statistics.mean(ASD1)
meanTD1 = statistics.mean(TD1)
print(meanASD1, meanTD1)

std_asd1 = statistics.pstdev(ASD1)
std_td1 = statistics.pstdev(TD1)
print(std_asd1, std_td1)

import scipy.stats as stats

ASD1=[]
TD1=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=plainsequences('TD')
  modeltd, final_losstd = GRU_model(inputtd, outputtd, max_lentd, 2)
  inputasd, outputasd, max_lenasd=plainsequences('ASD')
  modelasd, final_lossasd=GRU_model(inputasd, outputasd, max_lenasd, 2)
  ASD1.append(final_lossasd)
  TD1.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print(ASD1, TD1)
import statistics
meanASD1 = statistics.mean(ASD1)
meanTD1 = statistics.mean(TD1)
print(meanASD1, meanTD1)

std_asd1 = statistics.pstdev(ASD1)
std_td1 = statistics.pstdev(TD1)
print(std_asd1, std_td1)

import scipy.stats as stats

ASD1=[]
TD1=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=geneyehandoverlaps('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=geneyehandoverlaps('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD1.append(final_lossasd)
  TD1.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print(ASD1, TD1)
import statistics
meanASD1 = statistics.mean(ASD1)
meanTD1 = statistics.mean(TD1)
print(meanASD1, meanTD1)

std_asd1 = statistics.pstdev(ASD1)
std_td1 = statistics.pstdev(TD1)
print(std_asd1, std_td1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Embedding, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
def LSTM_model2(input_data, output_data, max_len, number_of_features):
  max_sequence_len = max_len
  input_data_padded = pad_sequences(input_data, maxlen=max_sequence_len, padding='post', dtype='float32', value=0)
  output_data_padded = pad_sequences(output_data, maxlen=max_sequence_len, padding='post', dtype='float32', value=0)

  decoder_target_data = np.roll(output_data_padded, shift=-1, axis=0)
  decoder_target_data[:, -1, :] = 0  # Reset last time step to 0 (zero padding)
  latent_dim = 50

  # Encoder
  encoder_inputs = Input(shape=(None, number_of_features), name='encoder_inputs')
  encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_LSTM')
  encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
  encoder_states = [state_h, state_c]

# Decoder
  decoder_inputs = Input(shape=(None, number_of_features))
  decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
  decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
  decoder_dense = Dense(number_of_features, activation='linear')  # Using linear activation function
  decoder_outputs = decoder_dense(decoder_outputs)

  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


  encoder_model = Model(encoder_inputs, encoder_states)

  decoder_state_input_h = Input(shape=(latent_dim,))
  decoder_state_input_c = Input(shape=(latent_dim,))
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

  decoder_outputs, state_h, state_c = decoder_lstm(
  decoder_inputs, initial_state=decoder_states_inputs)
  decoder_states = [state_h, state_c]
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

  model.compile(optimizer='adam', loss='mse')
 # print(model.summary())
  result=model.fit([input_data_padded, output_data_padded], decoder_target_data, batch_size=50, epochs=30, validation_split=0.2)
  final_loss = result.history['val_loss'][-1]
  print(final_loss)
  return model, encoder_model, decoder_model, final_loss
#  print(result.history)
#  return model, encoder_model, decoder_model

input, output, maxlen=plainsequences('TD')
LSTM_model2(input, output, maxlen, 2)



input, output, maxlen=plainsequences('TD')
LSTM_model(input, output, maxlen, 2, 30)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, TimeDistributed, Embedding, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
def GRU_model(input_data, output_data, max_len, number_of_features):
  max_sequence_len = max_len
  input_data_padded = pad_sequences(input_data, maxlen=max_sequence_len, padding='post', dtype='float32', value=0)
  output_data_padded = pad_sequences(output_data, maxlen=max_sequence_len, padding='post', dtype='float32', value=0)

  decoder_target_data = np.roll(output_data_padded, shift=-1, axis=0)
  decoder_target_data[:, -1, :] = 0  # Reset last time step to 0 (zero padding)
  latent_dim = 50

  # Encoder
  encoder_inputs = Input(shape=(None, number_of_features), name='encoder_inputs')
  encoder_gru = GRU(latent_dim, return_state=True)
  encoder_outputs, state_h = encoder_gru(encoder_inputs)
  encoder_states = state_h

# Decoder
  decoder_inputs = Input(shape=(None, number_of_features))
  decoder_gru = GRU(latent_dim, return_sequences=True)
  decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_states)
  decoder_dense = Dense(number_of_features, activation='linear')  # Using linear activation function
  decoder_outputs = decoder_dense(decoder_outputs)

  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

  model.compile(optimizer='adam', loss='mse')
 # print(model.summary())
 # print(model.summary())
  result=model.fit([input_data_padded, output_data_padded], decoder_target_data, batch_size=50, epochs=30, validation_split=0.2)
  final_loss = result.history['val_loss'][-1]
  print(final_loss)
  return model, final_loss

input, output, maxlen=plainsequences('TD')
GRU_model(input, output, maxlen, 2)

input, output, maxlen=plainsequences('ASD')
GRU_model(input, output, maxlen, 2)

input, output, maxlen=geneyehandoverlaps('ASD')
GRU_model(input, output, maxlen, 2)

input, output, maxlen=geneyehandoverlaps('TD')
GRU_model(input, output, maxlen, 2)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

X=[]
Ytd=[]
Yasd=[]

# Show plot

for i in range(1, 20):
  windowlength=1000*i
  inputtd, outputtd, max_lentd=generate_sequences_together('TD',windowlength)
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = GRU_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=generate_sequences_together('ASD',windowlength)
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=GRU_model(inputasd, outputasd, max_lenasd, 2, 30)
  X.append(windowlength)
  Ytd.append(final_losstd)
  Yasd.append(final_lossasd)
# Create DataFrame

dataTD = pd.DataFrame({'x': X, 'y': Ytd})
sns.scatterplot(data=dataTD, x='x', y='y', color='blue')
dataASD = pd.DataFrame({'x': X, 'y': Yasd})
sns.scatterplot(data=dataASD, x='x', y='y', color='red')

# Add labels, title, and legend
plt.xlabel('windowlength')
plt.ylabel('validation error')
plt.title('lenth vs validation')
plt.legend()
plt.show()

import scipy.stats as stats

ASD1=[]
TD1=[]


for i in range(1, 20):
  inputtd, outputtd, max_lentd=geneyehandoverlaps('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = GRU_model(inputtd, outputtd, max_lentd, 2)
  inputasd, outputasd, max_lenasd=geneyehandoverlaps('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=GRU_model(inputasd, outputasd, max_lenasd, 2)
  ASD1.append(final_lossasd)
  TD1.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print(ASD1, TD1)
import statistics
meanASD1 = statistics.mean(ASD1)
meanTD1 = statistics.mean(TD1)
print(meanASD1, meanTD1)

std_asd1 = statistics.pstdev(ASD1)
std_td1 = statistics.pstdev(TD1)
print(std_asd1, std_td1)
