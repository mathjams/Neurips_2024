# -*- coding: utf-8 -*-

! apt-get install git

from google.colab import drive
drive.mount('/content/drive')

import statistics 
import scipy.stats as stats

ASD=[]
TD=[]

# Analyzes E/H errors
for i in range(1, 20):
  inputtd, outputtd, max_lentd=plainsequences('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=plainsequences('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD, TD)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
meanASD = statistics.mean(ASD)
meanTD = statistics.mean(TD)
print(meanASD, meanTD)

ASD=[]
TD=[]

# Analyzes WL errors
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

# Analyzes CO Errors

ASD=[]
TD=[] 
for i in range(1, 20):
  inputtd, outputtd, max_lentd=geneyehandoverlaps('TD')
  modeltd, encoder_modeltd, decoder_modeltd, final_losstd = LSTM_model(inputtd, outputtd, max_lentd, 2, 30)
  inputasd, outputasd, max_lenasd=geneyehandoverlaps('ASD')
  modelasd, encoder_modelasd, decoder_modelasd, final_lossasd=LSTM_model(inputasd, outputasd, max_lenasd, 2, 30)
  ASD.append(final_lossasd)
  TD.append(final_losstd)

t_statistic, p_value = stats.ttest_ind(ASD1, TD1)

print(f"t-statistic: {t_statistic}, p-value: {p_value}")
meanASD = statistics.mean(ASD)
meanTD = statistics.mean(TD)
print(meanASD, meanTD)


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
