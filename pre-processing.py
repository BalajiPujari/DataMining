import random
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None

directory_path = "D:/dm-project/aprioir"

df = pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print(df.shape)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df.drop_duplicates(inplace=True)



def map_to_malicious(label):
    return 'Malicious' if label != 'BENIGN' else label

df[' Label'] = df[' Label'].apply(map_to_malicious)


# now our df has many rows and columns, we will take sample of the data to show that the code works.

# out of all the data, we take 500 bening and 500 malicious rows, and combine them,

benign_rows = df[df[' Label'] == 'BENIGN'].sample(n=500, random_state=42) # taking 500 bening rows from the dataset
malicious_rows = df[df[' Label'] == 'Malicious'].sample(n=500, random_state=42) # taking 500 malicious rows from the dataset

final_df = pd.concat([benign_rows[1:51], malicious_rows[1:51]]) # comibing the 100 bening rows and 100 malicious rows to make sure that the model has equal class of bening and malicious
final_df.reset_index(drop=True, inplace=True)  # finally we took 100 of benign and 100 malicious samples, and stored it in final_df, we finally took 200 samples not 1000 !!!

df = final_df
df = df.sample(frac=1, random_state=random.seed())
# Apply the custom mapping function to df


df.to_csv('./sample1000.csv') # storing the 200 samples in sample1000.csv in the same directory.