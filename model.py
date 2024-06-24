import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import (
    top_rules,
    createCARs,
    M1Algorithm,
)
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None

directory_path = "D:/dm-project/aprioir/" # paste the root folder of dataset which contains the cicids2017 dataset


#loading datasets from the above path
df = pd.read_csv(directory_path+"sample1000.csv")
"""df2=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
df5=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df7=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
df8=pd.read_csv(directory_path+"/cicids2017/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")

df = pd.concat([df,df2])
del df2
df = pd.concat([df,df3])
del df3
df = pd.concat([df,df4])
del df4
df = pd.concat([df,df5])
del df5
df = pd.concat([df,df6])
del df6
df = pd.concat([df,df7])
del df7
df = pd.concat([df,df8])
del df8"""

print('datasets loaded')
#--------------------- Data Pre processing ------------------------#

def map_to_malicious(label):
    return 'Malicious' if label != 'BENIGN' else label

df[' Label'] = df[' Label'].apply(map_to_malicious)



#Data downsampling is bbeing performed
def downsample_benign_class(df, downsample_size):
    # Get the value counts of the 'Label' column
    if df.shape[0] < downsample_size:
        return df


    value_counts = df[' Label'].value_counts()
    print(value_counts)
    benign_count = value_counts['BENIGN']
    print(benign_count)

    df_benign_downsampled = df[df[' Label'] == 'BENIGN'].sample(n=downsample_size, random_state=42)
    df_other = df[df[' Label'] != 'BENIGN']
    df_downsampled = pd.concat([df_benign_downsampled, df_other], ignore_index=True)
    return df_downsampled


downsample_size = 1000 # for testing purpose it was taken as 1000
df_downsampled = downsample_benign_class(df, downsample_size)

df = df_downsampled

del df_downsampled

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df = df.sample(frac=1, random_state=42)

print('pre processing done')


# ----------------------- Random Forest Classifier ------------------------ #

X = df.drop(' Label', axis=1)
y = df[' Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

rf_feature_importance = rf_classifier.feature_importances_

feature_importance_dict = {feature: importance for feature, importance in zip(df.columns, rf_feature_importance)}
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names, sorted_importances = zip(*sorted_feature_importance)


#plots will be generated here for the downsampled data
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()


print('RF classifer done')
# --------------------- Classification Based on Association Rules -----------------------#

def classification(train, test):
    txns_train = TransactionDB.from_DataFrame(train)
    txns_test = TransactionDB.from_DataFrame(test)

    rules = top_rules(txns_train.string_representation)
    cars = createCARs(rules)

    classifier = M1Algorithm(cars, txns_train).build()
    accuracy = classifier.test_transactions(txns_test)

    return accuracy, classifier


feature_importance_list = [0.01, 0.008, 0.006]
best_accuracy = 0
best_classifer = None
best_top_features = None
best_k = 0

for k in feature_importance_list:

    selected_features = [feature for feature, feature_importance in feature_importance_dict.items() if
                         feature_importance >= k]
    X_train[' Label'] = y_train
    accuracy, classifier = classification(X_train[selected_features], X_test[selected_features])

    if accuracy > best_accuracy:
        best_classifier = classifier
        best_accuracy = accuracy
        best_top_features = selected_features
        best_k = k


print('Best accuracy', best_accuracy)
print('Feature Importance = ',best_k)
print('Number of features chosen = ', len(selected_features))

with open('./classifier.pkl','wb') as file :
    pickle.dump(best_classifier,file);

with open('./selected_features.pkl','wb') as file :
    pickle.dump(best_top_features,file);

data = pd.read_csv('./sample1000.csv')

test = data[1:20][selected_features]
actual_labels = data[1:20][' Label'].tolist()

test_txns = TransactionDB.from_DataFrame(test)

y_pred = classifier.predict_all(test_txns)

print(actual_labels)
print(y_pred)

def get_best_threshold(y_pred):
    y_pred_int = [int(float(value)) for value in y_pred]
    min_value = min(y_pred_int)
    max_value = max(y_pred_int)
    best_accuracy = 0
    best_threshold = None
    best_preds = None
    for threshold in range(min_value, max_value):
        predictions = []
        for value in y_pred_int:
            if value >= threshold:
                predictions.append('Malicious')
            else:
                predictions.append('BENIGN')
        accuracy = get_accuracy(predictions, actual_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_preds = predictions
    return best_accuracy, best_threshold, best_preds

def get_accuracy(predictions, actual_labels):
    correct = 0
    total = len(predictions)
    for i in range(total):
        if i < len(actual_labels):
            if predictions[i] == actual_labels[i]:
                correct += 1
    return correct / total



best_pred_accuracy, best_threshold, predictions = get_best_threshold(y_pred)

print("Best accuracy:", best_pred_accuracy)
print("Best threshold:", best_threshold)


compare_df = pd.DataFrame({'True Label (y_test)': actual_labels, 'Predicted Label (y_pred)': predictions})
print(compare_df)


