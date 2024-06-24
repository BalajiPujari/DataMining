import pandas as pd
from pyarc.data_structures import TransactionDB
import pickle

with open('Downloads/classifier.pkl','wb') as file :
    classifier = pickle.load(file)
with open('Downloads/selected_features.pkl','wb') as file :
    selected_features = pickle.load(file)


data = pd.read_csv('./test.csv')

test = data[selected_features]
actual_labels = data[' Label']

test_txns = TransactionDB.from_DataFrame(test)

y_pred = classifier.predict_all(test_txns)

compare_df = pd.DataFrame({'True Label (y_test)': actual_labels, 'Predicted Label (y_pred)': y_pred})
compare_df