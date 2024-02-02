import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/filip/baseline/data/input/train_data/train_62classes_new.csv')

_, df = train_test_split(df, test_size=0.2)

df.to_csv('/home/filip/baseline/data/input/train_data/train_62classes_new_20pc.csv', index=False)