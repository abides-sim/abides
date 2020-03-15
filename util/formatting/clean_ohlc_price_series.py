import pandas as pd
import sys
import numpy as np
import os
from random import sample
from dateutil.parser import parse

""" Clean OHLC WRDS data series into historical fundamental format."""

directory = sys.argv[1]
files = os.listdir(directory)
filename = os.path.join(directory, sample(files, 1)[0])

df = pd.read_pickle(filename, compression="bz2")
df.reset_index(level=-1, inplace=True)
df.level_1 = pd.to_datetime(df.level_1)

symbol = sample(df.index.unique().tolist(), 1)[0]
print(symbol, filename)
df = df[df.index==symbol]
df.set_index("level_1", inplace=True)

df = np.round((1000*df/df.iloc[0]).dropna())
df = (df["open"]*100).astype("int")
df.index = df.index - pd.DateOffset(year=2000, month=1, day=1)
pd.to_pickle(df, "clean.pkl".format(symbol, filename[:-4]))
