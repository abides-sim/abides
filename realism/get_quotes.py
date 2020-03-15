import sys
import pandas as pd

def read_simulated_quotes (file):
    df = pd.read_pickle(file, compression='bz2')
    df['Timestamp'] = df.index

    # Keep only the last bid and last ask event at each timestamp.
    df = df.drop_duplicates(subset=['Timestamp','EventType'], keep='last')

    del df['Timestamp']

    df_bid = df[df['EventType'] == 'BEST_BID'].copy()
    df_ask = df[df['EventType'] == 'BEST_ASK'].copy()

    if len(df) <= 0:
        print("There appear to be no simulated quotes.")
        sys.exit()

    df_bid['BEST_BID'] = [b for s,b,bv in df_bid['Event'].str.split(',')]
    df_bid['BEST_BID_VOL'] = [bv for s,b,bv in df_bid['Event'].str.split(',')]
    df_ask['BEST_ASK'] = [a for s,a,av in df_ask['Event'].str.split(',')]
    df_ask['BEST_ASK_VOL'] = [av for s,a,av in df_ask['Event'].str.split(',')]

    df_bid['BEST_BID'] = df_bid['BEST_BID'].str.replace('$','').astype('float64')
    df_ask['BEST_ASK'] = df_ask['BEST_ASK'].str.replace('$','').astype('float64')

    df_bid['BEST_BID_VOL'] = df_bid['BEST_BID_VOL'].astype('float64')
    df_ask['BEST_ASK_VOL'] = df_ask['BEST_ASK_VOL'].astype('float64')

    df = df_bid.join(df_ask, how='outer', lsuffix='.bid', rsuffix='.ask')
    df['BEST_BID'] = df['BEST_BID'].ffill().bfill()
    df['BEST_ASK'] = df['BEST_ASK'].ffill().bfill()
    df['BEST_BID_VOL'] = df['BEST_BID_VOL'].ffill().bfill()
    df['BEST_ASK_VOL'] = df['BEST_ASK_VOL'].ffill().bfill()
    df['MIDPOINT'] = (df['BEST_BID'] + df['BEST_ASK']) / 2.0
    return df

sim_file = sys.argv[1]
df_sim = read_simulated_quotes(sim_file)
df_sim = df_sim.drop(["EventType.bid", "Event.bid", "EventType.ask", "Event.ask"], axis=1).resample("1T").ffill()
df_sim.to_csv("simulated_quotes.csv")

'''
import matplotlib.pyplot as plt
plt.plot(df_sim["BEST_BID"], color="g")
plt.plot(df_sim["BEST_ASK"], color="r")
plt.show()

plt.plot(df_sim["BEST_ASK"]-df_sim["BEST_BID"])
plt.show()
'''