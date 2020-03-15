import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def synthetic_volume_profile(date, freq):
    mkt_open = pd.to_datetime(date.date()) + pd.to_timedelta('09:30:00')
    mkt_close = pd.to_datetime(date.date()) + pd.to_timedelta('16:00:00')
    day_range = pd.date_range(mkt_open, mkt_close, freq=f'{freq}s')

    vol_profile = {}
    for t, x in zip(day_range, range(int(-len(day_range) / 2), int(len(day_range) / 2), 1)):
        vol_profile[t] = x ** 2 + 2 * x + 2

    factor = 1.0 / sum(vol_profile.values())
    vol_profile = {k: v * factor for k, v in vol_profile.items()}
    return vol_profile


class Metrics:
    """ Utility class to calculate metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, bps=False):
        self.symbol = symbol
        self.date = date
        self.orderbook_df = orderbook_df
        self.orders_df = orders_df
        self.bps = bps

    def mid_price(self):
        """Returns the mid price in the form of a pandas series (Orderbook specific)"""
        mid_price = (self.orderbook_df.ask_price_1 + self.orderbook_df.bid_price_1) / 2
        return mid_price * 10000 if self.bps else mid_price

    def spread(self, type='naive'):
        """Returns the spread in the form of a pandas series (Orderbook specific)"""
        spread = None
        if type == 'naive':
            spread = self.orderbook_df.ask_price_1 - self.orderbook_df.bid_price_1
        elif type == 'effective':
            num_price_levels = 5
            volume_weighted_ask = sum([self.orderbook_df[f'ask_price_{level}'] * self.orderbook_df[f'ask_size_{level}']
                                       for level in range(1, num_price_levels + 1)]) / \
                                  sum(self.orderbook_df[f'ask_size_{level}'] for level in range(1, num_price_levels + 1))
            volume_weighted_bid = sum([self.orderbook_df[f'bid_price_{level}'] * self.orderbook_df[f'bid_size_{level}']
                                       for level in range(1, num_price_levels + 1)]) / \
                                  sum(self.orderbook_df[f'bid_size_{level}'] for level in range(1, num_price_levels + 1))
            spread = volume_weighted_ask.ffill() - volume_weighted_bid.ffill()
        return spread * 10000 if self.bps else spread

    def volume_order_imbalance(self):
        """Returns the volume order imbalance in the form of a pandas series (Orderbook specific)"""
        return self.orderbook_df.bid_size_1 / (self.orderbook_df.ask_size_1 + self.orderbook_df.bid_size_1)

    def order_flow_imbalance(self, tick_size=0.01, sampling_freq='1ms'):
        """Returns the order flow imbalance in the form of a pandas series (Orderbook specific)"""
        ofi_df = self.orderbook_df[['ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1']].copy().reset_index()
        ofi_df['mid_price_change'] = ((ofi_df['bid_price_1'] + ofi_df['ask_price_1']) / 2).diff().div(tick_size)
        ofi_df['PB_prev'] = ofi_df['bid_price_1'].shift()
        ofi_df['SB_prev'] = ofi_df['bid_size_1'].shift()
        ofi_df['PA_prev'] = ofi_df['ask_price_1'].shift()
        ofi_df['SA_prev'] = ofi_df['ask_size_1'].shift()
        ofi_df = ofi_df.dropna()
        bid_geq = ofi_df['bid_price_1'] >= ofi_df['PB_prev']
        bid_leq = ofi_df['bid_price_1'] <= ofi_df['PB_prev']
        ask_geq = ofi_df['ask_price_1'] >= ofi_df['PA_prev']
        ask_leq = ofi_df['ask_price_1'] <= ofi_df['PA_prev']
        ofi_df['OFI'] = pd.Series(np.zeros(len(ofi_df)))
        ofi_df['OFI'].loc[bid_geq] += ofi_df['bid_size_1'][bid_geq]
        ofi_df['OFI'].loc[bid_leq] -= ofi_df['SB_prev'][bid_leq]
        ofi_df['OFI'].loc[ask_geq] += ofi_df['SA_prev'][ask_geq]
        ofi_df['OFI'].loc[ask_leq] -= ofi_df['ask_size_1'][ask_leq]
        ofi_df = ofi_df.set_index('index')
        ofi_df = ofi_df[['mid_price_change', 'OFI']].resample(sampling_freq).sum().dropna()
        return ofi_df.OFI

    @staticmethod
    def twap(executed_trades_df):
        return executed_trades_df['fill_price'].cumsum() / pd.Series(np.arange(1, len(executed_trades_df) + 1),
                                                                     executed_trades_df.index)
    @staticmethod
    def vwap(executed_trades_df):
        return (executed_trades_df['fill_price'] * executed_trades_df['quantity']).cumsum() / executed_trades_df[
            'quantity'].cumsum()

    @staticmethod
    def slippage(exchange_executed_trades_df, agent_executed_trades_df, trade_direction, benchmark='VWAP', bps=True):
        direction = 1 if trade_direction == 'BUY' else -1
        if benchmark == 'VWAP':
            slippage = direction * (Metrics.vwap(exchange_executed_trades_df) - Metrics.vwap(agent_executed_trades_df)) / \
                       Metrics.vwap(exchange_executed_trades_df)
        elif benchmark == 'TWAP':
            slippage = direction * (Metrics.twap(exchange_executed_trades_df) - Metrics.twap(agent_executed_trades_df)) / \
                       Metrics.twap(exchange_executed_trades_df)
        slippage = slippage[~slippage.isnull()]
        return slippage * 10000 if bps else slippage


class Plots:
    """ Utility class to plot metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, log_folder, bps=False):
        self.log_folder = log_folder
        self.metrics = Metrics(symbol, date, orderbook_df, orders_df, bps)

    def plot_mid_price(self, bps=False):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Mid Price ({'bps' if bps else '$'})", size=24, fontweight='bold')
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Mid Price", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.mid_price())
        plt.savefig(self.log_folder + '/' + 'Mid_Price.jpg', bbox_inches='tight')

    def plot_spread(self, bps=True):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Spread ({'bps' if bps else '$'})", size=24, fontweight='bold')
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Spread", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.spread(type='naive'), label='Naive Spread')
        ax.plot(self.metrics.spread(type='effective'), label='Effective Spread')
        ax.legend()
        plt.savefig(self.log_folder + '/' + 'Spread.jpg', bbox_inches='tight')

    def plot_volume_order_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Volume Order Imbalance", size=24, fontweight='bold')
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Volume Order Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.volume_order_imbalance())
        plt.savefig(self.log_folder + '/' + 'Volume_Order_Imbalance.jpg', bbox_inches='tight')

    def plot_order_flow_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Order Flow Imbalance", size=24, fontweight='bold')
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Order Flow Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.order_flow_imbalance())
        plt.savefig(self.log_folder + '/' + 'Order_Flow_Imbalance.jpg', bbox_inches='tight')