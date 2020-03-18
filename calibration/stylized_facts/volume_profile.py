import datetime as dt
import pandas as pd
import numpy as np

import logging as log

import matplotlib.pyplot as plt

from calibration.log_util import get_transacted_orders

MARKET_REPLAY_DATES = ['20190628', '20190627', '20190626', '20190625', '20190624',
                       '20190621', '20190620', '20190619', '20190618', '20190617',
                       '20190614', '20190613', '20190612', '20190611', '20190610',
                       '20190607', '20190606', '20190605', '20190604', '20190603']


def generate_volume_profile(executed_orders_dict, freq='1min'):
    """

    :param executed_orders_dict: dictionary containing the transacted orders
    :param freq: sampling freq (e.g. '1min')
    :return: volume profile Pandas Dataframe
    """
    vol_profile = pd.DataFrame(columns=MARKET_REPLAY_DATES[::-1],
                               index=pd.date_range(start='09:30:00', end='16:00:00', freq=freq).time)
    for i, date in enumerate(executed_orders_dict):
        df = executed_orders_dict[date][['TIMESTAMP', 'SIZE']]
        df = df.set_index('TIMESTAMP')
        df = df.groupby(level=0).sum()
        df = df.resample(freq).agg(np.sum)
        df.index = df.index.time
        vol_profile[date] = df.SIZE
    vol_profile = vol_profile.fillna(0.0)
    vol_profile['20d_mean'] = vol_profile.mean(axis=1)
    return vol_profile


def plot_transacted_volume(security, executed_orders_dict, freq='1min'):
    """

    :param security: Name of the stock/symbol
    :param executed_orders_dict: dictionary containing the transacted orders
    :param freq: sampling freq (e.g. '1min')
    :return: None
    """
    fig, ax = plt.subplots(nrows=4, ncols=5)
    fig.set_size_inches(30, 20)
    fig.suptitle(f"Transacted Volumes for {security} between {MARKET_REPLAY_DATES[-1]} and {MARKET_REPLAY_DATES[0]}", size=24, fontweight='bold')
    for i, date in enumerate(executed_orders_dict):
        df = generate_volume_profile(executed_orders_dict, freq)
        plt.subplot(4, 5, i + 1)
        plt.title(date, fontweight='bold')
        df.SIZE.plot()
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(f'transacted_volume_{security}_{freq}_bins.png', format='png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.03)


def plot_cumulative_volume(security, executed_orders_dict, freq='1min'):
    """

    :param security: Name of the stock/symbol
    :param executed_orders_dict: dictionary containing the transacted orders
    :param freq: sampling freq (e.g. '1min')
    :return: None
    """
    fig, ax = plt.subplots(nrows=4, ncols=5)
    fig.set_size_inches(30, 20)
    fig.suptitle(f"Cumulative Transacted Volumes for {security} between {MARKET_REPLAY_DATES[-1]} and {MARKET_REPLAY_DATES[0]}", size=24, fontweight='bold')
    for i, date in enumerate(executed_orders_dict):
        df = generate_volume_profile(executed_orders_dict, freq)
        plt.subplot(4, 5, i + 1)
        plt.title(date, fontweight='bold')
        df.SIZE.cumsum().plot()
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(f'cumulative_transacted_volume_{security}_{freq}_bins.png', format='png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.03)


def plot_volume_profile(security, vol_profile):
    """

    :param security: Name of the stock/symbol
    :param vol_profile: Volume Profile Pandas Dataframe
    :return:
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(20, 5)
    fig.suptitle(f"Volume Profile for {security}, raw data (left) and smoothed data (right) using a Gaussian Kernel "
                 f"Smoother", size=24, fontweight='bold')

    vol_profile['20d_mean'].plot(ax=ax[0])

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(f'volume_profile_{security}_{freq}_bins.png', format='png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.03)


if __name__ == "__main__":

    start_time = dt.datetime.now()
    log.basicConfig(level=log.INFO)

    system_name = '  ABIDES: Volume Profile Generation'
    log.info('=' * len(system_name))
    log.info(system_name)
    log.info('=' * len(system_name))
    log.info(' ')

    security = 'NKE'
    data_folder = f'/efs/data/get_real_data/marketreplay-logs/log/'
    freq = '1min'

    executed_orders_dict = dict.fromkeys(MARKET_REPLAY_DATES)
    for date in executed_orders_dict:
        executed_orders_dict[date] = get_transacted_orders(data_folder +
                                                           f'marketreplay_{security}_{date}/EXCHANGE_AGENT.bz2')
    plot_transacted_volume(security, executed_orders_dict, freq)
    plot_cumulative_volume(security, executed_orders_dict, freq)

    vol_profile = generate_volume_profile(executed_orders_dict, freq)

    plot_volume_profile(vol_profile)

    end_time = dt.datetime.now()
    log.info(f'Total time taken for the study: {end_time - start_time}')