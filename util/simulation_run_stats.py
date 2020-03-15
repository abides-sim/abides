import argparse
import pandas as pd
from pandas.io.json import json_normalize
from glob import glob
import re
import os
from IPython.display import display, HTML
from matplotlib import pyplot as plt

"""

Scripts allows timing and memory usage analysis of ABIDES runs

Note: need to wrap ABIDES run with /usr/bin/time command for linux, or gtime for Mac (needs `brew install gnu-time`) 

"""


def make_numeric(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_run_statistics(lines):
    run_stats = dict()

    patterns = {
        'user_time (s)': r'User time \(seconds\): (\d+\.\d+)',
        'system_time (s)': r'System time \(seconds\): (\d+\.\d+)',
        'cpu_max_perc_usage': r'Percent of CPU this job got: (\d+)\%',
        'mem_max_usage (kB)': r'Maximum resident set size \(kbytes\): (\d+)',
        'messages_total': r'Event Queue elapsed: \d+ days \d{2}:\d{2}:\d{2}\.\d{6}, messages: (\d+), messages per second: \d+\.?\d*',
        'messages_per_second': r'Event Queue elapsed: \d+ days \d{2}:\d{2}:\d{2}\.\d{6}, messages: \d+, messages per second: (\d+\.?\d*)'
    }

    for line in lines:
        line = line.lstrip()
        for name, pattern in patterns.items():
            m = re.search(pattern, line)
            if m is None:
                continue
            else:
                # TODO: for messages currently only taking first value
                value = m.group(1)
                value = make_numeric(value)
                run_stats.update({name: value})

    return run_stats


def get_experiment_statistics(expt_path):
    expt_stat = []
    expt_name = os.path.basename(expt_path)

    for path in glob(f'{expt_path}/*__*.err'):
        pattern = r'.*__(\d+).err'
        m = re.search(pattern, path)
        param_value = m.group(1)
        param_value = make_numeric(param_value)
        with open(path, 'r') as f:
            run_output = f.readlines()
        run_stats = get_run_statistics(run_output)
        run_dict = {
            'run_path': path,
            'run_param_value': param_value,
            'run_stats': run_stats
        }
        expt_stat.append(run_dict)

    return expt_name, expt_stat


def dataframe_from_experiment_statistics(expt_name, expt_stat):
    # Normalize
    expt_df = json_normalize(expt_stat)
    expt_df.columns.name = expt_name
    # Clean
    expt_df = expt_df.dropna()
    expt_df = expt_df.sort_values(by='run_param_value')
    expt_df = expt_df.reset_index(drop=True)

    # Reorder columns
    cols = list(expt_df.columns)
    cols = [cols[0]] + cols[2:] + [cols[1]]
    expt_df = expt_df[cols]

    return expt_df


def dataframe_from_path(expt_path):
    expt_name, expt_stat = get_experiment_statistics(expt_path)
    expt_df = dataframe_from_experiment_statistics(expt_name, expt_stat)
    return expt_df


if __name__ == '__main__':

    log_files = glob('/home/ec2-user/efs/_abides/dev/dd/data_dump/*')

    no_mm_expt = []
    mm_expt = []

    for path in log_files:
        is_expt = True if 'mm' in path else False
        is_full_expt = is_expt and (True if '__' not in path else False)
        is_no_mm_expt = is_full_expt and (True if 'no_mm_dates' in path else False)
        is_mm_expt = is_full_expt and (True if 'with_mm' in path else False)
        if is_no_mm_expt: no_mm_expt.append(path)
        if is_mm_expt: mm_expt.append(path)

    expt_dfs = []

    for path in no_mm_expt:
        df = dataframe_from_path(path)
        expt_dfs.append(df)

    cols = expt_dfs[0].columns

    for col in cols[1:-1]:
        fig = plt.figure(figsize=(11, 8))
        for df in expt_dfs:
            x = df['run_param_value']
            y = df[col]
            plt.plot(x, y, label=df.columns.name)
        plt.legend()
        plt.title(col)
        plt.xlabel('num_agents')
        plt.ylabel(col)
        plt.show()
        fig.savefig(f'timings-plots/{col}.png', format='png', dpi=300, transparent=False, bbox_inches='tight',
                    pad_inches=0.03)


