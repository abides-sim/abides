from glob import glob
import pandas as pd
import pickle
from pprint import pprint
import argparse


def main(path_glob, out_filepath):
    """ Computes aggregated statistics (median) across multiple market impact experiments and saves the result.

        :param path_glob: Glob pattern for paths to cached experiment files to be aggregated. Note by `cached` the output of the function impact_single_day_pov.prep_data is meant.
        :param out_filepath: path to file storing aggregated statistics (.csv extension)

        :type path_glob: str
        :type out_filepath: str

    """
    impact_stats = []

    for path in glob(path_glob):

        print(f'Processing file {path}')

        try:
            with open(path, 'rb') as f:
                consolidated = pickle.load(f)

        except pickle.UnpicklingError:
            continue

        for elem in consolidated:
            stats_dict = elem['impact_statistics']
            pov = elem['pov']
            stats_dict.update({
                'pov': pov,
            })
            impact_stats.append(stats_dict)

    stats_df_aggregate = pd.DataFrame(impact_stats)
    median = stats_df_aggregate.groupby(['pov']).apply(pd.DataFrame.median).drop(columns=['pov'])
    print(f'Aggregated statistics (median) for glob {path_glob}:')
    print(median)
    median.to_csv(out_filepath, index=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLI utility for aggregating statistics of multi-day POV experiments.')

    parser.add_argument('path_glob',
                        help='Glob pattern for paths to cached experiment files to be aggregated. Note by `cached` the output of the function impact_single_day_pov.prep_data is meant. ',
                        type=str)
    parser.add_argument('out_file',
                        help='Path to csv output file.')

    args, remaining_args = parser.parse_known_args()

    path_glob = args.path_glob
    out_filepath = args.out_file

    main(path_glob, out_filepath)
