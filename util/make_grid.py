import argparse
import numpy as np
import sys

SCALE_OPTIONS = ['log', 'linear']

def check_both_int(a, b):
    if a.is_integer() and b.is_integer():
        return True
    else:
        return False

def process_args(args):
    """ Prints grid to standard error. """

    min_val, max_val, num_points, scale = args.min, args.max, args.num_points, args.scale
    
    if scale == 'linear':
        dtype = int if check_both_int(min_val, max_val) else float
        grid = np.linspace(min_val, max_val, num=num_points, dtype=dtype)
    elif scale == 'log':
        grid = np.logspace(min_val, max_val, num=num_points)
    else:
        raise ValueError(f"Option not in {SCALE_OPTIONS}")

    grid = np.unique(grid)

    for elem in grid:
        sys.stdout.write(str(elem)+'\n')


def parse_cli():
    parser = argparse.ArgumentParser(description='Generates a 1D grid of points from min to max')
    parser.add_argument('--min', type=float, required=True,
                        help='Minimum value.')
    parser.add_argument('--max', type=float, required=True,
                        help='Maximum value.')
    parser.add_argument('--num-points', type=int, required=True,
                        help='Number of grid points.')
    parser.add_argument('--scale', type=str, default='linear', choices=SCALE_OPTIONS,
                        help='Scaling of grid points. Note if "log" then min and max are interpreted '
                        'as 10 ** min and 10 ** max respectively.')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli()
    process_args(args)