import argparse
import itertools
from util import numeric
import random


def generate_random_tuples(list_of_lists, num_samples, seed):
    random.seed(a=seed)
    for n in range(num_samples):
        items = [random.choice(l) for l in list_of_lists]
        print(','.join(str(s) for s in items))


def parse_cli():
    parser = argparse.ArgumentParser(description='Prints a random selection of the Cartesian product of a group of lists.')
    parser.add_argument('-l', '--list', nargs='+', action='append',
                        help='Start of list', required=True, type=numeric)
    parser.add_argument('-n', '--num-samples', type=int, required=True, help='Number of tuples to print.')
    parser.add_argument('-s', '--random-seed', type=int, default=12345, help='Random seed.')

    args = parser.parse_args()
    generate_random_tuples(args.list, args.num_samples, args.random_seed)


if __name__ == "__main__":
    parse_cli()
