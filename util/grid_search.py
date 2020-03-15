import argparse
import itertools
from util import numeric


def parse_cli():
    parser = argparse.ArgumentParser(description='Prints the Cartesian product of a group of lists.')
    parser.add_argument('-l', '--list', nargs='+', action='append',
                        help='Start of list', required=True, type=numeric)
    args = parser.parse_args()
    prod = itertools.product(*args.list)
    for items in prod:
        print(','.join(str(s) for s in items))


if __name__ == "__main__":
    parse_cli()
