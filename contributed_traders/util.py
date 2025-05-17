#!/usr/bin/env python3
from pathlib import Path


def get_file(fname):
    return Path(__file__).resolve().parent / fname
