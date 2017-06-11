#!/usr/bin/env python3
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
import json
import sys
from pathlib import Path
from argparse import ArgumentParser

from typing import Iterable, IO, Dict, Any, DefaultDict


class ObjectFileNamer:
    def __init__(self) -> None:
        self.counters = DefaultDict[str, int](int)

    def __call__(self, path: Path) -> str:
        stem = path.stem
        self.counters[stem] += 1
        return f'{stem}.{self.counters[stem]}.o'


def configure(sources: Iterable[Path],
              cmd: str,
              blddir: Path,
              out: IO[str] = sys.stdout) -> None:
    namer = ObjectFileNamer()
    args = cmd.split()
    fortran_tasks = {str(path): {
        'source': str(path),
        'args': args + [str(blddir/namer(path))]
    } for path in sources}
    json.dump(fortran_tasks, out)


def parse_cli() -> Dict[str, Any]:
    """Handle the command-line interface."""
    blddir = '.'
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('sources', metavar='FILE', nargs='*', type=Path, help='Fortran source files')
    arg('--blddir', default=blddir, type=Path,
        help=f'build directory [default: {blddir}]')
    arg('--cmd', required=True,
        help='compilation command (object file will be appended) [example: "gfortran -c -o"]')
    return vars(parser.parse_args())


if __name__ == '__main__':
    configure(**parse_cli())
