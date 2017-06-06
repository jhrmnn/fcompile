#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import re
import sys
import hashlib
import json
import os
from collections import defaultdict
from argparse import ArgumentParser, Namespace
import asyncio
from pathlib import Path
from asyncio import Queue, PriorityQueue
from itertools import product, islice
import time

from typing import ( # noqa
    Dict, Any, DefaultDict, List, Iterator, Sequence, IO, Set, Tuple, Union,
    NamedTuple, NewType, Optional, TYPE_CHECKING, cast, Generator, TypeVar,
    Iterable
)

_T = TypeVar('_T')

Module = NewType('Module', str)
Source = NewType('Source', str)
Filename = Union[str, Source]
Hash = NewType('Hash', str)
Args = NewType('Args', Tuple[str, ...])

DEBUG = os.environ.get('DEBUG')
cachefile = '_fcompile_cache.json'


def parse_modules(f: IO[str]) -> Tuple[int, List[Module], Set[Module]]:
    defined = []
    used = set()
    nlines = 0
    for line in f:
        nlines += 1
        line = line.lstrip()
        if not line:
            continue
        if line[0] == '!':
            continue
        word = line[:line.find(' ')].lower()
        if word == 'module':
            module = re.match(r'module\s+(\w+)\s*', line, re.IGNORECASE).group(1)
            module = module.lower()
            if module != 'procedure':
                defined.append(Module(module))
        elif word == 'use':
            module = re.match(r'use\s+(\w+)\s*', line, re.IGNORECASE).group(1)
            used.add(Module(module.lower()))
    used.difference_update(defined)
    return nlines, defined, used


def get_priority(tree: Dict[_T, List[_T]]) -> Dict[_T, int]:
    priority: Dict[_T, int] = {}

    def getsetter(node: _T) -> int:
        try:
            return priority[node]
        except KeyError:
            pass
        p = sum(getsetter(n) for n in tree[node]) or 1
        priority[node] = p
        return p
    for node in tree:
        getsetter(node)
    return priority


def get_ancestors(tree: Dict[_T, Set[_T]]) -> Dict[_T, Set[_T]]:
    ancestors: Dict[_T, Set[_T]] = {}

    def getsetter(node: _T) -> Set[_T]:
        try:
            return ancestors[node]
        except KeyError:
            pass
        ancs = set(tree[node])
        for n in tree[node]:
            ancs.update(getsetter(n))
        ancestors[node] = ancs
        return ancs
    for node in tree:
        getsetter(node)
    return ancestors


class GraphWithCycles(Exception):
    pass


def get_topsort(tree: Dict[_T, List[_T]]) -> List[_T]:
    idxs = {node: n for n, node in enumerate(tree)}
    outgoing = [
        [idxs[child] for child in children] for node, children in tree.items()
    ]
    N = len(tree)
    nincoming = N*[0]
    for edges in outgoing:
        for n in edges:
            nincoming[n] += 1
    L = []
    S = [n for n in range(N) if nincoming[n] == 0]
    while S:
        n = S.pop()
        L.append(n)
        for m in outgoing[n]:
            nincoming[m] -= 1
            if not nincoming[m]:
                S.append(m)
    if sum(nincoming):
        raise GraphWithCycles()
    iidxs = {n: node for node, n in idxs.items()}
    return [iidxs[n] for n in L]


def get_subgraphs(tree: Dict[_T, List[_T]]) -> List[List[_T]]:
    bitree = {node: list(children) for node, children in tree.items()}
    labels: Dict[_T, int] = {}

    def assign(node: _T, label: int) -> None:
        if node in labels:
            return
        labels[node] = label
        for child in bitree[node]:
            assign(child, label)
    for i, node in enumerate(tree):
        assign(node, i)
    subgraphs: DefaultDict[int, List[_T]] = defaultdict(list)
    for node, label in labels.items():
        subgraphs[label].append(node)
    return list(subgraphs.values())


def get_hash(path: Path, args: Args = None) -> Hash:
    h = hashlib.new('sha1')
    if args is not None:
        h.update(' '.join(args).encode())
    with path.open('rb') as f:
        h.update(f.read())
    return Hash(h.hexdigest())


class TaskTree(NamedTuple):
    src_deps: Dict[Source, Set[Module]]
    src_mods: Dict[Source, List[Module]]
    mod_uses: Dict[Module, List[Source]]
    mod_defs: Dict[Module, Source]
    hashes: Dict[Filename, Hash]
    line_nums: Dict[Source, int]
    priority: Dict[Source, int]
    ancestors: Dict[Source, Set[Source]]


class Task(NamedTuple):
    source: Path
    args: Args
    includes: List[str]


class ModuleMultipleDefined(Exception):
    pass


class ModuleNotDefined(Exception):
    pass


def get_tree(tasks: Dict[Source, Task]) -> TaskTree:
    src_mods: Dict[Source, List[Module]] = {}
    mod_defs: Dict[Module, Source] = {}
    src_deps: Dict[Source, Set[Module]] = {}
    hashes: Dict[Filename, Hash] = {}
    line_nums: Dict[Source, int] = {}
    for src, task in tasks.items():
        with open(task.source) as f:
            nlines, defined, used = parse_modules(f)
        src_mods[src] = defined
        src_deps[src] = used
        line_nums[src] = nlines
        hashes[src] = get_hash(task.source, task.args)
        for module in defined:
            if module in mod_defs:
                raise ModuleMultipleDefined(module, [mod_defs[module], src])
            else:
                mod_defs[module] = src
    for used in src_deps.values():
        used.discard(Module('iso_c_binding'))
    if Module('mpi') not in mod_defs:
        for used in src_deps.values():
            used.discard(Module('mpi'))
    for src, task in tasks.items():
        if task.includes:
            for incdir, module in product(task.includes, src_deps[src]):  # type: ignore
                if os.path.exists(os.path.join(incdir, module + '.mod')):
                    src_deps[src].remove(module)
    for mod in set(mod for mods in src_deps.values() for mod in mods):
        if mod not in mod_defs:
            raise ModuleNotDefined(mod)
    mod_uses: DefaultDict[Module, List[Source]] = defaultdict(list)
    for src, modules in src_deps.items():
        for module in modules:
            mod_uses[module].append(src)
    priority = get_priority({
        src: [t for m in mods for t in mod_uses[m]]
        for src, mods in src_mods.items()
    })
    ancestors = get_ancestors({
        src: set(mod_defs[m] for m in mods)
        for src, mods in src_deps.items()
    })
    return TaskTree(
        src_deps, src_mods, mod_uses, mod_defs, hashes, line_nums, priority, ancestors
    )


if TYPE_CHECKING:
    TaskQueue = PriorityQueue[Tuple[int, Source, Args]]
    ResultQueue = Queue[Tuple[Source, int, float]]
else:
    TaskQueue, ResultQueue = None, None


# clear line and print
def pprint(s: Any) -> None:
    sys.stdout.write('\x1b[2K\r{0}\n'.format(s))


clocks: List[Tuple[Source, float, int]] = []


class CompilationError(Exception):
    pass


async def scheduler(tasks: Dict[Source, Task],
                    task_queue: TaskQueue,
                    result_queue: ResultQueue,
                    tree: TaskTree,
                    hashes: Dict[Filename, Hash],
                    changed_files: List[Source]) -> None:
    n_all_lines = sum(tree.line_nums[src] for src in changed_files)
    n_lines = 0
    waiting = set(changed_files)
    scheduled: Dict[Source, Tuple[int, Source, Args]] = {}
    while waiting or scheduled:
        blocking = waiting | set(scheduled)
        for src in list(waiting):
            if not (blocking & tree.ancestors[src]):
                hashes.pop(src, None)  # if compilation gets interrupted
                task_tuple = (
                    -tree.priority[src],
                    src,
                    Args(tasks[src].args + (str(tasks[src].source),))
                )
                task_queue.put_nowait(task_tuple)
                scheduled[src] = task_tuple
                waiting.remove(src)
        src, retcode, clock = await result_queue.get()
        if retcode != 0:
            raise CompilationError(src, retcode)
        clocks.append((src, clock, tree.line_nums[src]))
        hashes[src] = tree.hashes[src]
        n_lines += tree.line_nums[src]
        del scheduled[src]
        pprint(f'Compiled {src}.')
        sys.stdout.write(
            f' Progress: {len(waiting)} waiting, {len(scheduled)} scheduled, ' +
            f'{n_lines}/{n_all_lines} lines ({100*n_lines/n_all_lines:.1f}%)\r'
        )
        sys.stdout.flush()
        for mod in tree.src_mods[src]:
            modfile = mod + '.mod'
            modhash = get_hash(Path(modfile))
            if modhash != hashes.get(modfile):
                hashes[modfile] = modhash
                for src in tree.mod_uses.get(mod, []):  # modules may be unused
                    hashes.pop(src, None)
                    waiting.add(src)


async def worker(task_queue: TaskQueue, result_queue: ResultQueue) -> None:
    while True:
        _, taskname, args = await task_queue.get()
        proc = await asyncio.create_subprocess_exec(*args)
        now = time.time()
        retcode = await proc.wait()
        result_queue.put_nowait((taskname, retcode, time.time()-now))


def build(tasks: Dict[Source, Task], opts: Namespace) -> None:
    print('Scanning files...')
    tree = get_tree(tasks)
    try:
        with open(cachefile) as f:
            hashes = {k: Hash(v) for k, v in json.load(f)['hashes'].items()}
    except (ValueError, FileNotFoundError):
        hashes = {}
    changed_files = [
        src for src in tasks if tree.hashes[src] != hashes.get(src)
    ]
    print(f'Changed files: {len(changed_files)}/{len(tasks)}.')
    if not changed_files or opts.dry:
        return
    task_queue: TaskQueue = PriorityQueue()
    result_queue: ResultQueue = Queue()
    loop = asyncio.get_event_loop()
    workers = [
        loop.create_task(worker(task_queue, result_queue))
        for _ in range(opts.jobs)
    ]
    try:
        loop.run_until_complete(
            scheduler(tasks, task_queue, result_queue, tree, hashes, changed_files)
        )
    except CompilationError as e:
        print(f'Compilation of {e.args[0]} returned {e.args[1]}.')
        sys.exit(1)
    except:
        print()
        raise
    else:
        print()
    finally:
        with open(cachefile, 'w') as f:
            json.dump({'hashes': hashes}, f)
        if DEBUG:
            rows = list(islice(sorted(clocks, key=lambda x: -x[1]), 20))
            maxnamelen = max(len(r[0]) for r in rows)
            print(f'{"File":<{maxnamelen+2}}    {"Time [s]":<6}  {"Lines":<6}')
            for file, clock, nlines in rows:
                print(f'  {file:<{maxnamelen+2}}  {clock:>6.2f}  {nlines:>6}')
        for tsk in workers:
            tsk.cancel()


def read_tasks() -> Tuple[Dict[Source, Task], Namespace]:
    cpu_count = os.cpu_count()//2 or 1  # type: ignore
    parser = ArgumentParser(usage='usage: fcompile.py [options] <CONFIG.json')
    arg = parser.add_argument
    arg('-j', '--jobs', type=int, default=cpu_count,
        help=f'number of threads [default: {cpu_count}]')
    arg('--dry', action='store_true',
        help='print changed files and exit')
    arg('--ignore-errors', action='store_true',
        help='ignore errors during compilation')
    arg('--print-deps', action='store_true',
        help='print module dependencies and exit')
    opts = parser.parse_args()
    tasks = {
        Source(k): Task(Path(t['source']), Args(tuple(t['args'])), t.get('includes', []))
        for k, t in json.load(sys.stdin).items()
    }
    return tasks, opts


if __name__ == '__main__':
    try:
        build(*read_tasks())
    except KeyboardInterrupt:
        sys.exit(1)
