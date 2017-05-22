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
from contextlib import contextmanager
from itertools import product

from typing import ( # noqa
    Dict, Any, DefaultDict, List, Iterator, Sequence, IO, Set, Tuple, Union,
    NamedTuple, NewType, Optional, TYPE_CHECKING, cast, Generator, TypeVar
)

_T = TypeVar('_T')


import time

_clocks: DefaultDict[str, float] = defaultdict(float)


@contextmanager
def timing(label: str) -> Generator[None, None, None]:
    tm = time.time()
    try:
        yield
    finally:
        _clocks[label] += time.time()-tm


cachefile = '_fcompile_cache.json'


def parse_modules(f: IO[str]) -> Tuple[int, List[str], Set[str]]:
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
                defined.append(module)
        elif word == 'use':
            module = re.match(r'use\s+(\w+)\s*', line, re.IGNORECASE).group(1)
            used.add(module.lower())
    used.difference_update(defined)
    return nlines, defined, used


TaskId = NewType('TaskId', str)
Hash = NewType('Hash', str)


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


def get_hash(path: Path, args: Sequence[str] = None) -> Hash:
    h = hashlib.new('sha1')
    if args is not None:
        h.update(' '.join(args).encode())
    with path.open('rb') as f:
        h.update(f.read())
    return Hash(h.hexdigest())


class TaskTree(NamedTuple):
    src_deps: Dict[TaskId, Set[str]]
    src_mods: Dict[TaskId, List[str]]
    mod_uses: Dict[str, List[TaskId]]
    mod_defs: Dict[str, TaskId]
    hashes: Dict[TaskId, Hash]
    line_nums: Dict[TaskId, int]
    priority: Dict[TaskId, int]


class Task(NamedTuple):
    source: Path
    args: List[str]
    includes: List[str]


class ModuleMultipleDefined(Exception):
    pass


class ModuleNotDefined(Exception):
    pass


def get_tree(tasks: Dict[TaskId, Task]) -> TaskTree:
    src_mods: Dict[TaskId, List[str]] = {}
    mod_defs: Dict[str, TaskId] = {}
    src_deps: Dict[TaskId, Set[str]] = {}
    hashes: Dict[TaskId, Hash] = {}
    line_nums: Dict[TaskId, int] = {}
    for taskid, task in tasks.items():
        with open(task.source) as f:
            nlines, defined, used = parse_modules(f)
        src_mods[taskid] = defined
        src_deps[taskid] = used
        line_nums[taskid] = nlines
        hashes[taskid] = get_hash(task.source, task.args)
        for module in defined:
            if module in mod_defs:
                raise ModuleMultipleDefined(module, [mod_defs[module], taskid])
            else:
                mod_defs[module] = taskid
    for used in src_deps.values():
        used.discard('iso_c_binding')
    if 'mpi' not in mod_defs:
        for used in src_deps.values():
            used.discard('mpi')
    for taskid, task in tasks.items():
        if task.includes:
            for incdir, module in product(task.includes, src_deps[taskid]):
                if os.path.exists(os.path.join(incdir, module + '.mod')):
                    src_deps[taskid].remove(module)
    for mod in set(mod for mods in src_deps.values() for mod in mods):
        if mod not in mod_defs:
            raise ModuleNotDefined(mod)
    mod_uses: DefaultDict[str, List[TaskId]] = defaultdict(list)
    for taskid, modules in src_deps.items():
        for module in modules:
            mod_uses[module].append(taskid)
    priority = get_priority({
        taskid: [t for m in mods for t in mod_uses[m]]
        for taskid, mods in src_mods.items()
    })
    return TaskTree(
        src_deps, src_mods, dict(mod_uses),
        mod_defs, hashes, line_nums, priority
    )


if TYPE_CHECKING:
    TaskQueue = PriorityQueue[Tuple[int, TaskId, List[str]]]
    ResultQueue = Queue[Tuple[TaskId, int]]
else:
    TaskQueue, ResultQueue = None, None


# clear line and print
def pprint(s: Any) -> None:
    sys.stdout.write('\x1b[2K\r{0}\n'.format(s))


async def scheduler(
    tasks: Dict[TaskId, Task], task_queue: TaskQueue, result_queue: ResultQueue,
    tree: TaskTree, hashes: Dict[str, Hash], changed_files: List[TaskId]
) -> None:
    n_all_lines = sum(tree.line_nums[taskid] for taskid in changed_files)
    n_lines = 0
    waiting = set(changed_files)
    scheduled: Dict[TaskId, Tuple[int, TaskId, List[str]]] = {}
    while waiting or scheduled:
        blocking = set(
            mod for mod, src in tree.mod_defs.items()
            if src in waiting or src in scheduled
        )
        for taskid in list(waiting):
            if not (tree.src_deps[taskid] & blocking):
                hashes.pop(taskid, None)  # if compilation gets interrupted
                task_tuple = (
                    -tree.priority[taskid],
                    taskid,
                    tasks[taskid].args + [str(tasks[taskid].source)]
                )
                task_queue.put_nowait(task_tuple)
                scheduled[taskid] = task_tuple
                waiting.remove(taskid)
        taskid, retcode = await result_queue.get()
        hashes[taskid] = tree.hashes[taskid]
        n_lines += tree.line_nums[taskid]
        pprint(f'Compiled {taskid}.')
        sys.stdout.write(
            f' Progress: {len(waiting)} waiting, {len(scheduled)} scheduled, ' +
            f'{n_lines}/{n_all_lines} lines ({100*n_lines/n_all_lines:.1f}%)\r'
        )
        sys.stdout.flush()
        del scheduled[taskid]
        for mod in tree.src_mods[taskid]:
            modfile = mod + '.mod'
            modhash = get_hash(Path(modfile))
            if modhash != hashes.get(modfile):
                hashes[modfile] = modhash
                for taskid in tree.mod_uses.get(mod, []):  # modules may be unused
                    hashes.pop(taskid, None)
                    try:
                        task_tuple = scheduled.pop(taskid)
                    except KeyError:
                        pass
                    else:
                        task_queue._queue.remove(task_tuple)  # type: ignore
                    waiting.add(taskid)


async def worker(task_queue: TaskQueue, result_queue: ResultQueue) -> None:
    while True:
        _, taskname, args = await task_queue.get()
        retcode = await(await asyncio.create_subprocess_exec(*args)).wait()
        result_queue.put_nowait((taskname, retcode))


def build(tasks: Dict[TaskId, Task], opts: Namespace) -> None:
    print('Scanning files...')
    tree = get_tree(tasks)
    try:
        with open(cachefile) as f:
            hashes = {
                k: Hash(v) for k, v in json.load(f)['hashes'].items()
            }
    except (ValueError, FileNotFoundError):
        hashes = {}
    changed_files = [
        taskid for taskid in tasks if tree.hashes[taskid] != hashes.get(taskid)
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
    finally:
        print()
        with open(cachefile, 'w') as f:
            json.dump({'hashes': hashes}, f)
        if _clocks:
            print('Timing:')
            for label, clock in _clocks.items():
                print(f'  {label + ":":<15} {clock:.2f}')
    for tsk in workers:
        tsk.cancel()


def read_tasks() -> Tuple[Dict[TaskId, Task], Namespace]:
    cpu_count = os.cpu_count()
    parser = ArgumentParser(usage='usage: fcompile.py [options] <CONFIG.json')
    add = parser.add_argument
    add('-j', '--jobs', type=int, default=cpu_count,
        help=f'number of threads [default: {cpu_count}]')
    add('--dry', action='store_true',
        help='print changed files and exit')
    add('--ignore-errors', action='store_true',
        help='ignore errors during compilation')
    add('--print-deps', action='store_true',
        help='print module dependencies and exit')
    opts = parser.parse_args()
    tasks = {
        TaskId(k): Task(Path(t['source']), t['args'], t.get('includes', []))
        for k, t in json.load(sys.stdin).items()
    }
    return tasks, opts


if __name__ == '__main__':
    try:
        build(*read_tasks())
    except KeyboardInterrupt:
        sys.exit(1)
