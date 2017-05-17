#!/usr/bin/env python3
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
from itertools import product

from typing import ( # noqa
    Dict, Any, DefaultDict, List, Iterator, Sequence, IO, Set, Tuple, Union,
    NamedTuple, NewType, Optional, TYPE_CHECKING, cast
)


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
    return TaskTree(src_deps, src_mods, dict(mod_uses), mod_defs, hashes, line_nums)


if TYPE_CHECKING:
    TaskQueue = PriorityQueue[Tuple[int, TaskId, List[str]]]
    ResultQueue = Queue[Tuple[TaskId, int]]
else:
    TaskQueue, ResultQueue = None, None


# clear line and print
def pprint(s: Any) -> None:
    sys.stdout.write('\x1b[2K\r{0}\n'.format(s))


async def scheduler(
    task_queue: TaskQueue, result_queue: ResultQueue,
    tree: TaskTree, hashes: Dict[str, Hash], changed_files: List[TaskId]
) -> None:
    waiting = set(changed_files)
    scheduled: Set[TaskId] = set()
    while waiting | scheduled:
        for taskid in list(waiting):
            if all(
                    tree.mod_defs[mod] not in waiting | scheduled
                    for mod in tree.src_deps[taskid]
            ):
                hashes.pop(taskid, None)  # if compilation gets interrupted
                task_queue.put_nowait((
                    tree.line_nums[taskid],
                    taskid,
                    tasks[taskid].args + [str(tasks[taskid].source)]
                ))
                waiting.remove(taskid)
                scheduled.add(taskid)
        taskid, retcode = await result_queue.get()
        hashes[taskid] = tree.hashes[taskid]
        pprint(f'Compiled {taskid}.')
        sys.stdout.write(f'Progress: {len(waiting)} waiting, {len(scheduled)} scheduled\r')
        sys.stdout.flush()
        scheduled.remove(taskid)
        for mod in tree.src_mods[taskid]:
            modfile = mod + '.mod'
            modhash = get_hash(Path(modfile))
            if modhash != hashes.get(modfile):
                hashes[modfile] = modhash
                for taskid in tree.mod_uses.get(mod, []):  # modules may be unused
                    hashes.pop(taskid, None)
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
    if opts.dry:
        print(changed_files)
        return
    if not changed_files:
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
            scheduler(task_queue, result_queue, tree, hashes, changed_files)
        )
    finally:
        with open(cachefile, 'w') as f:
            json.dump({'hashes': hashes}, f)
    for tsk in workers:
        tsk.cancel()


if __name__ == '__main__':
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
    try:
        build(tasks, opts)
    except KeyboardInterrupt:
        sys.exit(1)
