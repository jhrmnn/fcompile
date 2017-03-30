#!/usr/bin/env python3
import re
import sys
import hashlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
import time
from argparse import ArgumentParser, Namespace
import asyncio
from pathlib import Path
from asyncio import Queue, LifoQueue
from typing import Dict, Any, DefaultDict, List, Iterator, Sequence, IO  # noqa
from typing import Set, Tuple, Union, NamedTuple, NewType, Optional  # noqa


Hash = NewType('Hash', str)
Filename = NewType('Filename', str)
CompileQueue = LifoQueue[Tuple[Filename, List[str]]]
ResultQueue = Queue[Tuple[Optional[Filename], float]]


class Task(NamedTuple):
    source: Path
    args: List[str]
    includes: List[str]


class Clock:
    def __init__(self, active: bool = True) -> None:
        self.active = active
        self.clocks: DefaultDict[str, float] = defaultdict(float)
        self.stack: List[str] = []
        self.last = 0.

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        if self.active:
            label = '>'.join(self.stack + [name])
            self.clocks[label]
            self.stack.append(name)
            tm = time.time()
        try:
            yield
        finally:
            if self.active:
                self.last = time.time()-tm
                self.clocks[label] += self.last
                self.stack.pop(-1)

    def print(self, header: str = None) -> None:
        if not self.active:
            return
        if header is not None:
            print(header)
        for label, clock in sorted(self.clocks.items()):
            print(label, clock)


is_debug = bool(os.environ.get('DEBUG'))
timing = Clock(active=is_debug)

cachename = '_fcompile_cache.json'


def get_file_hash(path: Path, args: Sequence[str] = None) -> Hash:
    with timing('sha1'):
        h = hashlib.new('sha1')
        if args is not None:
            h.update(' '.join(args).encode())
        with path.open('rb') as f:
            h.update(f.read())
        return Hash(h.hexdigest())


async def get_worker(
        compile_queue: CompileQueue, result_queue: ResultQueue,
        ignore_errors: bool = False) -> None:
    clock = Clock()
    while True:
        task = await compile_queue.get()
        if task is None:
            break
        filename, args = task
        with clock('compilation'):
            proc = await asyncio.create_subprocess_exec(*args)
            await proc.wait()
            if proc.returncode != 0 and not ignore_errors:
                break
        await result_queue.put((filename, clock.last))
    await result_queue.put((None, clock.clocks['compilation']))


def parse_modules(f: IO[str]) -> Tuple[int, Set[str], Set[str]]:
    defined = set()
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
                defined.add(module)
        elif word == 'use':
            module = re.match(r'use\s+(\w+)\s*', line, re.IGNORECASE).group(1)
            used.add(module.lower())
    used.difference_update(defined)
    return nlines, defined, used


class DependencyTree:
    def __init__(self, tasks: Dict[Filename, Task]) -> None:
        # object attributes
        self.filenames: List[Filename] = list(tasks.keys())
        self.source_modules: DefaultDict[Filename, List[str]] = defaultdict(list)
        self.source_dependencies: DefaultDict[Filename, List[Filename]] = defaultdict(list)
        self.module_dependants: DefaultDict[str, List[Filename]] = defaultdict(list)
        self.source_dependants: DefaultDict[Filename, List[Filename]] = defaultdict(list)
        self.line_numbers: Dict[Filename, int] = {}
        # helper dictionaries
        module_sources: DefaultDict[str, List[Filename]] = defaultdict(list)
        module_dependencies: Dict[Filename, Set[str]] = {}
        # scan files
        for filename in self.filenames:
            with open(tasks[filename].source) as f:
                nlines, defined, used = parse_modules(f)
            for module in defined:
                module_sources[module].append(filename)
            module_dependencies[filename] = used
            self.line_numbers[filename] = nlines
        # deal with special modules
        for used in module_dependencies.values():
            try:
                used.remove('iso_c_binding')
            except KeyError:
                pass
        if 'mpi' not in module_sources:
            for used in module_dependencies.values():
                try:
                    used.remove('mpi')
                except KeyError:
                    pass
        for source, task in tasks.items():
            if len(task.includes) == 0:
                continue
            for module in list(module_dependencies[source]):
                for incdir in task.includes:
                    if os.path.exists(os.path.join(incdir, module + '.mod')):
                        module_dependencies[source].remove(module)
        # check trivial inconsistencies
        for module in list(module_sources):
            if len(module_sources[module]) > 1:
                print(
                    f'error: Multiple definition of module {module} '
                    f'in {module_sources[module]}'
                )
                sys.exit(1)
        all_used_modules = set(
            module
            for modules in module_dependencies.values()
            for module in modules
        )
        for module in all_used_modules:
            if module not in module_sources:
                print(f'error: No source for module {module}')
                sys.exit(1)
        # populate dictionaries
        for filename, modules in module_dependencies.items():
            for module in modules:
                modulefile = module_sources[module][0]
                self.module_dependants[module].append(filename)
                self.source_dependants[modulefile].append(filename)
                self.source_dependencies[filename].append(modulefile)
        for module, (filename,) in module_sources.items():
            self.source_modules[filename].append(module)


async def get_master(
        tasks: Dict[Filename, Task],
        compile_queue: CompileQueue, result_queue: ResultQueue, tree: DependencyTree,
        changed_files: List[Filename], source_hashes: Dict[Filename, Hash],
        compiled_hashes: Dict[Filename, Hash]) -> None:
    # setup queues and workers
    queue = list(changed_files)  # local master queue of tasks
    blocking = {  # does file block
        filename: filename in queue for filename in tree.filenames
    }
    submitted: List[Filename] = []  # local queue of submitted tasks
    # file stats
    stat = Stat(
        sum(tree.line_numbers[filename] for filename in changed_files),
        len(changed_files)
    )
    file_timings = {}
    # main build loop
    start_time = time.time()
    filename: Optional[Filename]
    try:
        while len(queue + submitted) > 0:
            to_queue = []  # files to queue for compile in this cycle
            for filename in queue:
                if all(
                    not blocking[filename]
                    for filename in tree.source_dependencies[filename]
                ):
                    to_queue.append(filename)
            # compile longest files asap
            to_queue.sort(key=lambda filename: tree.line_numbers[filename])
            # queue
            for filename in to_queue:
                queue.remove(filename)
                submitted.append(filename)
                if filename in compiled_hashes:
                    del compiled_hashes[filename]  # make sure hashes are ok
                await compile_queue.put((
                    filename,
                    tasks[filename].args + [str(tasks[filename].source)]
                ))
            if not submitted:
                continue
            with timing('waiting'):
                filename, clock = await result_queue.get()  # wait for compiled files
            if filename is None:  # worker finished prematurely
                # has_error = True
                await result_queue.put((None, clock))
                break
            # process compiled file
            blocking[filename] = False  # unblock
            compiled_hashes[filename] = source_hashes[filename]
            stat.file_compiled(tree.line_numbers[filename])
            current_time = time.time()
            file_timings[filename] = clock
            estimated_time = (current_time-start_time)*stat.all_lines/stat.compiled_lines
            pprint(f'Compiled {filename}.')
            submitted.remove(filename)
            # check if module files changed
            for module in tree.source_modules[filename]:
                modulefile = Filename(module + '.mod')
                modulehash = get_file_hash(Path(modulefile))
                if modulehash != compiled_hashes.get(modulefile):
                    compiled_hashes[modulefile] = modulehash
                    # depending files are not up-to-date anymore
                    for dependant in tree.module_dependants[module]:
                        if dependant in compiled_hashes:
                            del compiled_hashes[dependant]
                    # queue depending files
                    for dependant in tree.module_dependants[module]:
                        if dependant not in queue:
                            queue.append(dependant)
                            blocking[dependant] = True  # block
                            stat.file_compiled(tree.line_numbers[dependant])
            # print progress line
            progress_line = \
                'Progress: {5}/{6} files, {0}/{1} lines ({2:.1f}%), {3:.1f}s/{4:.1f}s' \
                .format(
                    stat.compiled_lines, stat.all_lines,
                    (100*stat.compiled_lines)/stat.all_lines,
                    current_time-start_time, estimated_time,
                    stat.compiled_files, stat.all_files
                )
            if is_debug:
                compile_queue_size = compile_queue.qsize()
                progress_line += ' [compile_queue: {}, running: {}]'.format(
                    compile_queue_size, len(submitted)-compile_queue_size
                )
            sys.stdout.write(progress_line + '\r')
            sys.stdout.flush()
    finally:
        # print()
        # for _ in workers:
        #     compile_queue.put(None)  # terminate workers
        # for _ in workers:
        #     while True:  # throw away compiled unprocessed files
        #         obj, clock = result_queue.get()
        #         if obj is None:
        #             break
        #     timing.clocks['compilation'] += clock
        # for thread in workers:
        #     thread.join()  # terminate threads
        if stat.compiled_files > 0:
            print(
                'Parallelization: {0:.2f}'
                .format(sum(file_timings.values())/(current_time-start_time))
            )
        if is_debug and stat.compiled_files > 0:
            timings = sorted(file_timings.items(), key=lambda it: it[1])
            median_compile_time = timings[stat.compiled_files//2][1]
            print('Median time per file: {0:.3f}s'.format(median_compile_time))
            print('Files with longest compile time:')
            for filename, clock in timings[:-4:-1]:
                print('    {0}: {1:.1f}s'.format(filename, clock))
        with open(cachename, 'w') as f:
            json.dump({'hashes': compiled_hashes}, f)
    # return 1 if has_error else 0


class Stat:
    def __init__(self, all_lines: int, all_files: int) -> None:
        self.all_lines = all_lines
        self.compiled_lines = 0
        self.all_files = all_files
        self.compiled_files = 0

    def file_compiled(self, lines: int) -> None:
        self.compiled_files += 1
        self.compiled_lines += lines


# clear line and print
def pprint(s: Any) -> None:
    sys.stdout.write('\x1b[2K\r{0}\n'.format(s))


def build(tasks: Dict[Filename, Task], opts: Namespace) -> None:
    # prepare dependency tree
    print('Scanning files...')
    with timing('scanning'):
        tree = DependencyTree(tasks)
    if opts.print_deps:
        for source, modulefiles in sorted(tree.source_dependencies.items()):
            print('{0}: {1}'.format(source, ', '.join(modulefiles)))
        return
    # get hashes of source files & args
    source_hashes = {
        filename: get_file_hash(
            tasks[filename].source,
            args=tasks[filename].args
        ) for filename in tree.filenames
    }
    # read compiled hashes
    try:
        with open(cachename) as f:
            compiled_hashes = {
                Filename(k): Hash(v) for k, v in json.load(f)['hashes'].items()
            }
    except (ValueError, FileNotFoundError):
        compiled_hashes = {}
    # get changed files
    changed_files = [
        filename for filename in tree.filenames
        if source_hashes[filename] != compiled_hashes.get(filename)
    ]
    print('Changed files: {0}/{1}.'.format(len(changed_files), len(tree.filenames)))
    # check if continue
    if opts.dry:
        print(changed_files)
        return
    if not changed_files:
        return
    # start build loop
    compile_queue: CompileQueue = LifoQueue()  # queue of tasks to be compiled asap
    result_queue: ResultQueue = Queue()  # queue of results of compilation
    workers = [
        get_worker(compile_queue, result_queue, opts.ignore_errors)
        for _ in range(opts.jobs)
    ]
    master = get_master(tasks, compile_queue, result_queue, tree, changed_files, source_hashes, compiled_hashes)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(master, *workers))
    loop.close()


def main(argv: List[str]) -> int:
    parser = ArgumentParser(usage='usage: fcompile.py [options] <CONFIG.json')
    parser.add_argument(
        '-j', '--jobs', type=int, default=os.cpu_count(),
        help='number of threads'
    )
    parser.add_argument(
        '--dry', action='store_true',
        help='print changed files and exit'
    )
    parser.add_argument(
        '--ignore-errors', action='store_true',
        help='ignore errors during compilation'
    )
    parser.add_argument(
        '--print-deps', action='store_true',
        help='print module dependencies and exit'
    )
    args = parser.parse_args(args=sys.argv[1:])
    tasks = {
        Filename(k): Task(Path(t['source']), t['args'], t.get('includes', []))
        for k, t in json.load(sys.stdin).items()
    }
    try:
        with timing('all'):
            build(tasks, args)
    except KeyboardInterrupt:
        return 1
    finally:
        timing.print(header='Timing:')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
