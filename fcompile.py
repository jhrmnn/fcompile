#!/usr/bin/env python
from __future__ import print_function
import re
import sys
import hashlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
import subprocess
import time
from threading import Thread
from multiprocessing import cpu_count
from optparse import OptionParser
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue


cachename = '_fcompile_cache.json'


def parse_modules(f):
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


def get_file_hash(filename, prepend=None):
    with timing('sha1'):
        h = hashlib.new('sha1')
        if prepend is not None:
            h.update(prepend)
        with open(filename, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()


class Clock(object):
    def __init__(self):
        self.active = os.environ.get('TIMING')
        self.clocks = defaultdict(float)
        self.stack = []

    @contextmanager
    def __call__(self, name):
        if self.active:
            label = '>'.join(self.stack + [name])
            self.clocks[label]
            self.stack.append(name)
            tm = time.time()
        try:
            yield
        finally:
            if self.active:
                self.clocks[label] += time.time()-tm
                self.stack.pop(-1)

    def print(self):
        if not self.active:
            return
        for label, clock in sorted(self.clocks.items()):
            print(label, clock)


timing = Clock()


def worker(compile_queue, result_queue):
    clock = Clock()
    while True:
        task = compile_queue.get()
        if task is None:
            break
        filename, args = task
        with clock('compilation'):
            try:
                subprocess.check_call(args)
            except (subprocess.CalledProcessError, OSError):
                break
        result_queue.put(filename)
    result_queue.put(clock.clocks['compilation'])


class DependencyTree(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def scan(self, tasks):
        self.module_sources = defaultdict(list)
        self.dependencies = {}
        self.dependants = defaultdict(list)
        self.source_dependants = defaultdict(list)
        self.source_modules = defaultdict(list)
        self.line_numbers = {}
        for filename in self.filenames:
            with open(tasks[filename]['source']) as f:
                nlines, defined, used = parse_modules(f)
            for module in defined:
                self.module_sources[module].append(filename)
            self.dependencies[filename] = used
            self.line_numbers[filename] = nlines
        assert(all(
            len(filenames) == 1 for filenames in self.module_sources.values()
        ))
        self.module_sources = dict(
            (module, filenames[0]) for module, filenames in self.module_sources.items()
        )
        all_used_modules = set(
            module for modules in self.dependencies.values() for module in modules
        )
        for module in all_used_modules:
            if module not in self.module_sources:
                raise RuntimeError('No source for module {0}'.format(module))
        for filename, modules in self.dependencies.items():
            for module in modules:
                self.dependants[module].append(filename)
                self.source_dependants[self.module_sources[module]].append(filename)
        for module, filename in self.module_sources.items():
            self.source_modules[filename].append(module)

    def sort(self):
        dependant_numbers = dict(
            (filename, len(self.source_dependants.get(filename, [])))
            for filename in self.filenames
        )
        queue = [
            filename for filename in self.filenames
            if not self.source_dependants.get(filename)
        ]
        edge_counter = defaultdict(int)
        self.filenames = []
        while queue:
            file_n = queue.pop()
            self.filenames.append(file_n)
            for module in self.dependencies[file_n]:
                file_m = self.module_sources[module]
                edge_counter[file_m] += 1
                if edge_counter[file_m] == dependant_numbers[file_m]:
                    queue.append(file_m)
        if any(
            edge_counter[filename] != dependant_numbers[filename]
            for filename in self.filenames
        ):
            raise RuntimeError('Dependency tree has cycles')
        self.file_ranks = dict(
            (filename, rank) for rank, filename in enumerate(self.filenames)
        )
        self.file_barriers = dict(
            (filename, max(
                self.file_ranks[dependant] for dependant in self.source_dependants[filename]
            ) if self.source_dependants[filename] else -1) for filename in self.filenames
        )


def pprint(s):
    print(s + (80-len(s))*' ')


def build(tasks, opts):
    # prepare dependency tree
    tree = DependencyTree(list(tasks.keys()))
    print('Scanning files...')
    with timing('scanning'):
        tree.scan(tasks)
    with timing('sort'):
        tree.sort()
    # get hashes of source files
    with timing('hashing'):
        source_hashes = dict((filename, get_file_hash(
            tasks[filename]['source'],
            prepend=' '.join(tasks[filename]['args']).encode()
        )) for filename in tree.filenames)
    # read compiled hashes
    if os.path.exists(cachename):
        with open(cachename) as f:
            compiled_hashes = json.load(f)['hashes']
    else:
        compiled_hashes = {}
    # get changed files
    changed_files = [
        filename for filename in tree.filenames
        if source_hashes[filename] != compiled_hashes.get(filename)
    ]
    total_nlines = sum(tree.line_numbers[filename] for filename in changed_files)
    compiled_nlines = 0
    print('Changed files: {0}/{1}.'.format(len(changed_files), len(tree.filenames)))
    if not changed_files:
        return
    # setup queues and workers
    queue = list(changed_files)  # local master queue of tasks
    running = []  # local queue of running tasks
    compile_queue = Queue()  # shared queue of tasks to be compiled asap
    result_queue = Queue()  # shared queue of results of compilation
    pool = [
        Thread(target=worker, args=(compile_queue, result_queue))
        for _ in range(opts.jobs)
    ]
    for thread in pool:
        thread.start()
    # main build loop
    start_time = time.time()
    print('Start compiling.')
    try:
        while queue + running:
            barrier = max(tree.file_barriers[filename] for filename in queue + running)
            while queue and tree.file_ranks[queue[-1]] > barrier:
                filename = queue.pop()
                running.append(filename)
                compile_queue.put((
                    filename,
                    tasks[filename]['args'] + [tasks[filename]['source']]
                ))
            if not running:
                continue
            filename = result_queue.get()
            if isinstance(filename, float):
                result_queue.put(filename)
                break
            compiled_hashes[filename] = source_hashes[filename]
            compiled_nlines += tree.line_numbers[filename]
            current_time = time.time()
            estimated_time = (current_time-start_time)*total_nlines/compiled_nlines
            pprint('Compiled {0}.'.format(filename, 80*' '))
            running.remove(filename)
            for module in tree.source_modules[filename]:
                modulefile = module + '.mod'
                modulehash = get_file_hash(modulefile)
                if modulehash != compiled_hashes.get(modulefile):
                    compiled_hashes[modulefile] = modulehash
                    for dependant in tree.dependants[module]:
                        if dependant not in queue:
                            try:
                                idx = next(
                                    queue.index(name) for name in queue
                                    if tree.file_ranks[name] > tree.file_ranks[dependant]
                                )
                            except StopIteration:
                                idx = len(queue)
                            queue.insert(idx, dependant)
                            total_nlines += tree.line_numbers[dependant]
            sys.stdout.write(
                'Progress: {0}/{1} lines ({2:.1f}%), {3:.1f}s/{4:.1f}s\r'.format(
                    compiled_nlines, total_nlines,
                    (100.*compiled_nlines)/total_nlines,
                    current_time-start_time, estimated_time
                )
            )
            sys.stdout.flush()
    finally:
        for _ in pool:
            compile_queue.put(None)
        for _ in pool:
            while True:
                res = result_queue.get()
                if isinstance(res, float):
                    break
            timing.clocks['compilation'] += res
        for thread in pool:
            thread.join()
        with open(cachename, 'w') as f:
            json.dump({'hashes': compiled_hashes}, f)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-j', '--jobs', type='int', default=cpu_count())
    opts, _ = parser.parse_args()
    tasks = json.load(sys.stdin)
    try:
        with timing('all'):
            build(tasks, opts)
    except KeyboardInterrupt:
        print()
    finally:
        timing.print()
