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
    from Queue import Queue, LifoQueue
else:
    from queue import Queue, LifoQueue


cachename = '_fcompile_cache.json'
is_debug = os.environ.get('DEBUG')


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
        self.active = is_debug
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
        self.source_modules = defaultdict(list)
        self.dependencies = defaultdict(list)
        self.source_dependencies = defaultdict(list)
        self.dependants = defaultdict(list)
        self.source_dependants = defaultdict(list)
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
                modulefile = self.module_sources[module]
                self.dependants[module].append(filename)
                self.source_dependants[modulefile].append(filename)
                self.source_dependencies[filename].append(modulefile)
        for module, filename in self.module_sources.items():
            self.source_modules[filename].append(module)


def pprint(s):
    print(s + (100-len(s))*' ')


def build(tasks, opts):
    # prepare dependency tree
    tree = DependencyTree(list(tasks.keys()))
    print('Scanning files...')
    with timing('scanning'):
        tree.scan(tasks)
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
    total_nfiles = len(changed_files)
    compiled_nfiles = 0
    print('Changed files: {0}/{1}.'.format(len(changed_files), len(tree.filenames)))
    if opts.dry:
        print(changed_files)
        return
    if not changed_files:
        return
    # setup queues and workers
    queue = list(changed_files)  # local master queue of tasks
    blocking = dict(
        (filename, filename in queue) for filename in tree.filenames
    )
    running = []  # local queue of running tasks
    compile_queue = LifoQueue()  # shared queue of tasks to be compiled asap
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
            to_queue = []
            with timing('queueing'):
                for filename in queue:
                    if all(
                        not blocking[filename]
                        for filename in tree.source_dependencies[filename]
                    ):
                        to_queue.append(filename)
            to_queue.sort(key=lambda filename: len(tree.source_dependants[filename]))
            for filename in to_queue:
                queue.remove(filename)
                running.append(filename)
                compile_queue.put((
                    filename,
                    tasks[filename]['args'] + [tasks[filename]['source']]
                ))
            if not running:
                continue
            with timing('waiting'):
                filename = result_queue.get()
            if isinstance(filename, float):
                result_queue.put(filename)
                break
            blocking[filename] = False
            compiled_hashes[filename] = source_hashes[filename]
            compiled_nlines += tree.line_numbers[filename]
            compiled_nfiles += 1
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
                        if dependant in compiled_hashes:
                            del compiled_hashes[dependant]
                    for dependant in tree.dependants[module]:
                        if dependant not in queue:
                            queue.append(dependant)
                            blocking[dependant] = True
                            total_nlines += tree.line_numbers[dependant]
                            total_nfiles += 1
            progress_line = \
                'Progress: {5}/{6} files, {0}/{1} lines ({2:.1f}%), {3:.1f}s/{4:.1f}s' \
                .format(
                    compiled_nlines, total_nlines,
                    (100.*compiled_nlines)/total_nlines,
                    current_time-start_time, estimated_time,
                    compiled_nfiles, total_nfiles
                )
            if is_debug:
                progress_line += ' [compile_queue: {0}]'.format(
                    compile_queue.qsize()
                )
            sys.stdout.write(progress_line + '\r')
            sys.stdout.flush()
    finally:
        print()
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
    assert(not any(blocking.values()))


if __name__ == '__main__':
    parser = OptionParser(usage='usage: fcompile.py [options] <CONFIG.json')
    parser.add_option('-j', '--jobs', type='int', default=cpu_count(), help='number of threads')
    parser.add_option('--dry', action='store_true', help='print changed files and exit')
    opts, _ = parser.parse_args()
    tasks = json.load(sys.stdin)
    try:
        with timing('all'):
            build(tasks, opts)
    except KeyboardInterrupt:
        pass
    finally:
        timing.print()
