#!/usr/bin/env python
from __future__ import print_function, division
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
    def __init__(self, active=True):
        self.active = active
        self.clocks = defaultdict(float)
        self.stack = []
        self.last = None

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
                self.last = time.time()-tm
                self.clocks[label] += self.last
                self.stack.pop(-1)

    def print(self, header=None):
        if not self.active:
            return
        if header:
            print(header)
        for label, clock in sorted(self.clocks.items()):
            print(label, clock)


timing = Clock(active=is_debug)


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
            except subprocess.CalledProcessError:
                break
            except:
                import traceback
                traceback.print_exc()
                break
        result_queue.put((filename, clock.last))
    result_queue.put((None, clock.clocks['compilation']))


class DependencyTree(object):
    def __init__(self, tasks):
        # object attributes
        self.filenames = tasks.keys()
        self.source_modules = defaultdict(list)
        self.source_dependencies = defaultdict(list)
        self.module_dependants = defaultdict(list)
        self.source_dependants = defaultdict(list)
        self.line_numbers = {}
        # helper dictionaries
        module_sources = defaultdict(list)
        module_dependencies = defaultdict(list)
        # scan files
        for filename in self.filenames:
            with open(tasks[filename]['source']) as f:
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
        # check trivial inconsistencies
        for module in list(module_sources):
            if len(module_sources[module]) > 1:
                raise RuntimeError(
                    'Module {0} defined in {1}'
                    .format(module, module_sources[module])
                )
            module_sources[module] = module_sources[module][0]
        all_used_modules = set(
            module for modules in module_dependencies.values() for module in modules
        )
        for module in all_used_modules:
            if module == 'mpi':
                continue
            if module not in module_sources:
                raise RuntimeError('No source for module {0}'.format(module))
        # populate dictionaries
        for filename, modules in module_dependencies.items():
            for module in modules:
                modulefile = module_sources[module]
                self.module_dependants[module].append(filename)
                self.source_dependants[modulefile].append(filename)
                self.source_dependencies[filename].append(modulefile)
        for module, filename in module_sources.items():
            self.source_modules[filename].append(module)


# clear line and print
def pprint(s):
    sys.stdout.write('\x1b[2K\r{0}\n'.format(s))


def build(tasks, opts):
    has_error = False
    # prepare dependency tree
    print('Scanning files...')
    with timing('scanning'):
        tree = DependencyTree(tasks)
    # get hashes of source files & args
    source_hashes = dict((
        filename,
        get_file_hash(
            tasks[filename]['source'],
            prepend=' '.join(tasks[filename]['args']).encode()
        )
    ) for filename in tree.filenames)
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
    # file stats
    n_all_lines = sum(tree.line_numbers[filename] for filename in changed_files)
    n_compiled_lines = 0
    n_all_files = len(changed_files)
    n_compiled_files = 0
    file_timings = {}
    print('Changed files: {0}/{1}.'.format(len(changed_files), len(tree.filenames)))
    # check if continue
    if opts.dry:
        print(changed_files)
        return
    if not changed_files:
        return
    # setup queues and workers
    queue = list(changed_files)  # local master queue of tasks
    blocking = dict(  # does file block
        (filename, filename in queue) for filename in tree.filenames
    )
    submitted = []  # local queue of submitted tasks
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
    try:
        while queue + submitted:
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
                compile_queue.put((
                    filename,
                    tasks[filename]['args'] + [tasks[filename]['source']]
                ))
            if not submitted:
                continue
            with timing('waiting'):
                filename, clock = result_queue.get()  # wait for compiled files
            if filename is None:  # worker finished prematurely
                has_error = True
                result_queue.put((None, clock))
                break
            # process compiled file
            blocking[filename] = False  # unblock
            compiled_hashes[filename] = source_hashes[filename]
            n_compiled_lines += tree.line_numbers[filename]
            n_compiled_files += 1
            current_time = time.time()
            file_timings[filename] = clock
            estimated_time = (current_time-start_time)*n_all_lines/n_compiled_lines
            pprint('Compiled {0}.'.format(filename))
            submitted.remove(filename)
            # check if module files changed
            for module in tree.source_modules[filename]:
                modulefile = module + '.mod'
                modulehash = get_file_hash(modulefile)
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
                            n_all_lines += tree.line_numbers[dependant]
                            n_all_files += 1
            # print progress line
            progress_line = \
                'Progress: {5}/{6} files, {0}/{1} lines ({2:.1f}%), {3:.1f}s/{4:.1f}s' \
                .format(
                    n_compiled_lines, n_all_lines,
                    (100*n_compiled_lines)/n_all_lines,
                    current_time-start_time, estimated_time,
                    n_compiled_files, n_all_files
                )
            if is_debug:
                compile_queue_size = compile_queue.qsize()
                progress_line += \
                    ' [compile_queue: {0}, running: {1}]' \
                    .format(
                        compile_queue_size, len(submitted)-compile_queue_size
                    )
            sys.stdout.write(progress_line + '\r')
            sys.stdout.flush()
    finally:
        print()
        for _ in pool:
            compile_queue.put(None)  # terminate workers
        for _ in pool:
            while True:  # throw away compiled unprocessed files
                obj, clock = result_queue.get()
                if obj is None:
                    break
            timing.clocks['compilation'] += clock
        for thread in pool:
            thread.join()  # terminate threads
        if is_debug and n_compiled_files > 0:
            file_timings = sorted(file_timings.items(), key=lambda it: it[1])
            median_compile_time = file_timings[n_compiled_files//2][1]
            print('Median time per file: {0:.3f}s'.format(median_compile_time))
            print('Files with longest compile time:')
            for filename, clock in file_timings[:-4:-1]:
                print('    {0}: {1:.1f}s'.format(filename, clock))
        with open(cachename, 'w') as f:
            json.dump({'hashes': compiled_hashes}, f)
    return 1 if has_error else 0


if __name__ == '__main__':
    parser = OptionParser(usage='usage: fcompile.py [options] <CONFIG.json')
    parser.add_option('-j', '--jobs', type='int', default=cpu_count(), help='number of threads')
    parser.add_option('--dry', action='store_true', help='print changed files and exit')
    opts, _ = parser.parse_args()
    tasks = json.load(sys.stdin)
    try:
        with timing('all'):
            sys.exit(build(tasks, opts))
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        timing.print(header='Timing:')
