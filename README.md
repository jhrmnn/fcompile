# `fcompile` — fast Fortran build tool

Fcompile is a specialized build tool written in Python 3.6 with `asyncio`, that can do one thing only: given a set of Fortran source files, it compiles them into object files with as few recompilations as possible. This is achieved by hashing generated module files on the fly and recompiling the (automatically determined) files that depend on them only if a module file changed.

Fcompile reads the necessary information in a JSON format from the standard input. See an example configuration for details:

```json
{
  "a.f90": {
  	"source": "src/a.f90",
  	"args": ["gfortran", "-c", "-o", "build/a.o"]
  },
  "b.f90": {
    "source": "src/b.f90",
    "args": ["mpifort", "-c", "-o", "build/b.o"]
  }
}
```

Fcompile provides a convenient progress line during compilation:

```
...
Compiled SCGW/scgw_allocations.f90.
Compiled inner_product.f90.
Compiled python_interface_stub.f90.
 Progress: 90 waiting, 1036 scheduled, 223126/735391 lines (30.3%), ETA: 255.5 s
```
