# `fcompile` — fast Fortran build tool

Fcompile is a specialized threaded build tool written for Python 2.6–3.5 that can do one thing only: given a set of Fortran source files, it compiles them into object files with as few recompilations as possible. This is achieved by hashing generated module files on-the-fly and recompiling the (automatically determined) files that depend on them only if a module file changed.

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
Compiled DFPT_phonon/trans_hessian_to_dynamical_matrix.f90.
Compiled DFPT_phonon/integrate_first_order_rho_p1.f90.
Compiled my_triple_Y.f90.
Compiled triple_Y.f90.
Progress: 55/1013 files, 56915/497658 lines (11.4%), 30.5s/267.0s
```
