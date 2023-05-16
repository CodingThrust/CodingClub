# Message Passing Interface (MPI)
## Mindset
Multi-Universe model: all processors run exactly the same program without sharing memories. Without communication, they will end up with returning the same result.
Then, we introduce a web-based oracle object `MPI.COMM_WORLD`. When a processor queries that object with the `Get_rank()` function, that object returns a number that corresponds to the processor ID.
## Watch YouTube Video
* [MPI Basic](https://youtu.be/c0C9mQaxsD4)
* [MPI Advancd](https://youtu.be/q9OfXis50Rg)
## List of packages
1. Python: [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
2. Julia: [MPI.jl](https://juliaparallel.org/MPI.jl/dev/)
3. Backends: [MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/), Intel MPI et al.

## Coding example: Distributed hello-world with MPI.jl
We are working in the Julia project folder, using the project's [local environment](https://pkgdocs.julialang.org/v1/environments/)

1. Add `MPI.jl` to your project dependencies.
```julia
julia> using Pkg; Pkg.activate("."); Pkg.add("MPI")
```

2. Configure the MPI backend ([doc](https://juliaparallel.org/MPI.jl/dev/configuration/))
```julia
julia> using Pkg; Pkg.add("MPIPreferences");

julia> using MPIPreferences; MPIPreferences.use_system_binary()
```
You will see a [LocalPreferences.toml](LocalPreferences.toml) in your working folder.

3. You need to build the MPI package again for the new MPI backend.
```bash
julia --project -e 'using Pkg; Pkg.build("MPI")'
```

4. You may test the program with
```bash
mpiexec -n 3 julia --project mpi.jl
```

## Another example
Go through this example: https://juliaparallel.org/MPI.jl/dev/examples/06-scatterv/

## Using school cluster.
1. Please check the tested [LSF script](julia-helloworld-lsf.job). This script can be executed on a cluster with
```bash
bsub < julia-helloworld-lsf.job
```

2. The [slurm script](julia-helloworld-slurm.slurm) is not tested. It can be executed on a cluster with
```bash
sbatch < julia-helloworld-slurm.slurm
```
