# Message Passing Interface (MPI)
1. Multi-Universe model: all processors run exactly the same program without sharing memories. Without communication, they will end up with returning the same result.
Then, we introduce a web-based oracle object `MPI.COMM_WORLD`. When a processor queries that object with the `Get_rank()` function, that object returns a number that corresponds to the processor ID.
2. Watch YouTube Video: [MPI Basic](https://youtu.be/c0C9mQaxsD4)
3. List of packages:
    1. Python: [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
    2. Julia: [MPI.jl](https://juliaparallel.org/MPI.jl/dev/)
    3. Backends: [MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/), Intel MPI et al.
4. Coding example: distributed computing with MPI.
5. Using school cluster.