# Essential Julia packages
You may find trending Julia packages here: https://juliahub.com/ui/Packages

## Development
1. [PkgTemplates](https://github.com/JuliaCI/PkgTemplates.jl) `package development`, `continuous integration`

Create new Julia packages, the easy way.

```julia
julia> using PkgTemplates

julia> tpl = Template(; user="GiggleLiu", plugins=[
           GitHubActions(; extra_versions=["nightly"]),
           Git(),
           Codecov(),
           Documenter{GitHubActions}(),
       ])

julia> tpl("PkgName")
```

2. [Revise](https://github.com/timholy/Revise.jl) `package development`, `reload`

`Revise.jl` allows you to modify code and use the changes without restarting Julia. With Revise, you can be in the middle of a session and then update packages, switch git branches, and/or edit the source code in the editor of your choice; any changes will typically be incorporated into the very next command you issue from the REPL. This can save you the overhead of restarting Julia, loading packages, and waiting for code to JIT-compile.

## High performance computing
1. [CUDA](https://github.com/JuliaGPU/CUDA.jl) `CUDA`, `kernels`

The CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs using Julia. It features a user-friendly array abstraction, a compiler for writing CUDA kernels in Julia, and wrappers for various CUDA libraries.

Example: [cuda.jl](cuda.jl)

2. [LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl) `AVX`, `SIMD`, `CPU`, `speed`

Macro(s) for vectorizing loops.

3. [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) `small arrays`, `static`

StaticArrays provides a framework for implementing statically sized arrays in Julia, using the abstract type StaticArray{Size,T,N} <: AbstractArray{T,N}. Subtypes of StaticArray will provide fast implementations of common array and linear algebra operations. Note that here "statically sized" means that the size can be determined from the type, and "static" does not necessarily imply immutable.

## Optimization
1. [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) `optimize`, `LBFGS`, `Nelder-Mead`

Univariate and multivariate optimization in Julia.

Example: [optim.jl](optim.jl)

2. [JuMP](https://github.com/jump-dev/JuMP.jl) `mathematical optimization`, `fast`

JuMP is a domain-specific modeling language for mathematical optimization embedded in Julia. You can find out more about us by visiting jump.dev.

3. [ForwradDiff](https://github.com/JuliaDiff/ForwardDiff.jl) and [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) `autodiff`, `llvm`, `NIPS best paper`, `forward`, `reverse`

Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.


## Linear Algebra
1. [ITensors](https://github.com/ITensor/ITensors.jl), [TensorOperations](https://github.com/Jutho/TensorOperations.jl) and [OMEinsum](https://github.com/under-Peter/OMEinsum.jl) `tensor`, `contraction`, `physics`

    * ITensor is a library for rapidly creating correct and efficient tensor network algorithms.
    * TensorOperations provides fast tensor operations using a convenient Einstein index notation.
    * This is a repository for the Google Summer of Code project on Differentiable Tensor Networks. It implements one function that both computer scientists and physicists love, the Einstein summation. It features large scale tensor network contraction.
 
2. [KrylovKit](https://github.com/Jutho/KrylovKit.jl) `sparse matrix`, `linear operator`, `eigen value problem`, `iterative solver`, `krylov space`

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular value and eigenvalue problems and the application of functions of linear maps or operators to vectors.

3. [ExponentialUtilities](https://github.com/SciML/ExponentialUtilities.jl) `expmv`, `krylov space`

ExponentialUtilities is a package of utility functions for matrix functions of exponential type, including functionality for the matrix exponential and phi-functions. These methods are more numerically stable, generic (thus support a wider range of number types), and faster than the matrix exponentiation tools in Julia's Base. The tools are used by the exponential integrators in OrdinaryDiffEq. The package has no external dependencies, so it can also be used independently.

## Plotting

1. [Makie](https://github.com/MakieOrg/Makie.jl) `plot`, `gpu`

Example: [makie.jl](cuda.jl)


2. [Luxor](https://github.com/JuliaGraphics/Luxor.jl)

Luxor is a Julia package for drawing simple 2D vector graphics. Think of it as a high-level easier to use interface to Cairo.jl, with shorter names, fewer underscores, default contexts, and simplified functions. In Luxor, the emphasis is on simplicity and ease of use.

3. [UnicodePlots](https://github.com/JuliaPlots/UnicodePlots.jl)
Advanced Unicode plotting library designed for use in Julia's REPL.

## Scientific Computing
1. [DifferetialEquations](https://github.com/SciML/DifferentialEquations.jl) `ODE`, `PDE`, `fast`

This is a suite for numerically solving differential equations written in Julia and available for use in Julia, Python, and R. The purpose of this package is to supply efficient Julia implementations of solvers for various differential equations. Equations within the realm of this package include:

    * Discrete equations (function maps, discrete stochastic (Gillespie/Markov) simulations)
    * Ordinary differential equations (ODEs)
    * Split and Partitioned ODEs (Symplectic integrators, IMEX Methods)
    * Stochastic ordinary differential equations (SODEs or SDEs)
    * Stochastic differential-algebraic equations (SDAEs)
    * Random differential equations (RODEs or RDEs)
    * Differential algebraic equations (DAEs)
    * Delay differential equations (DDEs)
    * Neutral, retarded, and algebraic delay differential equations (NDDEs, RDDEs, and DDAEs)
    * Stochastic delay differential equations (SDDEs)
    * Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)
    * Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions)
    * (Stochastic) partial differential equations ((S)PDEs) (with both finite difference and finite element methods)
   
2. [Yao](https://github.com/QuantumBFS/Yao.jl) `quantum`, `differentiable`, `fast`, `gpu`
    
Yao is an open source framework that aims to empower quantum information research with software tools. It is designed with following in mind:
    * quantum algorithm design;
    * quantum software 2.0;
    * quantum computation education.


## FileIO

1. DelimitedFiles

```julia
julia> using DelimitedFiles

julia> x = randn(100, 100);julia> writedlm(x);

julia> writedlm("_test.dat", x);

julia> y = readdlm("_test.dat");

julia> x ≈ y
true
```

2. [JLD2](https://github.com/JuliaIO/JLD2.jl)

JLD2 saves and loads Julia data structures in a format comprising a subset of HDF5, without any dependency on the HDF5 C library. JLD2 is able to read most HDF5 files created by other HDF5 implementations supporting HDF5 File Format Specification Version 3.0 (i.e. libhdf5 1.10 or later) and similarly those should be able to read the files that JLD2 produces. JLD2 provides read-only support for files created with the JLD package.