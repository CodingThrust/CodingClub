# Linear Algebra

## Setup BLAS backend and multi-threading
```julia
julia> using LinearAlgebra

julia> BLAS.get_config()
LinearAlgebra.BLAS.LBTConfig
Libraries: 
└ [ILP64] libopenblas64_.so

julia> BLAS.get_num_threads()
4
```

If you want to switch to MKL, please type
```julia
julia> using Pkg; Pkg.add("MKL")

julia> using LinearAlgebra, MKL

julia> BLAS.get_config()
LinearAlgebra.BLAS.LBTConfig
Libraries: 
└ [ILP64] libmkl_rt.1.dylib
```

[read more...](https://github.com/JuliaLinearAlgebra/MKL.jl)

Input unicode characters
https://docs.julialang.org/en/v1/manual/unicode-input/
