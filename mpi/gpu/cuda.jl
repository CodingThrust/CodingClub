using CuYao, CUDA
CUDA.set_runtime_version!("local")

reg = zero_state(10) |> cu
reg |> put(10, 2=>X)
res = measure(reg)
println(res)
