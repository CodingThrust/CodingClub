using CUDA, CUDA.GPUArrays

function my_permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    Base.checkdims_perm(dest, src, perm)

    # get the new strides of destination tensor
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->dest_strides[findfirst(==(i), perm)], N)

    function permutedims_kernel(ctx, dest, src, dest_strides_perm)
        # find the cartesian index in source tensor
        LI = @linearidx src
        I = @inbounds CartesianIndices(src)[LI]

        # the corresponding linear index in the destination tensor
        dest_index = map_index(I.I, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
        return
    end
    gpu_call(permutedims_kernel, dest, src, dest_strides_perm)
    return dest
end

# get linear index from cartesian indices and strides.
@inline @generated function map_index(I::NTuple{N}, dest_strides::NTuple{N,T}) where {N,T}
    Expr(:call, :+, one(T), [:(@inbounds (I[$i]-1) * dest_strides[$i]) for i in 1:N]...)
end

using Test
@testset "permutedims" begin
    a = randn(10, 10, 4, 6, 20)
    b = randn(20, 4, 6, 10, 10)
    c1 = my_permutedims!(CuArray(a), CuArray(b), (5, 4, 2, 3, 1))
    c2 = permutedims!(a, b, (5, 4, 2, 3, 1))
    @test c1 isa CuArray
    @test Array(c1) ≈ c2
end