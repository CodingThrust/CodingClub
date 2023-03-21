### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ b9a9214e-8830-11eb-1751-9d9161202c76
using PlutoUI, Viznet, Compose, Plots

# ╔═╡ bd3a4ece-8b09-11eb-2fcb-0710286e9892
using ForwardDiff: Dual

# ╔═╡ 0f22e7d6-88c5-11eb-0600-ff9b80f2113e
using NiLang, Random

# ╔═╡ 95b2f940-8a4c-11eb-1813-7b28745b5050
using Optim

# ╔═╡ 2b36cfd2-8d77-11eb-0b32-afc6294b5f50
using Statistics: mean

# ╔═╡ 5dfb2158-8c4a-11eb-33e6-f96facaf76fa
using ChainRules

# ╔═╡ f5e3b3de-8b61-11eb-2875-7594a13c1897
using LinearAlgebra

# ╔═╡ 270fef44-8c49-11eb-01af-43bd6e72da09
using Zygote

# ╔═╡ c239e57a-8b62-11eb-1f10-87cda664a87c
using StochasticOptimizers

# ╔═╡ b01b19b6-8bc8-11eb-1a02-bf79f074bad3
begin
	using TupleTools, TropicalNumbers
	using NiLang.AD: GVar
	
	@i function bond_tensor(res::Matrix{T}) where T
		x ← zero(T)
		SWAP(res[2, 2], x)
		x → one(T)
	end

	@i function vertex_tensor(res::Array{T}, n::Int, val::T) where T
		for i=2:length(res)-1
			x ← zero(T)
			SWAP(res[i], x)
			x → one(T)
		end
		x ← one(T)
		res[1] *= x
		res[end] *= val
	end

	@i @inline function :(*=)(+)(z::Tropical, x::Tropical, y::Tropical)
   	 	if x.n > y.n
			z.n += x.n
		else
			z.n += y.n
		end
	end

	@i @inline function (:*=(identity))(x::Tropical, y::Tropical)
		x.n += y.n
	end

	@i @inline function (:*=(*))(out!::Tropical, x::Tropical, y::Tropical)
		out!.n += x.n + y.n
	end

	"""
		i_einsum!(ixs, xs, iy, y::AbstractArray{T})

	A naive reversible implementation of `i_einsum` function for tropical numbers.
		* `ixs`: input tensor indices,
		* `xs`: input tensors,
		* `iy`: output tensor indices,
		* `y`: accumulated tensor, notice it is initialized to 0 as output!

	# NOTE: this function is general purposed and slow!
	"""
	@i function i_einsum!(ixs, xs, iy, y::AbstractArray{T}) where {T<:Tropical}
		@routine begin
			# outer legs and inner legs
			outer_indices ← unique(iy)
			inner_indices ← setdiff(TupleTools.vcat(ixs...), outer_indices)

			# find size for each leg
			all_indices ← TupleTools.vcat(ixs..., iy)
			all_sizes ← TupleTools.vcat(size.(xs)..., size(y))
			outer_sizes ← [map(i->all_sizes[i], indexin(outer_indices, [all_indices...]))...]
			inner_sizes ← [map(i->all_sizes[i], indexin(inner_indices, [all_indices...]))...]

			# cartesian indices for outer and inner legs
			outer_ci ← CartesianIndices((outer_sizes...,))
			inner_ci ← CartesianIndices((inner_sizes...,))

			# for indexing tensors (leg binding)
			indices ← (outer_indices..., inner_indices...)
			locs_xs ← map(ix->map(i->findfirst(isequal(i), indices), ix), ixs)
			locs_y ← map(i->findfirst(isequal(i), outer_indices), iy)
		end
		i_loop!(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
		~@routine
	end

	"""take an index subset from `ind`"""
	index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(Tuple(ind), locs))

	"""
	loop and accumulate products to y, the GPU version, the CPU version.
	"""
	@i function i_loop!(locs_xs::NTuple{N,Any}, xs::NTuple{N, AbstractArray}, locs_y, y::AbstractArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T<:Tropical}
		@invcheckoff @inbounds for i in outer_ci
			@routine begin
				el ← zero(T)
				ind_y ← outer_ci[i]
				iy ← index_map(ind_y, locs_y)
				branch_keeper ← zeros(Bool, size(inner_ci)...)
				pl ← ones(T, size(inner_ci)...)
				for ind_x in inner_ci
					pli ← one(T)
					ind_xy ← CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
					for I=1:N
						pli *= xs[I][index_map(ind_xy, locs_xs[I])]
					end
					if (el.n < pli.n, branch_keeper[ind_x])
						FLIP(branch_keeper[ind_x])
						SWAP(el, pli)
					end
					SWAP(pl[ind_x], pli)
					pli → one(T)
				end
			end
			@inbounds y[iy] *= el
			~@routine
		end
	end
	
	# patches
	Base.zero(x::Tropical{GVar{T,GT}}) where {T,GT} =zero(Tropical{GVar{T,GT}})
    Base.zero(::Type{Tropical{GVar{T,T}}}) where T = Tropical(GVar(zero(Tropical{T}).n, zero(T)))
	
    NiLang.AD.GVar(x::Tropical{T}) where T = Tropical(GVar{T,T}(x.n, zero(T)))

	function NiLangCore.deanc(x::T, v::T) where T<:Tropical
		x === v || NiLangCore.deanc(content(x), content(v))
	end
end

# ╔═╡ d8669492-8c98-11eb-1cf5-374da9bd4556
html"<button onclick='present()'>present</button>"

# ╔═╡ f0b4563a-8b0b-11eb-3085-458f5d9f88b8
md"""
```math
\newcommand{\comment}[1]{{\bf  \color{blue}{\text{◂~ #1}}}}
```
"""

# ╔═╡ ce5af22a-8830-11eb-13c8-49c680526bd9
md"# Cool automatic differentiation applications

-- Jinguo Liu"

# ╔═╡ 22768818-8a95-11eb-1cfe-69534f8b0314
md"""
* What is automatic differentiation (AD)?
    * A true history of AD
    * Forward mode AD
    * Reverse mode AD 
        * primitves on tensors (including tensorflow, pytorch et al.)
        * primitves on elementary instructions (usually source code transformation based)
        * defined on a reversible program
* Some applications in **scientific computing**
    * solving the graph embedding problem
    * inverse engineering a hamiltonian
    * obtaining maximum independent set (MIS) configurations
    * towards differentiating `expmv` ``\comment{will be used in our emulator}``
"""

# ╔═╡ 68be48da-8a93-11eb-226b-7b1f2be99cb6
md"""
## The true history of automatic differentiation
"""

# ╔═╡ da08c542-8a93-11eb-3375-d79ccd2de122
md"""
* 1964 ~ Robert Edwin Wengert, A simple automatic derivative evaluation program. ``\comment{first forward mode AD}``
* 1970 ~ Seppo Linnainmaa, Taylor expansion of the accumulated rounding error. ``\comment{first backward mode AD}``
* 1986 ~ Rumelhart, D. E., Hinton, G. E., and Williams, R. J., Learning representations by back-propagating errors.
* 1992 ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation. ``\comment{foundation of source code transformation based AD.}``
* 2000s ~ The boom of tensor based AD frameworks for machine learning.
* 2018 ~ People re-invented AD as differential programming ([wiki](https://en.wikipedia.org/wiki/Differentiable_programming) and this [quora answer](https://www.quora.com/What-is-Differentiable-Programming).)
![](https://qph.fs.quoracdn.net/main-qimg-fb2f8470f2120eb49c8142b08d9c4132)
* 2020 ~ Me, Differentiate everything with a reversible embeded domain-specific language ``\comment{AD based on reversible programming}``.
"""

# ╔═╡ 0c80347a-8aff-11eb-0862-1d551e3af00b
md"## Forward mode automatic differentiation"

# ╔═╡ f55f9cba-8afe-11eb-3bf9-e3e5ecbf3a56
md"""
Forward mode AD attaches a infitesimal number $\epsilon$ to a variable, when applying a function $f$, it does the following transformation
```math
\begin{align}
    f(x+g \epsilon) = f(x) + f'(x) g\epsilon + \mathcal{O}(\epsilon^2)
\end{align}
```

The higher order infinitesimal is ignored. 

**In the program**, we can define a *dual number* with two fields, just like a complex number
```
f((x, g)) = (f(x), f'(x)*g)
```
"""

# ╔═╡ 662c94c4-8b0a-11eb-39a2-9f37e689fbd3
res = sin(Dual(π/4, 2.0))

# ╔═╡ 9e24dd6c-8b0a-11eb-020a-a1d7bf2e87bf
res === Dual(sin(π/4), cos(π/4)*2.0)

# ╔═╡ fc682956-8b00-11eb-3102-952881130049
md"
We can apply this transformation consecutively, it reflects the chain rule.
```math
\begin{align}
\frac{\partial \vec y_{i+1}}{\partial x} &= \boxed{\frac{\partial \vec y_{i+1}}{\partial \vec y_i}}\frac{\partial \vec y_i}{\partial x}\\
&\text{local Jacobian}
\end{align}
```
"

# ╔═╡ 167cbe40-8bf5-11eb-077b-bdd217c9bd78
let
	lb = textstyle(:math, fontsize(8), width=0.5, height=0.5)
	tb = textstyle(:default, fontsize(10), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("white"), Compose.stroke("black"); r=0.08)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	ebr = bondstyle(:default, Compose.stroke("red"), linewidth(0.5mm))
	ebd = bondstyle(:default, linewidth(0.5mm), dashed=true)
	eba = bondstyle(:default, linewidth(0.5mm), Compose.arrow(), Compose.stroke("red"), Compose.fill("red"))
		
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, fill("red"), θ=π/2-atan((y .- x)...)-1π/6)
		ebr >> (x, y)
		t >> mid
	end
	
	Compose.set_default_graphic_size(15cm, 5cm)
	x = (0.1, 0.5)
	fi0 = (0.35, 0.5)
	fi1 = (0.7, 0.5)
	fi2 = (1.0, 0.5)
	img = canvas() do
		nb >> fi0
		nb >> fi1
		lb >> (fi0 .- (0.05, 0.1), "f_{i-1}")
		lb >> (fi1 .- (0.02, 0.1), "f_{i}")
		lb >> (x, "x")
		lb >> ((fi1 .+ fi0) ./ 2 .- (0.02, 0.0), raw"\vec{y}_{i}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i+1}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.05, 0.0), "\\vec{y}_{i+1}")
		lb >> (x .- (0.00, 0.25), raw"\color{red}{1}")
		lb >> ((fi1 .+ fi0) ./ 2 .- (0.05, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i}}{\partial x}}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i+1}}{\partial x}}")
		ebd >> (x, fi0)
		eb >> (fi0, fi1)
		eb >> (fi1, fi2)
		#arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
		arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
	end
	img
end

# ╔═╡ d29a56a4-8d86-11eb-1749-0972a594a0e4
let
	x = Dual(π/4, 1.0)
	for i=1:10
		x = sin(x)
	end
	x
end

# ╔═╡ 0ba6245c-8bf5-11eb-1005-1b72ffae3412
md"""
**Example:** Computing two gradients $\frac{\partial z\sin x}{\partial x}$ and $\frac{\partial \sin^2x}{\partial x}$ at one sweep
"""

# ╔═╡ aa316d52-8b00-11eb-1e7e-43e6f504977f
let
	lb = textstyle(:math, fontsize(8), width=1.0, height=0.5)
	tb = textstyle(:default, fontsize(3.5), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(4.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("black"), Compose.stroke("transparent"); r=0.05)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	
	x_x = (0.1, 0.25)
	x_y = (0.9, 0.5)
	x_y2 = (0.9, 0.25)
	x_z = (0.3, 0.5)
	x_sin = (0.3, 0.25)
	x_mul = (0.5, 0.5)
	x_square = (0.5, 0.25)
	
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, θ=π/2-atan((y .- x)...)-1π/6)
		eb >> (x, y)
		t >> mid
	end

	img = canvas() do
		nb >> x_sin
		nb >> x_mul
		nb >> x_square
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		tb_big >> (x_square, "^2")
		arrow(x_sin, x_mul)
		arrow(x_x, x_sin)
		arrow(x_mul, x_y)
		arrow(x_square, x_y2)
		arrow(x_z, x_mul)
		arrow(x_sin, x_square)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.02, 0.04), "x+ϵˣ")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (0.08, 0.04), "sin(x)+cos(x)*ϵˣ")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.04, 0.055), "z*sin(x)\n+z*cos(x)*ϵˣ")
		tb >> ((x_y2 .+ x_square) ./ 2 .- (-0.04, 0.055), "sin(x)^2\n+2*sin(x)*cos(x)*ϵˣ")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.02), "z")
	end
	
	Compose.set_default_graphic_size(100mm, 100mm/2)
	Compose.compose(context(0, -0.15, 1, 2), img)
end

# ╔═╡ 8a55b3e4-8d67-11eb-203e-0b49c8e3e4aa
md"so the gradients are $z\cos x$ and $2\sin x\cos x$"

# ╔═╡ f0468078-8b0c-11eb-1bdb-9f6f9496dcf2
md"""
**What if we want to compute gradients for multiple inputs?**

The computing time grows **linearly** as the number of variables that we want to differentiate. But does not grow significantly with the number of outputs.
"""

# ╔═╡ 6335bc36-8b15-11eb-3731-5b76661d10fa
md"""
## Reverse mode automatic differentiation

"""

# ╔═╡ 2f60bb2c-8b2e-11eb-04f9-b79f81439af3
md"On the other side, the back-propagation can differentiate **many inputs** with respect to a **single output** efficiently"

# ╔═╡ 61be9354-8b25-11eb-3205-db9b03a18f18
md"""
```math
\begin{align}
    \frac{\partial \mathcal{L}}{\partial \vec y_i} = \frac{\partial \mathcal{L}}{\partial \vec y_{i+1}}&\boxed{\frac{\partial \vec y_{i+1}}{\partial \vec y_i}}\\
&\text{local jacobian?}
\end{align}
```
"""

# ╔═╡ 30f3b718-8bfc-11eb-02cc-777ebc108429
let
	lb = textstyle(:math, fontsize(8), width=0.5, height=0.5)
	tb = textstyle(:default, fontsize(10), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("white"), Compose.stroke("black"); r=0.08)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	ebr = bondstyle(:default, Compose.stroke("red"), linewidth(0.5mm))
	ebd = bondstyle(:default, linewidth(0.5mm), dashed=true)
	eba = bondstyle(:default, linewidth(0.5mm), Compose.arrow(), Compose.stroke("red"), Compose.fill("red"))
		
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, fill("red"), θ=π/2-atan((y .- x)...)-1π/6)
		ebr >> (x, y)
		t >> mid
	end
	
	Compose.set_default_graphic_size(15cm, 5cm)
	x = (0.1, 0.5)
	fi0 = (0.35, 0.5)
	fi1 = (0.7, 0.5)
	fi2 = (0.9, 0.5)
	img = canvas() do
		nb >> fi0
		nb >> fi1
		lb >> (fi0 .- (0.02, 0.1), "f_{i}")
		lb >> (fi1 .- (0.05, 0.1), "f_{i+1}")
		lb >> (fi2 .- (0.05, 0.0), raw"\mathcal{L}")
		lb >> ((fi0 .+ x) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i+1}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.05, 0.0), "\\vec{y}_{i+1}")
		lb >> (fi2 .- (0.05, 0.25), raw"\color{red}{1}")
		lb >> ((fi0 .+ x) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i}}{\partial x}}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i+1}}{\partial x}}")
		ebd >> (fi1, fi2)
		eb >> (fi0, fi1)
		eb >> (x, fi0)
		#arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
		arrow( (fi0 .+ fi1) ./ 2 .+ (-0.08, -0.3), (fi0 .+ x) ./ 2 .+ (0.05, -0.3),)
	end
	img
end

# ╔═╡ f67d1d28-8d6c-11eb-2e47-09e0aece0967
md"### How to visite local Jacobians in the reversed order? "

# ╔═╡ 0ebb6588-8b52-11eb-21c0-4fd9e70a77e7
md"
**Design Decision**

1. Compute forward pass and caching inetermediate results into a global stack $\Sigma$ （packages except NiLang），
2. reversible programming."

# ╔═╡ 085145dc-8c10-11eb-12dc-3be1d68fe85c
md"""
**Example:** Computing the gradient $\frac{\partial z\sin x}{\partial x}$ and $\frac{\partial z\sin x}{\partial z}$ by back propagating cached local information.
"""

# ╔═╡ 80caf3ba-8b1f-11eb-3030-5378078e2df9
let
	lb = textstyle(:math, fontsize(10), width=1.0, height=0.5)
	tb = textstyle(:default, fontsize(3.5), Compose.font("monospace"))
	tbc = textstyle(:default, fontsize(3.5), fill("red"), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(4), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("black"), Compose.stroke("transparent"); r=0.05)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	
	x_x = (0.1, 0.2)
	x_y = (0.9, 0.5)
	x_z = (0.1, 0.7)
	x_sin = (0.3, 0.3)
	x_mul = (0.5, 0.5)

	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, θ=π/2-atan((y .- x)...)-1π/6)
		eb >> (x, y)
		t >> mid
	end
	img1 = canvas() do
		nb >> x_sin
		nb >> x_mul
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		arrow(x_sin, x_mul)
		arrow(x_x, x_sin)
		arrow(x_mul, x_y)
		arrow(x_z, x_mul)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x \n push(Σ,x)")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.15, 0.04), "s = sin(x) \n push(Σ,s)")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.04), "y = z*sin(x)")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "z\n push(Σ,z)")
	end
	img2 = canvas() do
		nb >> x_sin
		nb >> x_mul
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		arrow(x_mul, x_sin)
		arrow(x_sin, x_x)
		arrow(x_y, x_mul)
		arrow(x_mul, x_z)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x = pop(Σ)\nx̄ = cos(x)*s̄")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.12, 0.04), "z = pop(Σ)\ns̄ = z*ȳ")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.06), "y\nȳ=1")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "s = pop(Σ)\nz̄ = s*ȳ")
	end
	
	Compose.set_default_graphic_size(150mm, 75mm/1.4)
	Compose.compose(context(), 
	(context(0, -0.1, 0.5, 1.4), img1),
	(context(0.5, -0.1, 0.5, 1.4), img2)
	)
end



# ╔═╡ 8bed03b2-8c1b-11eb-1043-23ff31b46991
md"Here, we use $\overline y$ for $\frac{\partial \mathcal{L}}{\partial y}$, which is also called the adjoint."

# ╔═╡ 0790392e-8d04-11eb-3585-d17e29d379d9
md"### Primitives on different scales"

# ╔═╡ 330ccb76-8d04-11eb-3bfd-434a93465975
md"We call the leaf nodes defining AD rules \"**primitives**\""

# ╔═╡ dfb179fe-8cae-11eb-24b6-3bdfa57a0d45
md"
**Design Decision**

* A: If we define primitives on **arrays**, we need tons of manually defined backward rules. (Jax, Pytorch, Zygote.jl, ReverseDiff.jl et al.)
* B: If we define primitives on **scalar instructions**, we will have worse tensor performance. (Tapenade, Adept, NiLang et al.)

*Note*: Here, implementing AD on scalars means specifically the **optimal checkpointing** approach, rather than a package like Jax, Zygote and ReverseDiff that having scalar support.
"

# ╔═╡ f2c75336-8cab-11eb-09ce-e99acb20aee8
let
	w, h = 0.22, 0.1
	lb = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"))
	lb2 = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"), fill("red"))
	tb = Compose.compose(context(), Compose.text(0.0, 0.0, ""), fontsize(3), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3), fill("white"), Compose.font("monospace"))
	eb = bondstyle(:default, linewidth(0.5mm))
	ar = bondstyle(:default, linewidth(0.3mm), Compose.arrow())
	xprog = (0.25, 0.15)
	xtensors = (0.25, 0.5)
	t1 = (0.5, 0.15)
	t2 = (0.5, 0.5)
	t3 = (0.5, 0.85)
	xscalars2 = (0.25, 0.85)
	
	function box(loc, text; color="black")
		(color=="black" ? lb : lb2) >> loc
		tb_big >> (loc, text)
	end
	Compose.set_default_graphic_size(10cm, 5cm)
	canvas() do
		box(xprog, "Program")
		ar >> (xprog, xtensors .+ (0, -h-0.03))
		#ar >> (xprog, xscalars .+ (-w/2, -h-0.03))
		ar >> (xtensors, xscalars2 .+ (0, -h-0.05))
		box(xtensors, "Functions on arrays")
		#box(xscalars, "Functions on Scalars")
		box(xscalars2, "Finite instructions"; color="red")
		tb >> (t1, "Neural networks")
		tb >> (t2, "matrix multiplication")
		tb >> (t3, "+, -, *")
	end
end

# ╔═╡ d1760114-8b52-11eb-3241-252292bf96ac
html"""
<table>
<tr>
<th width=200></th>
<th width=300>on tensors</th>
<th width=300>on finite instructions</th>
</tr>
<tr style="vertical-align:top">
<td>meaning</td>
<td>defining backward rules manully for functions on tensors</td>
<td>defining backward rules on a limited set of basic scalar operations, and generate gradient code using source code transformation</td>
</tr>
<tr style="vertical-align:top">
<td>pros and cons</td>
<td>
<ol>
<li style="color:green">Good tensor performance</li>
<li style="color:green">Mature machine learning ecosystem</li>
<li style="color:red">Need to define backward rules manually</li>
</ol>
</td>
<td>
<ol>
<li style="color:green">Reasonalbe scalar performance</li>
<li style="color:red">hard to utilize GPU kernels (except NiLang.jl) and BLAS</li>
</ol>
</td>
<td>
</td>
</tr>
<tr style="vertical-align:top">
<td>packages</td>
<td>Jax<br>PyTorch</td>
<td><a href="http://tapenade.inria.fr:8080/tapenade/">Tapenade</a><br>
<a href="http://www.met.reading.ac.uk/clouds/adept/">Adept</a><br>
<a href="https://github.com/GiggleLiu/NiLang.jl">NiLang.jl</a>
</td>
</tr>
</table>
"""

# ╔═╡ 8cab36f8-8c1a-11eb-009a-3d6dbdb83d85
md"""
## The AD ecosystem in Julia

Please check JuliaDiff: [https://juliadiff.org/](https://juliadiff.org/)

A short list:
* Forward mode AD: ForwardDiff.jl
* Reverse mode AD (tensor): ReverseDiff.jl/Zygote.jl
* Reverse mode AD (scalar): NiLang.jl

Warnings
* The main authors of `Tracker`, `ReverseDiff` and `Zygote` are not maintaining them anymore.
"""
#=
|       |   Rules | Favors Tensor? | Type |
| ---- | ---- | --- | --- |
|  Zygote   |  C  |  ✓   |   R     |
|  ReverseDiff  |  D    | ✓    | R |
|  Nabla   |  D→C  |   ✓  |   R     |
|  Tracker  |  D    | ✓    | R |
|  Yota   |  C  |  ✓   |     R   |
|  NiLang   |  -  |  ×   |  R      |
|  Enzyme   |  -  |  ×   |  R      |
|  ForwardDiff   |  -  |  ×   |    F    |
|  Diffractor   |  ?  |  ?   |  ?      |

* R: reverse mode
* F: forward mode
* C: ChainRules
* D: DiffRules
"""
=#

# ╔═╡ 01e46500-8ced-11eb-04b7-a1e05a81b1b2
md"# Quick summary
1. The history of AD is longer than many people have thought. People are most familar with *reverse mode AD with primitives implemented on tensors* that brings the boom of machine learning. There are also AD frameworks that can differentiate a general program directly, which does not require users defining AD rules manually.
2. **Forward mode AD** propagate gradients forward, it has a computational overhead propotional to the number of input parameters.
2. **Backward mode AD** propagate gradients backward, it has a computational overhead propotional to the number of output parameters.
    * primitives on **tensors** v.s. **scalars**
    * reverse the program tape by **caching/checkpointing** v.s. **reversible programming**
4. Julia has one of the most active AD community!

#### Forward v.s. Backward
when is forward mode AD more useful?

* It is often combined with backward mode AD for obtaining Hessians (forward over backward).
* Having <20 input parameters.

when is backward mode AD more useful?
* In most variational optimizations, especially when we are training a neural network with ~ 100M parameters.
"

# ╔═╡ eb0cba98-8a54-11eb-132f-6320f3893da9
md"## 1. Embedding a peterson Graph"

# ╔═╡ 43c6b8ec-8a79-11eb-2ff1-cb8fd958b693
md"""
One day, A postdoc of Anders Sandvik Jun Takahashi went to me, said "Hey, Jinguo, can you help me figure out what is the minimum embedding dimension of a Peterson graph?"

A Peterson graph is a famous 3-regular graph with very high symmetry. It is well know to graph theory people. It looks like
"""

# ╔═╡ 99839afc-8a83-11eb-3f38-1bef2960f969
md"It has 10 vertices, 15 edges, while these vertices are all equivalent to each other. By embedding a graph into a k-dimensional space, it requires
1. assigning a k-dimensional vector to each node as the Euclidean coordinate,
2. the distance between each pair of connected nodes are the same, meanwhile, the distance between each pair of disconnected nodes are same too.
3. the distance between disconnected vertices are larger than connect vertices"

# ╔═╡ a8a92160-88c5-11eb-0a7e-cbec45b627f0
# connected vertex-pairs in a petersen graph
const L1 = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
    (8, 10), (7, 10), (7, 9), (6, 9)];

# ╔═╡ b1016e9c-88c5-11eb-12a5-41a914a794d3
# disconnected vertex-pairs in a petersen graph
const L2 = [(1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
    (1, 10), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
    (2, 10), (3, 5), (3, 6), (3, 7), (3, 9), (3, 10),
    (4, 6), (4, 7), (4, 8), (4, 10), (5, 6), (5, 7),
    (5, 8), (5, 9), (6, 7), (6, 10), (7, 8), (8, 9),
    (9, 10)];

# ╔═╡ 385638ea-8a81-11eb-166b-a9caf2b2792f
let
	L1 = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
		(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
		(8, 10), (7, 10), (7, 9), (6, 9)]
	x1 = (0.0, -0.45)
	x2 = (0.0, -0.25)
	nodes = [
		[Viznet.rot(x1..., 2π/5*i) .+ 0.5 for i=0:4]...,
		[Viznet.rot(x2..., 2π/5*i) .+ 0.5 for i=0:4]...
	]
	Compose.set_default_graphic_size(8cm, 8cm)
	nb = nodestyle(:circle, fill("white"), Compose.stroke("black"))
	eb = bondstyle(:default)
	eb2 = bondstyle(:default, Compose.stroke("#DDDDDD"))
	canvas() do
		for n in nodes
			nb >> n
		end
		for (i,j) in L1
			eb >> (nodes[i], nodes[j])
		end
		for (i,j) in L2
			eb2 >> (nodes[i], nodes[j])
		end
	end
end

# ╔═╡ 38a9382e-8b58-11eb-3e0b-656260c672cd
md"For dimension $k\in 1,2,\dots,10$, we assign a coordinate to each vertex. Then we define the loss as 
```math
\begin{align}
    \begin{split}
D_1 &= \{d_{(i,j)} | (i,j) \in L_1\}\\
D_2 &= \{d_{(i,j)} | (i,j) \in L_2\}\\
        \mathcal{L} &= {\rm var}(D_1) + {\rm var}(D_2) \\
        &+\exp({\rm relu}({\rm mean}(D_1)- {\rm mean}(D_2) + 0.1)) - 1 \comment{if $d_2$ < $d_1$, punish}
    \end{split}
\end{align}
```

"

# ╔═╡ 1d503598-8d7c-11eb-1cf4-cf0755194043
md"""
`relu` is defined as `x > 0 ? x : 0`
"""

# ╔═╡ bf5702ae-88c5-11eb-2bb2-d7d5cf4eccdf
@i function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    for i=1:length(x1)
		@routine begin
			diff ← zero(T)
			diff += x1[i] - x2[i]
		end
        dist! += diff ^ 2
        ~@routine
    end
end

# ╔═╡ 9a474d6a-885f-11eb-2db5-2145c0b9183f
"""The loss of graph embedding problem."""
@i function embedding_loss(out!::T, x) where T
	@routine @invcheckoff begin
		@zeros T v1 varsum1 varsum2 s1 s2 m1 v2 m2 diff
		d1 ← zeros(T, length(L1))
		d2 ← zeros(T, length(L2))
		# 1. compute distances
        for i=1:length(L1)
            sqdistance(d1[i], x[:,L1[i][1]],x[:,L1[i][2]])
        end
        for i=1:length(L2)
            sqdistance(d2[i], x[:,L2[i][1]],x[:,L2[i][2]])
        end
		# 2. compute variances
        NiLang.i_var_mean_sum(v1, varsum1, m1, s1, d1)
        NiLang.i_var_mean_sum(v2, varsum2, m2, s2, d2)
        m1 -= m2 - 0.1
    end
    out! += v1 + v2
    if m1 > 0
        # to ensure mean(v2) > mean(v1)
        # if mean(v1)+0.1 - mean(v2) > 0, punish it.
        out! += exp(m1)
        out! -= 1
    end
    ~@routine
end

# ╔═╡ 5ef95fa8-8a50-11eb-35e7-9354bf50b0e7
md"Seed = $(@bind seed Slider(1:10000))"

# ╔═╡ 8f2d529c-8a50-11eb-1910-29b61b1b8f82
md"dimension $(@bind dimension NumberField(1:10; default=5))"

# ╔═╡ 9b86ac04-8a4c-11eb-278e-5777f3552f25
x_minimizer, x_minimum = let
	Random.seed!(seed)
	x = randn(dimension,10)
	# `NiLang.AD.gradient` to obtain the gradients
	res = Optim.optimize(x->embedding_loss(0.0, x)[1], x->NiLang.AD.gradient(embedding_loss, (0.0, x); iloss=1)[2], x, LBFGS(), Optim.Options(f_abstol=1e-12, f_reltol=1e-12, g_abstol=1e-12, g_reltol=1e-12), inplace=false)
	res.minimizer, res.minimum
end;

# ╔═╡ c697cf2c-8d76-11eb-1817-132326c4a92e
x_minimum

# ╔═╡ 8cae97aa-8d76-11eb-1740-0752e4297de2
d1s = [norm(x_minimizer[:,a] .- x_minimizer[:,b]) for (a, b) in L1]

# ╔═╡ 01e6cbb4-8d77-11eb-2519-23c89d5de971
d2s = [norm(x_minimizer[:,a] .- x_minimizer[:,b]) for (a, b) in L2]

# ╔═╡ 071bc508-8d77-11eb-01e6-950e171e6b45
mean(d2s)/mean(d1s)

# ╔═╡ 77562ce2-8a79-11eb-389a-a35e42e65171
md"""
His work of finding the SO(5) symmetric tensor order representation is later published as

"Valence-bond solids, vestigial order, and emergent SO(5) symmetry in a two-dimensional quantum magnet." (Phys. Rev. Research 2, 033459, Jun Takahashi, Anders W. Sandvik)
"""

# ╔═╡ 0f5cc712-8a84-11eb-0f31-91b30ad895af
md"## 2. Inverse engineering a Hamiltonian"

# ╔═╡ be1a31d8-8a91-11eb-2ead-fb4b7de74afc
md"""
This problem is from "Notes on Adjoint Methods for 18.335", Steven G. Johnson

Consider a 1D Shrodinger equation
```math
\left[-\frac{d^2}{dx^2} + V(x)\right]\Psi(x) = E\Psi(x), x \in [-1,1]
```

"""

# ╔═╡ 25bc662a-8c14-11eb-0689-45ac05c280aa
md"We can solve its gound state numerically by discretizing the space and diagonalize the Hamiltonian matrix. The Hamiltonian matrix is

```math
A = \frac{1}{Δx^2}\left(
\begin{matrix}
2 & -1 & 0 & \ldots & 0 & -1\\
-1 & 2 & -1 & 0 & \ldots & \\
0 & -1 & 2 & -1 & 0 & \ldots \\
\vdots &  &  & \ddots &  & \\
 & & & -1 & 2 & -1\\
-1 & 0 & \ldots & 0 & -1 & 2
\end{matrix}
\right) + {\rm diag}(V)
```
"

# ╔═╡ 3aa7ea18-8c15-11eb-010e-9d6a9e7dd952
md"where the matrix size is equal the descretized lattice size"

# ╔═╡ 4eda1c4c-8b5f-11eb-2485-6b34cdef4ef8
dx = 0.02;

# ╔═╡ 40ec8444-8b5f-11eb-0de5-97ed546aa4a1
xgrid = -1.0:dx:1.0;

# ╔═╡ cf8235b6-8b5d-11eb-2997-41da626b503f
@i function hamiltonian!(a, x, V::AbstractVector{T}) where T
	@routine begin
		@zeros T dx2 invdx2
		n ← length(x)
		dx2 += (@const Float64(x.step))^2
		invdx2 += 1/dx2
	end
	@safe @assert size(a) == (n, n)
	for i=1:n
		a[i, i] += 2 * invdx2
		a[i, i] += V[i]
		a[i, mod1(i+1, n)] -= invdx2
		a[mod1(i+1, n), i] -= invdx2
	end
	~@routine
end

# ╔═╡ 82c60414-8c4a-11eb-27c2-e9294ecbebba
hamiltonian(x, V) = hamiltonian!(zeros(length(x), length(x)), x, V)[1]

# ╔═╡ 3dcc9014-8c4b-11eb-0a15-a5e9e495c7b7
hamiltonian(xgrid, randn(length(xgrid)))

# ╔═╡ eb77b59a-8c9b-11eb-2683-c150952cf85b
md"Because we are going to use Zygote (with rules set defined in ChainRules)"

# ╔═╡ 63d9c6ce-8c4a-11eb-384d-a1fa4fe5ccd6
function ChainRules.rrule(::typeof(hamiltonian), x, V)
	y = hamiltonian(x, V)
	function hamiltonian_pullback(Δy)
		gV = NiLang.AD.grad((~hamiltonian!)(GVar.(y, Δy), x, GVar.(V))[3])
		return (ChainRules.NO_FIELDS, ChainRules.DoesNotExist(), gV)
	end
	return y, hamiltonian_pullback
end

# ╔═╡ 8e9f00ca-8c15-11eb-3079-a147fd2b0149
md"We want the ground state be a house."

# ╔═╡ 20e1c2ae-8b5f-11eb-334c-d74f310f9cc5
ψ0 = [abs(xi)<0.5 ? 1 - abs(xi) : 0 for xi in xgrid]; normalize!(ψ0);

# ╔═╡ 88757052-8b5f-11eb-189c-5b74c7c9acde
plot(xgrid, ψ0)

# ╔═╡ 5b357e9e-8c15-11eb-1ce8-6b91fac05cc0
md"So we define a loss function
```math
\begin{align}
E, \psi &= {\rm eigensolve}(A)\\
\mathcal{L} &= \sum_i |(|(\psi_0)_i| - |(\psi_G)_i|)|
\end{align}
```
"

# ╔═╡ bd15d5ca-8c18-11eb-33a8-7306b70c02e0
md"where $\psi_G$ is state vector in $\psi$ that corresponds to the minimum energy."

# ╔═╡ 2a7a0da4-8b63-11eb-2d1c-9b852a39dacf
function solve_wave(x, V)
	a = hamiltonian(x, V)
	ψ = LinearAlgebra.eigen(LinearAlgebra.Hermitian(a)).vectors[:,1]
end

# ╔═╡ a0918e24-8b61-11eb-3f39-69bd8ce0c618
function loss(x, V, ψ0)
	ψ = solve_wave(x, V)
	sum(map(abs, map(abs, ψ) - map(abs, ψ0))) * dx
end

# ╔═╡ eaa8f57e-8b61-11eb-28b7-bf487a201b80
loss(xgrid, randn(length(xgrid)), ψ0)

# ╔═╡ 4543b0b2-8b64-11eb-0984-71f0a8027e6b
solve_wave(xgrid, randn(length(xgrid))) |> norm

# ╔═╡ c21f7c16-8b66-11eb-1502-8362d656896a
loss(xgrid, randn(length(xgrid)), ψ0)

# ╔═╡ 4b667f7a-8b60-11eb-1773-6904f0858ad7
Zygote.gradient(v->loss(xgrid, v, ψ0), randn(length(xgrid)))

# ╔═╡ a0a78f52-8b62-11eb-2e7a-4d3f975461ce
@bind clock Clock(0.1)

# ╔═╡ e64c8508-8b62-11eb-191c-45ec2ae6cd8c
it = adam(v->loss(xgrid, v, ψ0), x->Zygote.gradient(v->loss(xgrid, v, ψ0), x)[1], randn(length(xgrid)); η=1.0);

# ╔═╡ a910ae42-8b62-11eb-0474-7f81c709a77c
let
	clock
	state = step!(it)
	v = minimizer(state)
	ψ = solve_wave(xgrid, v)
	@show loss(xgrid, v, ψ0)
	plot(xgrid, abs.(ψ); label="ψ")
	plot!(xgrid, abs.(ψ0); label="ψ0")
	plot!(xgrid, normalize(v); label="V")
end |> PlutoUI.as_svg

# ╔═╡ a1e7dd02-8bc8-11eb-29e0-91df4126872a
md"""## 3. Obtaining MIS configurations"""

# ╔═╡ 7919105e-8ce4-11eb-3f31-ff0fc17ca375
md"We are able to get the weighted maximum independent set (MIS) size of the following graph

```math
S = \max_{\vec s}\left(\sum_i w_i s_i-\infty \sum_{ij\in E} s_i s_j\right), s_i \in \{0,1\}
```
where $s_i$ and $w_i$ are the configuration (in MIS: 1, not in MIS: 0) and weight of node $i$.
"

# ╔═╡ 1e954d20-8ce7-11eb-078e-0baaaa7132d4
md"Question: how to get the configuration with MIS?"

# ╔═╡ ee3c1284-8d7d-11eb-3f4c-59ceabbd180c
md"The optimal configuration is a gradient!"

# ╔═╡ 32fdc4e0-8ce7-11eb-3acd-ed90c6e2e780
md"""
```math
\frac{\partial S}{\partial w_i} = \begin{cases}
1 & s_i \in \vec s_{\rm max}\\
0 & otherwise
\end{cases}
```
"""

# ╔═╡ 083f13f0-8cea-11eb-0b3a-a96ab05aece9
md"The actual problem is harder, if we fix the boundary configurations `a, b, c, d`, what is the optimal configurations for interior?"

# ╔═╡ f96dc3b4-8bcf-11eb-15ec-232d15b3e075
function vizconfig(nodes, edges, config=zeros(Int, length(nodes)))
	Compose.set_default_graphic_size(12cm, 12cm)
	tb = textstyle(:default, fill("white"))
	nb = nodestyle(:default)
	nb2 = nodestyle(:default, fill("red"))
	eb = bondstyle(:default)
	canvas() do
		for (i, (t, p)) in enumerate(nodes)
			(config[i]==1 ? nb2 : nb) >> p
			tb >> (p, t)
		end
		for (i,j) in edges
			eb >> (nodes[i].second, nodes[j].second)
		end
	end
end;

# ╔═╡ beee2d1e-8bcf-11eb-295c-a3b45bfc3aa8
nodes_simple = let
	a = 0.12
	ymid = xmid = 0.5
	X = 0.33
	Y = 0.17
	D = 0.15
	y = [ymid-Y, ymid-Y+D, ymid-a/2, ymid+a/2, ymid+Y-D, ymid+Y]
	x = [xmid-X, xmid-X+D, xmid-1.5a, xmid-a/2, xmid+a/2, xmid+1.5a, xmid+X-D, xmid+X]
	xmin, xmax, ymin, ymax = x[1], x[end], y[1], y[end]
	["a"=>(xmid, y[1]), "b"=>(xmin, ymid), "c"=>(xmid, ymax), "d"=>(xmax, ymid),
		"i"=>(x[3], y[3]), "j"=>(x[4], y[3]),
		"k"=>(x[5], y[3]), "l"=>(x[6], y[3]), "m"=>(x[3], y[4]),
		"n"=>(x[4], y[4]), "o"=>(x[5], y[4]), "p"=>(x[6], y[4])]
end;

# ╔═╡ 13bc3822-8bd0-11eb-1caa-1d49b3b74272
function find_edges(nodes, distance)
	edges = Tuple{Int,Int}[]
	for (i, p) in enumerate(nodes)
		for (j,p2) in enumerate(nodes)
			if i<j && sqrt(sum(abs2, p2 .- p)) < distance
				push!(edges, (i,j))
			end
		end
	end
	edges
end

# ╔═╡ 06f15870-8bd0-11eb-3b4c-db7d9e3e7adf
edges_simple = find_edges(getindex.(nodes_simple, 2), 0.23);

# ╔═╡ 8d25f9e0-8bcc-11eb-1df4-33556074d6ff
let
	img = vizconfig(nodes_simple, edges_simple)
	Compose.set_default_graphic_size(14cm, 7cm)
	Compose.compose(context(0.0, -0.5, 1.0, 2.0), img)
end

# ╔═╡ dec89520-8bcf-11eb-3ab4-97cc22ca2368
"""
* `optsize` stores the MIS size the configuration specified by the 4th argument.
* `out` stores the contraction results, which is a 2^4 tensor. The entries represents the MIS size for a given boundary configuration (e.g. MIS size is 2 for a = b = c = d = 0)
* `x` is the node weights.
* `config` is the boundary configurations.
"""
@i function compute_mis(optsize, out::Array{T,4}, x::Vector{T}, config) where T
	@routine begin
		# defining contraction patterns
		ixs ← (('a',), ('b',), ('c',), ('d',), ('i',), ('j',), ('k',), ('l',), ('m',), ('n',),('o',), ('p',),('a', 'i'), ('a', 'j'), ('a', 'k'), ('a', 'l'), ('b', 'i'), ('b', 'm'), ('c', 'm'), ('c', 'n'), ('c', 'o'), ('c', 'p'), ('d', 'l'), ('d', 'p'), ('i', 'j'), ('i', 'm'), ('i', 'n'), ('j', 'k'), ('j', 'm'), ('j', 'n'), ('j', 'o'), ('k', 'l'), ('k', 'n'), ('k', 'o'), ('k', 'p'), ('l', 'o'), ('l', 'p'), ('m', 'n'), ('n', 'o'), ('o', 'p'))
		iy ← ('a', 'b', 'c', 'd')
		# construct tropical tensors
		xs ← ([ones(T,2) for i=1:length(x)]..., [ones(T, 2, 2) for j=1:28]...)
		for i=1:length(x)
			vertex_tensor(xs |> tget(i), 1, x[i])
		end
		for j=length(x)+1:length(x)+28
			bond_tensor(xs |> tget(j))
		end
	end
	# contract tropical tensors
	i_einsum!(ixs, xs, iy, out)
	# store the entry with specific boundary configuration to `optsize`
	optsize += out[config...].n
	~@routine
end

# ╔═╡ 546c8fb4-8bd2-11eb-30e4-5de043658636
compute_mis(0, ones(Tropical{Int},2,2,2,2), Tropical.(ones(Int,12)), [1,1,2,2])

# ╔═╡ c1aa24ce-8cea-11eb-3dd1-bd817acf2d8b
md"
We want to differentiate the weights (3rd argument) with respect to the loss (1st argument).
"

# ╔═╡ d34fc06e-8ce8-11eb-0604-7139a6e1fbc3
md" $(@bind ca CheckBox()) a $(@bind cb CheckBox()) b $(@bind cc CheckBox()) c $(@bind cd CheckBox()) d "

# ╔═╡ 85978e48-8bd9-11eb-340d-1f60d233522a
configs = NiLang.AD.gradient(compute_mis, (0, ones(Tropical{Int64},2,2,2,2), Tropical.(ones(Int,12)), 1 .+ [ca,cb,cc,cd]); iloss=1)[3];

# ╔═╡ 0a3f57c2-8be2-11eb-11f9-d1c6d78f0af0
vizconfig(nodes_simple, edges_simple, content.(configs))

# ╔═╡ 4180d4ee-8cec-11eb-139d-f9a0b79dcf3f
md"Here is a quick reference of the function definitions of Tropical algebra and Tropical einsum.

Note: For regular tensors, we can use existing backward rules defined in `ChainRules`.
"

# ╔═╡ 57e182e2-8bd4-11eb-2c36-835b43413c1c
vertex_tensor(ones(TropicalF64,2), 1, Tropical(1.0))[1]

# ╔═╡ 2dad711e-8cec-11eb-1b78-a79d8201a019
bond_tensor(ones(TropicalF64,2, 2))

# ╔═╡ abb3b8c0-8a94-11eb-35b9-fbfd31f19501
md"""
## 4. Towards differentiating `expmv`
"""

# ╔═╡ 4bd1668e-8cb5-11eb-2ea7-f5398916dd60
let
	F = 1.7
	W, w, h = 0.47, 0.22, 0.1/F
	lb = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"))
	lb2 = Compose.compose(context(), polygon([(-W, -h), (-W, h), (W, h), (W, -h)]), Compose.stroke("transparent"), fill("red"))
	lb3 = Compose.compose(context(), polygon([(-w/2, -h), (-w/2, h), (w/2, h), (w/2, -h)]), Compose.stroke("transparent"), fill("black"))
	lb4 = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"), fill("green"))
	tb = textstyle(:default, fontsize(3.5), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3), fill("white"), Compose.font("monospace"))
	eb = bondstyle(:default, linewidth(0.5mm))
	ar = bondstyle(:default, linewidth(0.3mm), Compose.arrow())
	xprog = (0.5, 0.15/F)
	xtensors = (0.25, 0.5/F)
	xscalars = (0.75, 0.5/F)
	xscalars2 = (0.25, 0.85/F)
	xscalars3 = (0.25, 1.2/F)
	xscalars4 = (0.5, 1.55/F)
	xpade = (0.6, 0.85/F)
	
	function box(loc, text; color="black")
		(color=="black" ? lb : (color=="red" ? lb2 : lb4)) >> loc
		tb_big >> (loc, text)
	end
	Compose.set_default_graphic_size(10cm, F*5cm)
	img = canvas() do
		box(xprog, "expmv")
		ar >> (xprog, xtensors .+ (w/2, -h-0.03/F))
		ar >> (xprog, xscalars .+ (-w/2, -h-0.03/F))
		ar >> (xtensors, xscalars2 .+ (0, -h-0.05/F))
		ar >> (xscalars2, xscalars3 .+ (0, -h-0.05/F))
		ar >> (xscalars3, xscalars3 .+ (0, 0.2/F))
		ar >> (xscalars, xscalars3 .+ (0.5, 0.2/F))
		ar >> (xtensors, xpade .+ (-w/2, -h-0.02/F))
		box(xtensors, "expm (small dense)")
		lb3 >> xpade
		tb_big >> (xpade, "Pade")
		box(xscalars, "Sparse mv (Lanczos)")
		box(xscalars2, "Linear solver")
		box(xscalars3, "QR decomp."; color="green")
		box(xscalars4, "+, -, *, /"; color="red")
	end
end

# ╔═╡ 5482dd8a-8cba-11eb-182e-6d7fed2543f5
md"""
### Resouces
* [Differentiating sparse operations](https://nextjournal.com/giggle/how-to-write-a-program-differentiably)
* [How to compute expm](http://eprints.ma.man.ac.uk/634/1/covered/MIMS_ep2006_394.pdf)
"""

# ╔═╡ b1b8ccb4-8ce5-11eb-36c7-f5be8f8d1115
md"## More interesting AD examples

* **Gate based quantum simulation**, Yao.jl: Extensible, Efficient Framework for Quantum Algorithm Design, Xiu-Zhe Luo, Jin-Guo Liu, Pan Zhang, Lei Wang [arxiv: 1912.10877](https://arxiv.org/abs/1912.10877)
* **Reverse time migration**, Reverse time migration with optimal checkpointing, William W. Symes [DOI](https://doi.org/10.1190/1.2742686)
* **Gaussian mixture models, Bundle adjustment and hand tracking**, A benchmark of selected algorithmic differentiation tools on some problems in computer vision and machine learning, Filip Srajer,Zuzana Kukelova &Andrew Fitzgibbon [DOI](https://doi.org/10.1080/10556788.2018.1435651)

### Videos

* [Transformations & AutoDiff | MIT Computational Thinking Spring 2021 | Lecture 3](https://www.youtube.com/watch?v=AAREeuaKCic&ab_channel=TheJuliaProgrammingLanguage) ``\comment{AD in image processing}``
"

# ╔═╡ f8a2feee-8d63-11eb-0e4a-874a49df59ed
md"""
## Quick Summary

* Every program is differentiable
* For packages implementing AD rules on tensors, they have problems handling effective codes.
"""

# ╔═╡ Cell order:
# ╠═b9a9214e-8830-11eb-1751-9d9161202c76
# ╟─d8669492-8c98-11eb-1cf5-374da9bd4556
# ╟─f0b4563a-8b0b-11eb-3085-458f5d9f88b8
# ╟─ce5af22a-8830-11eb-13c8-49c680526bd9
# ╟─22768818-8a95-11eb-1cfe-69534f8b0314
# ╟─68be48da-8a93-11eb-226b-7b1f2be99cb6
# ╟─da08c542-8a93-11eb-3375-d79ccd2de122
# ╟─0c80347a-8aff-11eb-0862-1d551e3af00b
# ╟─f55f9cba-8afe-11eb-3bf9-e3e5ecbf3a56
# ╠═bd3a4ece-8b09-11eb-2fcb-0710286e9892
# ╠═662c94c4-8b0a-11eb-39a2-9f37e689fbd3
# ╠═9e24dd6c-8b0a-11eb-020a-a1d7bf2e87bf
# ╟─fc682956-8b00-11eb-3102-952881130049
# ╟─167cbe40-8bf5-11eb-077b-bdd217c9bd78
# ╠═d29a56a4-8d86-11eb-1749-0972a594a0e4
# ╟─0ba6245c-8bf5-11eb-1005-1b72ffae3412
# ╟─aa316d52-8b00-11eb-1e7e-43e6f504977f
# ╟─8a55b3e4-8d67-11eb-203e-0b49c8e3e4aa
# ╟─f0468078-8b0c-11eb-1bdb-9f6f9496dcf2
# ╟─6335bc36-8b15-11eb-3731-5b76661d10fa
# ╟─2f60bb2c-8b2e-11eb-04f9-b79f81439af3
# ╟─61be9354-8b25-11eb-3205-db9b03a18f18
# ╟─30f3b718-8bfc-11eb-02cc-777ebc108429
# ╟─f67d1d28-8d6c-11eb-2e47-09e0aece0967
# ╟─0ebb6588-8b52-11eb-21c0-4fd9e70a77e7
# ╟─085145dc-8c10-11eb-12dc-3be1d68fe85c
# ╟─80caf3ba-8b1f-11eb-3030-5378078e2df9
# ╟─8bed03b2-8c1b-11eb-1043-23ff31b46991
# ╟─0790392e-8d04-11eb-3585-d17e29d379d9
# ╟─330ccb76-8d04-11eb-3bfd-434a93465975
# ╟─dfb179fe-8cae-11eb-24b6-3bdfa57a0d45
# ╟─f2c75336-8cab-11eb-09ce-e99acb20aee8
# ╟─d1760114-8b52-11eb-3241-252292bf96ac
# ╟─8cab36f8-8c1a-11eb-009a-3d6dbdb83d85
# ╟─01e46500-8ced-11eb-04b7-a1e05a81b1b2
# ╟─eb0cba98-8a54-11eb-132f-6320f3893da9
# ╟─43c6b8ec-8a79-11eb-2ff1-cb8fd958b693
# ╟─385638ea-8a81-11eb-166b-a9caf2b2792f
# ╟─99839afc-8a83-11eb-3f38-1bef2960f969
# ╠═0f22e7d6-88c5-11eb-0600-ff9b80f2113e
# ╠═a8a92160-88c5-11eb-0a7e-cbec45b627f0
# ╠═b1016e9c-88c5-11eb-12a5-41a914a794d3
# ╟─38a9382e-8b58-11eb-3e0b-656260c672cd
# ╟─1d503598-8d7c-11eb-1cf4-cf0755194043
# ╠═9a474d6a-885f-11eb-2db5-2145c0b9183f
# ╠═bf5702ae-88c5-11eb-2bb2-d7d5cf4eccdf
# ╠═95b2f940-8a4c-11eb-1813-7b28745b5050
# ╟─5ef95fa8-8a50-11eb-35e7-9354bf50b0e7
# ╟─8f2d529c-8a50-11eb-1910-29b61b1b8f82
# ╠═9b86ac04-8a4c-11eb-278e-5777f3552f25
# ╠═c697cf2c-8d76-11eb-1817-132326c4a92e
# ╠═8cae97aa-8d76-11eb-1740-0752e4297de2
# ╠═01e6cbb4-8d77-11eb-2519-23c89d5de971
# ╠═2b36cfd2-8d77-11eb-0b32-afc6294b5f50
# ╠═071bc508-8d77-11eb-01e6-950e171e6b45
# ╟─77562ce2-8a79-11eb-389a-a35e42e65171
# ╟─0f5cc712-8a84-11eb-0f31-91b30ad895af
# ╟─be1a31d8-8a91-11eb-2ead-fb4b7de74afc
# ╟─25bc662a-8c14-11eb-0689-45ac05c280aa
# ╟─3aa7ea18-8c15-11eb-010e-9d6a9e7dd952
# ╠═4eda1c4c-8b5f-11eb-2485-6b34cdef4ef8
# ╠═40ec8444-8b5f-11eb-0de5-97ed546aa4a1
# ╠═cf8235b6-8b5d-11eb-2997-41da626b503f
# ╠═82c60414-8c4a-11eb-27c2-e9294ecbebba
# ╠═3dcc9014-8c4b-11eb-0a15-a5e9e495c7b7
# ╟─eb77b59a-8c9b-11eb-2683-c150952cf85b
# ╠═5dfb2158-8c4a-11eb-33e6-f96facaf76fa
# ╠═63d9c6ce-8c4a-11eb-384d-a1fa4fe5ccd6
# ╟─8e9f00ca-8c15-11eb-3079-a147fd2b0149
# ╠═20e1c2ae-8b5f-11eb-334c-d74f310f9cc5
# ╠═88757052-8b5f-11eb-189c-5b74c7c9acde
# ╟─5b357e9e-8c15-11eb-1ce8-6b91fac05cc0
# ╟─bd15d5ca-8c18-11eb-33a8-7306b70c02e0
# ╠═2a7a0da4-8b63-11eb-2d1c-9b852a39dacf
# ╠═a0918e24-8b61-11eb-3f39-69bd8ce0c618
# ╠═f5e3b3de-8b61-11eb-2875-7594a13c1897
# ╠═eaa8f57e-8b61-11eb-28b7-bf487a201b80
# ╠═4543b0b2-8b64-11eb-0984-71f0a8027e6b
# ╠═c21f7c16-8b66-11eb-1502-8362d656896a
# ╠═270fef44-8c49-11eb-01af-43bd6e72da09
# ╠═4b667f7a-8b60-11eb-1773-6904f0858ad7
# ╟─a0a78f52-8b62-11eb-2e7a-4d3f975461ce
# ╠═c239e57a-8b62-11eb-1f10-87cda664a87c
# ╠═e64c8508-8b62-11eb-191c-45ec2ae6cd8c
# ╠═a910ae42-8b62-11eb-0474-7f81c709a77c
# ╟─a1e7dd02-8bc8-11eb-29e0-91df4126872a
# ╟─7919105e-8ce4-11eb-3f31-ff0fc17ca375
# ╟─1e954d20-8ce7-11eb-078e-0baaaa7132d4
# ╟─ee3c1284-8d7d-11eb-3f4c-59ceabbd180c
# ╟─32fdc4e0-8ce7-11eb-3acd-ed90c6e2e780
# ╟─8d25f9e0-8bcc-11eb-1df4-33556074d6ff
# ╟─083f13f0-8cea-11eb-0b3a-a96ab05aece9
# ╟─f96dc3b4-8bcf-11eb-15ec-232d15b3e075
# ╠═beee2d1e-8bcf-11eb-295c-a3b45bfc3aa8
# ╠═13bc3822-8bd0-11eb-1caa-1d49b3b74272
# ╠═06f15870-8bd0-11eb-3b4c-db7d9e3e7adf
# ╠═dec89520-8bcf-11eb-3ab4-97cc22ca2368
# ╠═546c8fb4-8bd2-11eb-30e4-5de043658636
# ╟─c1aa24ce-8cea-11eb-3dd1-bd817acf2d8b
# ╟─d34fc06e-8ce8-11eb-0604-7139a6e1fbc3
# ╠═85978e48-8bd9-11eb-340d-1f60d233522a
# ╠═0a3f57c2-8be2-11eb-11f9-d1c6d78f0af0
# ╟─4180d4ee-8cec-11eb-139d-f9a0b79dcf3f
# ╠═b01b19b6-8bc8-11eb-1a02-bf79f074bad3
# ╠═57e182e2-8bd4-11eb-2c36-835b43413c1c
# ╠═2dad711e-8cec-11eb-1b78-a79d8201a019
# ╟─abb3b8c0-8a94-11eb-35b9-fbfd31f19501
# ╟─4bd1668e-8cb5-11eb-2ea7-f5398916dd60
# ╟─5482dd8a-8cba-11eb-182e-6d7fed2543f5
# ╟─b1b8ccb4-8ce5-11eb-36c7-f5be8f8d1115
# ╟─f8a2feee-8d63-11eb-0e4a-874a49df59ed
