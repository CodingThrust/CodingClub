### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2ec32c9d-0216-48a8-848d-79f7cc84798e
using DelimitedFiles, Test, BenchmarkTools, Statistics

# ╔═╡ 701dfc92-4891-477b-9e15-8083f8ca5531
begin
	using PlutoUI

	struct BondWrapper
		content
	end
	
	function Base.show(io::IO, mime::MIME"text/html", b::BondWrapper)
		print(io,
	"""<span style="color:#15AA77; vertical-align: top"><strong><code>$(b.content.defines) = </code></strong></span>""")
		Base.show(io, mime, b.content)
	end
	
	macro xbind(args...)
		esc(quote
			let
				bond = @bind $(args...)
				$BondWrapper(bond)
			end
		end)
	end
end;

# ╔═╡ 0b0e674d-688d-491d-8a51-36334ad40b1a
using Random

# ╔═╡ 8c702004-0534-43cb-b92c-1b494a1145b3
using Profile

# ╔═╡ c96ce06a-bd6a-419c-8c77-cc76ea0f8593
# pip install viznet, matplotlib
using PythonCall, CondaPkg

# ╔═╡ 0c3000cf-46d3-4481-9d6d-61980443dd93
"""General Annealing Problem"""
abstract type AnnealingProblem end

# ╔═╡ 9ec34d03-e9e6-460c-bc8b-39b70b6b8a37
"""
    SpinAnnealingProblem{T<:Real} <: AnnealingProblem

Annealing problem defined by coupling matrix of spins.
"""
struct SpinAnnealingProblem{T<:Real} <: AnnealingProblem  # immutable, with type parameter T (a subtype of Real).
    num_spin::Int
    coupling::Matrix{T}
    function SpinAnnealingProblem(coupling::Matrix{T}) where T
        size(coupling, 1) == size(coupling, 2) || throw(DimensionMismatch("input must be square matrix."))
        new{T}(size(coupling, 1), coupling)
    end
end

# ╔═╡ a7e70b41-9184-4127-b66f-dfc03b8b3004
"""
    load_coupling(filename::String) -> SpinAnnealingProblem

Load the data file into symmtric coupling matrix.
"""
function load_coupling(filename::String)
    data = readdlm(filename)
    is = Int.(view(data, :, 1)) .+ 1  #! @. means broadcast for the following functions, is here used correctly?
    js = Int.(view(data, :, 2)) .+ 1
    weights = data[:,3]
    num_spin = max(maximum(is), maximum(js))
    J = zeros(eltype(weights), num_spin, num_spin)
    @inbounds for (i, j, weight) = zip(is, js, weights)
        J[i,j] = weight/2
        J[j,i] = weight/2
    end
    SpinAnnealingProblem(J)
end

# ╔═╡ 35b7dcfe-717f-48b0-bd4f-4c72f880a229
@testset "loading" begin
    sap = load_coupling("programs/example.txt")
    @test size(sap.coupling) == (300, 300)
end

# ╔═╡ 6cc6e9a2-4409-4ad1-b8d1-be0826fe9592
abstract type AnnealingConfig end

# ╔═╡ a2a174b8-28d0-461f-97c6-8364ccada6c4
struct SpinConfig{Ts, Tf} <: AnnealingConfig
    config::Vector{Ts}
    field::Vector{Tf}
end

# ╔═╡ f1c1eae7-102f-4ca3-9fd2-d81529ada0a3
"""
    random_config(prblm::AnnealingProblem) -> SpinConfig

Random spin configuration.
"""
function random_config end   # where to put the docstring of a multiple-dispatch function is a problem. Using `abstract function` is proper.

# ╔═╡ caa805be-6c98-4b59-8404-c0db284c58e9
function random_config(prblm::SpinAnnealingProblem)
    config = rand([-1,1], prblm.num_spin)
    SpinConfig(config, prblm.coupling*config)
end

# ╔═╡ e692876e-72eb-4a19-a0b9-367ab3e84783
@testset "random config" begin
    sap = load_coupling("programs/example.txt")
    initial_config = random_config(sap)
    @test initial_config.config |> length == 300
    @test eltype(initial_config.config) == Int
end

# ╔═╡ 0d37406b-f16a-4b91-a138-ee59a92e2ca4
md"## Main program"

# ╔═╡ 6116801e-e6a8-4e75-9e04-6c07e4a17305
"""
    get_cost(config::AnnealingConfig, ap::AnnealingProblem) -> Real

Get the cost of specific configuration.
"""
get_cost(config::SpinConfig, sap::SpinAnnealingProblem) = sum(config.config'*sap.coupling*config.config)

# ╔═╡ 92c83b32-668a-4adb-8531-4258d094403c
"""
    propose(config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(config::SpinConfig, ::SpinAnnealingProblem)  # ommit the name of argument, since not used.
    ispin = rand(1:length(config.config))
    @inbounds ΔE = -config.field[ispin] * config.config[ispin] * 4 # 2 for spin change, 2 for mutual energy.
    ispin, ΔE
end

# ╔═╡ 83211927-e0b7-4439-90e1-9d348328be16
"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
@inline function flip!(config::SpinConfig, ispin::Int, sap::SpinAnnealingProblem)
    @inbounds config.config[ispin] = -config.config[ispin]  # @inbounds can remove boundary check, and improve performance
    @simd for i=1:sap.num_spin
        @inbounds config.field[i] += 2 * config.config[ispin] * sap.coupling[i,ispin]
    end
    config
end

# ╔═╡ 30ea57fc-f6ce-4fdf-b880-fc83012ed49f
"""
    anneal_singlerun!(config::AnnealingConfig, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * configuration that can be updated.
    * prblm: problem with `get_cost`, `flip!` and `random_config` interfaces.
    * tempscales: temperature scales, which should be a decreasing array.
    * num_update_each_temp: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)
    cost = get_cost(config, prblm)
    
    opt_config = config
    opt_cost = cost
    for beta = 1 ./ tempscales
        @simd for m = 1:num_update_each_temp  # single instriuction multiple data, see julia performance tips.
            proposal, ΔE = propose(config, prblm)
            if exp(-beta*ΔE) > rand()  #accept
                flip!(config, proposal, prblm)
                cost += ΔE
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = config
                end
            end
        end
    end
    opt_cost, opt_config
end

# ╔═╡ 01275de9-c88b-4404-94f9-8a9129b06e70
"""
    anneal(nrun::Int, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing with multiple runs.
"""
function anneal(nrun::Int, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)
    local opt_config, opt_cost
    for r = 1:nrun
        initial_config = random_config(prblm)
        cost, config = anneal_singlerun!(initial_config, prblm, tempscales, num_update_each_temp)
        if r == 1 || cost < opt_cost
            opt_cost = cost
            opt_config = config
        end
    end
    opt_cost, opt_config
end

# ╔═╡ dcac0777-ff00-4ccb-91bb-678680413826
Random.seed!(2)

# ╔═╡ 53510c2f-ed40-4dee-84c0-436cc8281076
tempscales = 10 .- (1:64 .- 1) .* 0.15 |> collect

# ╔═╡ a4269932-e5af-4966-a0a9-4757066048b8
sap = load_coupling("programs/example.txt")

# ╔═╡ 61b6649e-04cd-45ab-ab40-19283a556bd8
@testset "anneal" begin
    opt_cost, opt_config = anneal(30, sap, tempscales, 4000)
    @test anneal(30, sap, tempscales, 4000)[1] == -3858
    anneal(30, sap, tempscales, 4000)
    res = median(@benchmark anneal(30, $sap, $tempscales, 4000))
    @test res.time/1e9 < 2
    @test res.allocs < 500
end

# ╔═╡ a883bf1a-f38c-4759-9287-7bad55d9ac37
@xbind run_julia_benchmark CheckBox()

# ╔═╡ 0aebe7f7-3a88-4bf3-ab81-2d817f2d56b8
if run_julia_benchmark @benchmark anneal(30, $sap, $tempscales, 4000) end

# ╔═╡ 6a658549-c1e5-4ce2-bcce-ca08a5b0fe2c
with_terminal() do
	Profile.clear()
	@profile anneal(100, sap, tempscales, 4000)
	Profile.print()
end

# ╔═╡ 8fa64302-22ac-49b1-ae0d-035660eacdac
md"""
## Calling a Fortran program
* https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/index.html
* https://craftofcoding.wordpress.com/2017/02/26/calling-fortran-from-julia-i/
* https://craftofcoding.wordpress.com/2017/03/01/calling-fortran-from-julia-ii/
"""

# ╔═╡ 8dac622d-b115-46e5-8dd8-2150fcf53cc5
cd(joinpath(@__DIR__, "programs")) do
	run(`gfortran -shared -fPIC problem.f90 fsa.f90 -o fsa.so` & `nm fsa.so`)
end

# ╔═╡ e9ffb48e-8d57-468e-b47c-466d827faa78
@xbind run_fortran_benchmark CheckBox()

# ╔═╡ 9e51d1eb-8933-4e92-8c42-4d5c229b1231
if run_fortran_benchmark @benchmark ccall($((:test_, joinpath(@__DIR__, "programs/fsa.so"))), Int32, ()) end

# ╔═╡ 1acd9c9b-6a01-44ab-9cbe-9d0c94544f39
md"""
## What about Python?
We can use [PyCall](https://github.com/JuliaPy/PyCall.jl) to call python programs!

### **Challenge!**
1. use Python package [viznet](https://github.com/GiggleLiu/viznet) and [matplotlib](https://matplotlib.org/) for visualization
2. benchmark pure python version of simulated annealing, show the time
"""

# ╔═╡ 92e7af60-0566-4769-b90c-e0abbb2b2f76
# ╠═╡ show_logs = false
CondaPkg.add("seaborn")

# ╔═╡ c9d2f737-c034-4394-b18b-70e9f9ec28ea
plt = pyimport("matplotlib.pyplot")

# ╔═╡ 83d904d3-cdfe-499b-bfaa-c2ae643526a7
let
	N = 400
	t = LinRange(0, 2π, N)
	r = 0.5 .+ cos.(t)
	x, y = r .* cos.(t), r .* sin.(t)
	
	fig, ax = plt.subplots()
	ax.plot(x, y, "k")
	ax.set(aspect=1)
	plt.show()
end;

# ╔═╡ 0a1ad6b8-f848-4147-a4aa-b4edc0caa42b
pysa = try
	pyimport("testsa")
catch e
	pyimport("sys").path.append(joinpath(@__DIR__, "programs"))  # add current folder into path
	pyimport("testsa")
end

# ╔═╡ 01dd7234-f600-4597-b372-049e000ced1b
@xbind benchmark_python CheckBox()

# ╔═╡ a3575bea-e521-47e1-9167-62e999728786
if benchmark_python @benchmark pysa.test_codec() end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CondaPkg = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BenchmarkTools = "~1.3.2"
CondaPkg = "~0.2.18"
PlutoUI = "~0.7.50"
PythonCall = "~0.9.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "406566cb8f74fec80745bdcf25484266d91abb06"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "TOML"]
git-tree-sha1 = "741146cf2ced5859faae76a84b541aa9af1a78bb"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.18"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "SnoopPrecompile", "StructTypes", "UUIDs"]
git-tree-sha1 = "84b10656a41ef564c39d2d477d7236966d2b5683"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.12.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "a6a4771aba1dc8942bc0f44ff9f8ee0f893ef888"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.12"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "f27dabb05ec811675a9eefe49325a14ae7266b0b"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.12"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "087555b0405ed6adf526cef22b6931606b5af8ac"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.4.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═2ec32c9d-0216-48a8-848d-79f7cc84798e
# ╟─701dfc92-4891-477b-9e15-8083f8ca5531
# ╠═0c3000cf-46d3-4481-9d6d-61980443dd93
# ╠═9ec34d03-e9e6-460c-bc8b-39b70b6b8a37
# ╠═a7e70b41-9184-4127-b66f-dfc03b8b3004
# ╠═35b7dcfe-717f-48b0-bd4f-4c72f880a229
# ╠═6cc6e9a2-4409-4ad1-b8d1-be0826fe9592
# ╠═a2a174b8-28d0-461f-97c6-8364ccada6c4
# ╠═f1c1eae7-102f-4ca3-9fd2-d81529ada0a3
# ╠═caa805be-6c98-4b59-8404-c0db284c58e9
# ╠═e692876e-72eb-4a19-a0b9-367ab3e84783
# ╟─0d37406b-f16a-4b91-a138-ee59a92e2ca4
# ╠═30ea57fc-f6ce-4fdf-b880-fc83012ed49f
# ╠═01275de9-c88b-4404-94f9-8a9129b06e70
# ╠═6116801e-e6a8-4e75-9e04-6c07e4a17305
# ╠═92c83b32-668a-4adb-8531-4258d094403c
# ╠═83211927-e0b7-4439-90e1-9d348328be16
# ╠═0b0e674d-688d-491d-8a51-36334ad40b1a
# ╠═dcac0777-ff00-4ccb-91bb-678680413826
# ╠═53510c2f-ed40-4dee-84c0-436cc8281076
# ╠═a4269932-e5af-4966-a0a9-4757066048b8
# ╠═61b6649e-04cd-45ab-ab40-19283a556bd8
# ╟─a883bf1a-f38c-4759-9287-7bad55d9ac37
# ╠═0aebe7f7-3a88-4bf3-ab81-2d817f2d56b8
# ╠═8c702004-0534-43cb-b92c-1b494a1145b3
# ╠═6a658549-c1e5-4ce2-bcce-ca08a5b0fe2c
# ╟─8fa64302-22ac-49b1-ae0d-035660eacdac
# ╠═8dac622d-b115-46e5-8dd8-2150fcf53cc5
# ╟─e9ffb48e-8d57-468e-b47c-466d827faa78
# ╠═9e51d1eb-8933-4e92-8c42-4d5c229b1231
# ╟─1acd9c9b-6a01-44ab-9cbe-9d0c94544f39
# ╠═c96ce06a-bd6a-419c-8c77-cc76ea0f8593
# ╠═92e7af60-0566-4769-b90c-e0abbb2b2f76
# ╠═c9d2f737-c034-4394-b18b-70e9f9ec28ea
# ╠═83d904d3-cdfe-499b-bfaa-c2ae643526a7
# ╠═0a1ad6b8-f848-4147-a4aa-b4edc0caa42b
# ╟─01dd7234-f600-4597-b372-049e000ced1b
# ╠═a3575bea-e521-47e1-9167-62e999728786
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
