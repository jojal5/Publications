"""
    datalevel_loglike(Y::Vector{<:Real}, μ::Real, σ::Real, ξ::Real)

Compute the likelihood of the GEV parameters μ, σ and ξ as function of the data Y.

### Arguments
- `Y` : Vector of data.
- `μ` : GEV location parameter (real number).
- `σ` : GEV scale parameter (positive real number).
- `ξ` : GEV shape parameter (real number).

### Details

The function uses the Distributions.jl package.

### Examples

\```
 julia> datalevel_loglike(Y,0,1,.1)
\```

"""
function datalevel_loglike(Y::Vector{<:Real},μ::Real,σ::Real,ξ::Real)

    pd = GeneralizedExtremeValue(μ,σ,ξ)
    logL = loglikelihood(pd,Y)

    return logL

end

"""
    iGMRFupdate(F::GMRF.iGMRF,κ::Real)

Return an iGMRF object similar to F but with the precision parameter equals to κ.

### Arguments
- `F` : An iGMRF object. See the GMRF.jl package for more details.
- `κ` : The new iGMRF precision parameter. It should be a positive real number.

### Details

The function uses the GMRF.jl package.

### Examples

\```
 julia> F = iGMRFupdate(F,10)
\```

"""
function iGMRFupdate(F::GMRF.iGMRF,κ::Real)

    @assert κ>0

    F̃ = GMRF.iGMRF(F.G, F.rankDeficiency, κ, F.W, F.W̄)

    return F̃

end


"""
    latentgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, S::Dict, S̄::Dict, x̃::Vector{<:Real}, logL::Vector{<:Real})

Update the current latent iGMRF field on all vertices using the proposed candidates at vertices with data.

### Arguments
- `x`: The vector of the current state of the iGMRF at all vertices.
- `F`: The iGMRF object representing the current properties of the field.
- `S` : A dictionnary containing the vertices corresponding to the grid cells containing observations.
- `S̄` : A dictionnary containing the vertices corresponding to the grid cells containing no observations.
- `x̃` : The proposed next states for the vertices S.
- `logL` : The log-likelihood difference between the proposed and the current states at every vertice in S.

### Details

The function parallelizes the update using the conditional independence assumption.
The function directly modifies the input argument x.

### Examples

\```
 julia> latentgmrf_update!(x, F, S, S̄, x̃, logL)
\```

"""
function latentgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, S::Dict, S̄::Dict, x̃::Vector{<:Real}, logL::Vector{<:Real})


    for j in eachindex(S[:CondIndSubset])

        V = S[:CondIndSubset][j]

        ind = S[:CondIndIndex][j]

        latentcondgmrf_update!(x, F, V, x̃[ind], logL[ind])

    end

    for j in eachindex(S̄[:CondIndSubset])

        V = S̄[:CondIndSubset][j]

        latentcondgmrf_update!(x, F, V)

    end
end

"""
    latentcondgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, V::Vector{<:Real})
    latentcondgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, V::Vector{<:Real}, x̃::Vector{<:Real}, logL::Vector{<:Real})

Update the current latent iGMRF field on the vertices V. It is assumed that the vertices V are
conditionally indenpendent, so they can be updated in parallel. If no candidates x̃ and no likelihood term logL
are provided for the vertices V, it is assumed that they do not possess a likelihood term. In other words, the
corresponding grid cells contain no observations. Otherwise, candiates and the likelihood difference should be provided.


### Arguments
- `x`: The vector of the current state of the iGMRF at all vertices.
- `F`: The iGMRF object representing the current properties of the field.
- `V`: The vertices to update.
- `x̃`: Proposed candidates for the vertices V
- `logL` : Likelihood difference between the candidates and the current states at the vertices V.

### Details

The function parallelizes the update using the conditional independence assumption.
The function directly modifies the input argument x.

### Examples

\```
 julia> latentcondgmrf_update!(x, F, V)
 julia> latentcondgmrf_update!(x, F, V, x̃, logL)
\```

"""
function latentcondgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, V::Vector{<:Real})

    pd = GMRF.fullconditionals(F,x)[V]

    x̃ = rand.(pd)

    setindex!(x,x̃,V)

end

function latentcondgmrf_update!(x::Vector{<:Real}, F::GMRF.iGMRF, V::Vector{<:Real}, x̃::Vector{<:Real}, logL::Vector{<:Real})

    u = rand(length(V))

    pd = GMRF.fullconditionals(F,x)[V]

    lf = logpdf.(pd,x̃) - logpdf.(pd,x[V])

    lr = logL + lf

    ind = lr .> log.(u)

    setindex!(x,x̃[ind],V[ind])

end

"""
    latentfieldprecision_sample(F::GMRF.iGMRF, x::Vector{<:Real})

Sample the precision of the latent iGMRF using the complete conditional distribution.

### Arguments
- `F` : An iGMRF object. See the GMRF.jl package for more details.
- `x` : The current state of the field at every vertices.

### Details

The function uses the GMRF.jl package.

### Examples

\```
 julia> κ = latentfieldprecision_sample(F, x)
\```

"""
function latentfieldprecision_sample(F::GMRF.iGMRF, x::Vector{<:Real})

    m = prod(F.G.gridSize)
    k = F.rankDeficiency
    W = F.G.W

    b = dot(x, W*x)

    # Informative prior Gamma(1,1/100)
    pd = Gamma( (m - k)/2 + 1 , 1/(b/2 + 1/100) )

    # Improper prior
    # pd = Gamma( (m - k)/2 , 2/b)

    κ = rand(pd)

    return κ

end

"""
    regressioncoefficient_sample(F::GMRF.iGMRF, Covariate::Vector{<:Real}, U::Vector{<:Real}, β::Real)

Sample the regression coefficients of the latent iGMRF mean using the complete conditional distribution.

### Arguments
- `F`: An iGMRF object. See the GMRF.jl package for more details.
- `Covariate`: The covariate values at every vertices.
- `x`: The current state of the iGMRF field.

### Details

The function uses the GMRF.jl package.

### Examples

\```
 julia> κ = latentfieldprecision_sample(F, x)
\```

"""
# function regressioncoefficient_sample(F::GMRF.iGMRF, Covariate::Vector{<:Real}, U::Vector{<:Real}, β::Real)
function regressioncoefficient_sample(F::GMRF.iGMRF, Covariate::Vector{<:Real}, x::Vector{<:Real})

    κ = F.κ
    W = F.G.W

#     x = Covariate*β + U

    h = κ*Covariate'*(W*x)

    pd = NormalCanon(h,κ*Qₓ)

    β̃ = rand(pd)

    return β̃

end

function accrate(θ::Array{<:Real})

    d = θ[:,2:end] - θ[:,1:end-1]
    rate = 1 .- mean(d .≈ 0)

    return rate

end

function gmrfsample(b::Vector{<:Real},Q::AbstractArray{<:Real})

    F = cholesky(Q)

    μ = F\b

    z = randn(length(b))

    # Pivoting is on by default in SuiteSparse (https://julialinearalgebra.github.io/SuiteSparse.jl/latest/cholmod.html)
    v = F.UP\z

    y = μ + v

    return y

end

function slicematrix(A::AbstractMatrix{T}) where T
    n, m = size(A)
    B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
    for i in 1:m
        B[i] .= A[:, i]
    end
    return B
end

"""
Load an IDF CSV file
"""
function idf_load(stationID::AbstractString)

    # path = "/Users/jalbert/Dropbox (MAGI)/Files/Data/ECCC_idf_2019/CSV/"
    path = "/Users/jalbert/OneDrive - polymtl.ca/Files/Data/ECCC_idf_2019/CSV/"
    filename = path*stationID*".csv"

    df = CSV.read(filename, DataFrame)
    rename!(df,:Année => :Year)

    df2 = stack(df, Not(:Year); variable_name=:Duration, value_name=:Pcp)
    dropmissing!(df2,:Pcp)

    return df2

end

"""
Nearest neighbor search
"""
function nnsearch(X::Matrix{<:Real}, points::Matrix{<:Real})

    nPoints = size(points,2)
    ind = zeros(Int64,nPoints)

    for i=1:nPoints
        ind[i] = nnsearch(X,points[:,i])
    end

    return ind

end

function nnsearch(X::Matrix{<:Real}, point::Vector{<:Real})

    d = X .- point
    d² = dropdims(sum(d.^2,dims=1),dims=1)

    # Find the index of the minimum
    ind = argmin(d²)

    return ind

end

function initialize_β(y::Vector{<:Real},X::Array{<:Real})


    # removing the offset
    X̃ = X .- mean(X, dims=1)
    ỹ = y .- mean(y)

    β = X̃\ỹ

    return β

end

function initialize_U(Uᵢ::Vector{<:Real},W::SparseMatrixCSC{Int64,Int64},κ,S,S̄)

    m = size(W,1)

    U = Array{Float64}(undef, m)
    U[ S[:V] ] = Uᵢ

    Waa = W[S̄[:V],S̄[:V]]
    Wab = W[S̄[:V],S[:V]]

    b = -κ*Wab*Uᵢ
    Q = κ*Waa
    U[ S̄[:V] ] = gmrfsample(b,Q)

    return U

end

function initialize_rf(data::Dict)

    Y = data[:Y]
    X₁ᵢ = data[:X₁ᵢ]
    X₂ᵢ = data[:X₂ᵢ]
    G = data[:G]
    S = data[:S]
    S̄ = data[:S̄]

    # Initial values
    κ₁ = 100
    κ₂ = 80

    n = length(Y)

    μ₀ = Array{Float64}(undef, n)
    ϕ₀ = Array{Float64}(undef, n)
    ξ₀ = 0.0

    for i=1:n
        fd = Extremes.gumbelfitpwm(Y[i])
        μ₀[i] = fd.θ̂[1]
        ϕ₀[i] = fd.θ̂[2]
    end

    β₁ = initialize_β(log.(μ₀),X₁ᵢ)
    Uᵢ = log.(μ₀) - X₁ᵢ*β₁
    U = initialize_U(Uᵢ,G.W,κ₁,S,S̄)

    β₂ = initialize_β(ϕ₀,X₂ᵢ)
    Vᵢ = ϕ₀ - X₂ᵢ*β₂
    V = initialize_U(Vᵢ,G.W,κ₂,S,S̄)

    ini = Dict(:μ₀ => μ₀, :σ₀ => exp.(ϕ₀), :ξ₀ => ξ₀,
        :U => U, :β₁ => β₁, :κ₁ => κ₁,
        :V => V, :β₂ => β₂, :κ₂ => κ₂)

    return ini

end



# function initialize_rf(Y::Array{Array{Float64,1},1},S::Dict,S̄::Dict,X₁ᵢ::Vector{<:Real},X₂ᵢ::Vector{<:Real})
#
#     κ₁ = 100
#     κ₂ = 80
#
#     n = length(Y)
#
#     μ₀ = Array{Float64}(undef, n)
#     σ₀ = Array{Float64}(undef, n)
#     ξ₀ = 0.0
#
#     for i=1:n
#         fd = Extremes.gumbelfitpwmom(Y[i])
#         μ₀[i] = location(fd)
#         σ₀[i] = scale(fd)
#     end
#
#     ϕ₀ = log.(σ₀)
#
#     β₁ = initialize_β(log.(μ₀),X₁ᵢ)
#     Uᵢ = log.(μ₀) - X₁ᵢ*β₁
#     U = initialize_U(Uᵢ,G.W,κ₁,S,S̄)
#
#     β₂ = initialize_β(ϕ₀,X₂ᵢ)
#     Vᵢ = ϕ₀ - X₂ᵢ*β₂
#     V = initialize_U(Vᵢ,G.W,κ₂,S,S̄)
#
#     ini = Dict(:μ₀ => μ₀, :σ₀ => σ₀, :ξ₀ => ξ₀,
#         :U => U, :β₁ => β₁, :κ₁ => κ₁,
#         :V => V, :β₂ => β₂, :κ₂ => κ₂)
#
#     return ini
#
# end

function mcmc(datastructure::Dict, niter::Real, initialvalues::Dict, stepsize::Dict)


    Y = datastructure[:Y]
    X₁ᵢ = datastructure[:X₁ᵢ]
    X₂ᵢ = datastructure[:X₂ᵢ]
    G = datastructure[:G]
    S = datastructure[:S]
    S̄ = datastructure[:S̄]

    u = Array{Float64}(undef, n, niter)
    β₁ = Array{Float64}(undef, niter)
    κ₁ = Array{Float64}(undef, niter)

    v = Array{Float64}(undef, n, niter)
    β₂ = Array{Float64}(undef, niter)
    κ₂ = Array{Float64}(undef, niter)

    ξ = Array{Float64}(undef, niter)
    ξ[1] = initialvalues[:ξ₀]

    μ₀ = initialvalues[:μ₀]
    σ₀ = initialvalues[:σ₀]

    κ₁[1] = initialvalues[:κ₁]
    κ₂[1] = initialvalues[:κ₂]
    β₁[1] = initialvalues[:β₁]
    β₂[1] = initialvalues[:β₂]

    F₁ = GMRF.iGMRF(G,1,κ₁[1])
    F₂ = GMRF.iGMRF(G,1,κ₂[1])

    U = initialvalues[:U]
    V = initialvalues[:V]

    u[:,1] = U[S[:V]]
    v[:,1] = V[S[:V]]

    δ = randn(n)


    @showprogress for j=2:niter

        # Generate a candidate for {Uᵢ : i ∈ S}
        Ũ = u[:,j-1] + stepsize[:u]*randn!(δ)

        # Computing the corresponding candidates for μ at the grid cells containing the observations
        μ̃ = exp.(X₁ᵢ*β₁[j-1] + Ũ)

        # Evaluate the log-likelihood ratio at the data level between the candidates and the present state
        logL = datalevel_loglike.(Y, μ̃, σ₀, ξ[j-1]) - datalevel_loglike.(Y, μ₀, σ₀, ξ[j-1])

        # Updating the latent field U
        latentgmrf_update!(U, F₁, S, S̄, Ũ, logL)

        u[:,j] = U[S[:V]]

        x = X₁ᵢ*β₁[j-1]+u[:,j]
        μ₀ = exp.(x)
        β₁[j] = X₁ᵢ \ ( x .- mean(U[S[:V]]) )  # FONCTIONNE

        # Sampling the new precision parameter
        κ₁[j] = latentfieldprecision_sample(F₁, U)

        # Updating the iGMRF object with the new precision
        F₁ = GMRF.iGMRF(G,1,κ₁[j])



        # Generate a candidate for {Vᵢ : i ∈ S}
        Ṽ = v[:,j-1] + stepsize[:v]*randn!(δ)

        # Computing the corresponding candidates for σ at the grid cells containing the observations
        σ̃ = exp.(X₂ᵢ*β₂[j-1] + Ṽ)

        # Evaluate the log-likelihood ratio at the data level between the candidates and the present state
        logL = datalevel_loglike.(Y, μ₀, σ̃, ξ[j-1]) - datalevel_loglike.(Y, μ₀, σ₀, ξ[j-1])

        # Updating the latent field V
        latentgmrf_update!(V, F₂, S, S̄, Ṽ, logL)

        v[:,j] = V[S[:V]]

        x = X₂ᵢ*β₂[j-1]+v[:,j]
        σ₀ = exp.(x)
        β₂[j] = X₂ᵢ \ ( x .- mean(V[S[:V]]) )  # FONCTIONNE

        # Sampling the new precision parameter
        κ₂[j] = latentfieldprecision_sample(F₂, V)

        # Updating the iGMRF object with the new precision
        F₂ = GMRF.iGMRF(G,1,κ₂[j])

        ξ̃ = ξ[j-1] + stepsize[:ξ]*randn()
        logL = sum(datalevel_loglike.(Y, μ₀, σ₀, ξ̃) - datalevel_loglike.(Y, μ₀, σ₀, ξ[j-1]))
        if logL > log(rand())
            ξ[j] = ξ̃
        else
            ξ[j] = ξ[j-1]
        end

    end

    parmnames = vcat(["u[$i]" for i=1:n], ["v[$i]" for i=1:n], "κ₁", "β₁", "κ₂", "β₂","ξ")
    res = vcat(u, v, κ₁', β₁', κ₂', β₂', ξ')
    C = Mamba.Chains(collect(res'), names=parmnames)

    return C

end

function interpolation(C::Chains, datastructure::Dict)

    S = datastructure[:S]
    S̄ = datastructure[:S̄]
    G = datastructure[:G]

    n = length(datastructure[:Y])

    u = dropdims(C[:,["u[$i]" for i=1:n],1].value, dims=3)
    κ₁ = vec(C[:,"κ₁",1].value)

    v = dropdims(C[:,["v[$i]" for i=1:n],1].value, dims=3)
    κ₂ = vec(C[:,"κ₂",1].value)

    β₁ = vec(C[:,"β₁",1].value)
    β₂ = vec(C[:,"β₂",1].value)
    ξ = vec(C[:,"ξ",1].value)

    niter = length(κ₁)


    Waa = G.W[S̄[:V],S̄[:V]]
    Wab = G.W[S̄[:V],S[:V]]

    # number of grid cells
    m = prod(G.gridSize)

    U = Array{Float64}(undef, m, niter)
    V = Array{Float64}(undef, m, niter)


    for j=1:niter

        U[S[:V],j] = u[j,:]
        V[S[:V],j] = v[j,:]

        b = -κ₁[j]*Wab*U[ S[:V], j ]
        Q = κ₁[j]*Waa
        U[ S̄[:V] , j ] = gmrfsample(b,Q)

        b = -κ₂[j]*Wab*V[ S[:V], j]
        Q = κ₂[j]*Waa
        V[ S̄[:V] , j] = gmrfsample(b,Q)

    end

    parmnames = vcat(["u[$i]" for i=1:m], ["v[$i]" for i=1:m], "κ₁", "β₁", "κ₂", "β₂","ξ")
    res = vcat(U, V, κ₁', β₁', κ₂', β₂', ξ')
    completeChain = Mamba.Chains(collect(res'), names=parmnames)

    return completeChain

end


function getGEV(C::Chains, datastructure::Dict, X₁::Array{<:Real}, X₂::Array{<:Real})

    G = datastructure[:G]

    #  number of grid cells
    m = prod(G.gridSize)

    U = dropdims(C[:,["u[$i]" for i=1:m],1].value, dims=3)
    κ₁ = vec(C[:,"κ₁",1].value)

    V = dropdims(C[:,["v[$i]" for i=1:m],1].value, dims=3)
    κ₂ = vec(C[:,"κ₂",1].value)

    β₁ = dropdims(C[:,"β₁",1].value, dims=3)
    β₂ = dropdims(C[:,"β₂",1].value, dims=3)
    ξ = vec(C[:,"ξ",1].value)

    μ = exp.( β₁*X₁' + U )
    σ = exp.( β₂*X₂' + V )

#     pd = GeneralizedExtremeValue.(μ, σ, ξ)

    parmnames = vcat(["μ[$i]" for i=1:m], ["σ[$i]" for i=1:m], "ξ")
    res = vcat(μ', σ', ξ')
    gevChain = Mamba.Chains(collect(res'), names=parmnames)

    return gevChain

end

function cvmcriterion(pd::UnivariateDistribution,x̃::Array{<:Real,1})

    # Computes the Cramér-Von Mises criterion for one sample.
    # Assumes that there is no duplicate in x

    x = sort(x̃)
    n = length(x)

    T = 1/12/n + sum( ((2*i-1)/2/n - cdf(pd,x[i]) )^2 for i=1:n)

    ω² = T/n

    return ω²

end


function cvmcriterion(x̃::Array{<:Real,1},ỹ::Array{<:Real,1})

    # Computes the Cramér-Von Mises criterion for two samples.
    # Assumes that there is no duplicate in x and y

    x = sort(x̃)
    y = sort!(ỹ)
    n = length(x)
    m = length(y)

    ind = [fill(1,n); fill(2,m)]

    perm = sortperm([x;y])
    ind = ind[perm]

    r = findall(ind .== 1)
    s = findall(ind .== 2)

    U = n*sum( (r[i]-i)^2 for i=1:n ) + m*sum( (s[j]-j)^2 for j=1:m )

    ω² = U/n^2/m^2 - (4*m*n-1)/6/m/n

    return ω²

end
