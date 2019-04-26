import Base.rand
import Distributions.pdf, Distributions.logpdf, Distributions.cdf, Distributions.quantile
import Distributions.params, Distributions.mean, Distributions.loglikelihood

struct NonStandardBeta{T<:Real} <: ContinuousUnivariateDistribution
        a::T
        b::T
        α::T
        β::T
        NonStandardBeta(a,b,α,β) = new{Float64}(Float64(a),Float64(b),Float64(α),Float64(β))
end

params(d::NonStandardBeta) = (d.a, d.b, d.α, d.β)

function pdf(d::NonStandardBeta,y::Real)

    a, b, α, β = params(d)

    ỹ = (y-a)/(b-a)

    density = pdf(Beta(α,β),ỹ)/(b-a)

    return density

end

function logpdf(d::NonStandardBeta,y::Real)

    a, b, α, β = params(d)

    ỹ = (y-a)/(b-a)

    ldensity = logpdf(Beta(α,β),ỹ) - log(b-a)

    return ldensity

end

function loglikelihood(d::NonStandardBeta,y::Array{<:Real})

    a, b, α, β = params(d)

    n  = length(y)

    ỹ = (y .- a)/(b-a)

    loglike = loglikelihood(Beta(α,β),ỹ) .- n*log(b-a)

    return loglike

end

function cdf(d::NonStandardBeta,y::Real)

    a, b, α, β = params(d)

    ỹ = (y-a)/(b-a)

    F = cdf(Beta(α,β),ỹ)

    return F

end

function quantile(d::NonStandardBeta,p::Real)

    a, b, α, β = params(d)

    y = quantile(Beta(α,β),p)

    ỹ = a + (b-a)*y

    return ỹ

end

function mean(d::NonStandardBeta)

    a, b, α, β = params(d)

    m = mean(Beta(α,β))

    m̃ = a + (b-a)*m

    return m̃

end

function rand(d::NonStandardBeta)

    a, b, α, β = params(d)

    z = rand(Beta(α,β))

    y = a + (b-a)*z

    return y

end

function rand(d::NonStandardBeta, n::Int)

    a, b, α, β = params(d)

    z = rand(Beta(α,β),n)

    y = a .+ (b-a)*z

    return y

end



function nsbetabayesfit(y::Array{Float64} ; niter::Int=5000, warmup::Int=1000, thin::Int=1, stepSize::Array{Float64}=[.02, .75, .05] )

μ = zeros(niter)
ν = zeros(niter)
θ = zeros(niter)

acc_cand = falses(niter,3)


# Get the parameter initial values
# M = fit(Beta, y)
#
# α = params(M)[1]
# β = params(M)[2]
α = 1
β = 1

μ[1] = α/(α+β)
ν[1] = α+β
θ[1] = 0

for i=2:niter

        cand = rand(Normal(μ[i-1],stepSize[1]))
        α̃ = cand*ν[i-1]
        β̃ = ν[i-1]*(1-cand)

        α = μ[i-1]*ν[i-1]
        β = ν[i-1]*(1-μ[i-1])

        f = NonStandardBeta(θ[i-1],1,α,β)
        f̃ = NonStandardBeta(θ[i-1],1,α̃,β̃)

        lr = loglikelihood(f̃,y) - loglikelihood(f,y)
        u = rand(Uniform())
        if lr > log(u)
            μ[i] = cand
            α = cand*ν[i-1]
            β = ν[i-1]*(1-cand)
            acc_cand[i,1] = true;
        else
            μ[i] = μ[i-1]
        end

        cand = abs(rand(Normal(ν[i-1],stepSize[2])))
        α̃ = μ[i]*cand
        β̃ = cand*(1-μ[i])

        f = NonStandardBeta(θ[i-1],1,α,β)
        f̃ = NonStandardBeta(θ[i-1],1,α̃,β̃)

        lr = loglikelihood(f̃,y) - loglikelihood(f,y) - log(cand) + log(ν[i-1])
        u = rand(Uniform())
        if lr > log(u)
            ν[i] = cand
            acc_cand[i,2] = true;
            α = μ[i]*cand
            β = cand*(1-μ[i])
        else
            ν[i] = ν[i-1]
        end

        cand = abs(rand(Normal(θ[i-1],stepSize[3])))

        f = NonStandardBeta(θ[i-1],1,α,β)
        f̃ = NonStandardBeta(cand,1,α,β)

        lr = loglikelihood(f̃,y) - loglikelihood(f,y)
        u = rand(Uniform())
        if lr > log(u)
            θ[i] = cand
            acc_cand[i,3] = true;
        else
            θ[i] = θ[i-1]
        end

    end

    taux_acc = [count(acc_cand[warmup+1:end,j]) ./ (niter-warmup) for j=1:3]

    println("Acceptation rate for μ is $(taux_acc[1])")
    println("Acceptation rate for ν is $(taux_acc[2])")
    println("Acceptation rate for θ is $(taux_acc[3])")

    if any(taux_acc .< .3) | any(taux_acc .> .7)
        @warn "Acceptation rate is not optimal. Try to provide different step sizes in the random walks."
    end

#     MCMC = Dict(:μ => μ[warmup+1:end], :ν => ν[warmup+1:end], :θ => θ[warmup+1:end])

    itr = (warmup+1):thin:niter

    # MCMC = DataFrame( Iteration = 1:length(itr), μ = μ[itr], ν = ν[itr], θ = θ[itr])

    a = θ[itr]
    α = μ[itr].*ν[itr]
    β = ν[itr].*(1.0 .- μ[itr])

    fd = NonStandardBeta.(a,1,α,β)

    return fd

end


function qqestimation(data::Array{Float64,1}, y::Array{Float64,1})

    q = sort(data)
    n = length(q)
    p = (collect(1:n) .- .5) / n

    q̂ = similar(q)

    for i=1:n

        q̂[i]= quantile(y,p[i])

    end

    df_QQplot = DataFrame(Q = q, Q̂ = q̂)

    return df_QQplot

end


function qqestimation(data::Array{Float64,1}, fd::Distribution)

    q = sort(data)
    n = length(q)
    p = (collect(1:n) .- .5) / n

    q̂ = quantile.(fd,p)

    df_QQplot = DataFrame(Q = q, Q̂ = q̂)

    return df_QQplot

end

function qqestimation(data::Array{Float64,1}, fd::Array{T,1} where T<:Distribution)

    q = sort(data)
    n = length(q)
    p = (collect(1:n) .- .5) / n

    q̂ = similar(q)

    for i=1:n

        qi = quantile.(fd,p[i])
        q̂[i] = mean(qi)

    end

    df_QQplot = DataFrame(Q = q, Q̂ = q̂)

    return df_QQplot

end

function qqci(p::Array{<:Real,1}, fd::Array{T,1} where T<:Distribution)

    n = length(p)

    qinf = Array{Float64}(undef,n)
    qsup = Array{Float64}(undef,n)
    qmean = Array{Float64}(undef,n)

    for j=1:length(p)

        qmcmc = quantile.(fd,p[j])
        qinf[j] = quantile(qmcmc,.025)
        qsup[j] = quantile(qmcmc,.975)
        qmean[j] = mean(qmcmc)

    end

    df_ci = DataFrame(Qmean = qmean, Qinf = qinf, Qsup = qsup)

    return df_ci

end
