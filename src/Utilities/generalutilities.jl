
"""

Multivariate Normal is not Zygote compatible ! This is a work around.

"""

struct Zygote_MvNormal{Tx<:AbstractVector, TY<:AbstractMatrix} <: ContinuousMultivariateDistribution
    x::Tx
    Σy::TY
end

function Distributions.logpdf(f::Zygote_MvNormal, y::AbstractVector{<:Real})
    μ, C = mean(f), cholesky(Symmetric(cov(f))).L
    V = C\ (y - μ)
    XinvA = V'V
    return -(length(y) * log(2π)  + XinvA) / 2 - logdet(C)
end
# In case we store the upper Triangular.
function Distributions.logpdf(f::Zygote_MvNormal{Tx,TY}, y::AbstractVector{<:Real}) where {Tx<:AbstractVector{<:Real} , TY <: UpperTriangular}
    μ, U = mean(f), f.Σy

    XinvA = A_Kinv_A(U, (y - μ))
    return -(length(y) * log(2π)  + XinvA)/2 - logdet(U)
end



Zygote_MvNormal(x::AbstractVector, σ²::AbstractVector{<:Real}) = Zygote_MvNormal(x, Diagonal(σ²))
Zygote_MvNormal(x::AbstractVector, σ²::Real) = Zygote_MvNormal( x, Fill(σ², length(x)))


Base.length(f::Zygote_MvNormal) = length(f.x)
Distributions.mean(f::Zygote_MvNormal) =f.x
Distributions.cov(f::Zygote_MvNormal) = f.Σy
Distributions.cov(f::Zygote_MvNormal{Tx, TY}) where {Tx<:AbstractVector{<:Real} , TY <: UpperTriangular} = f.Σy'*f.Σy


function Distributions.rand(rng::AbstractRNG, f::Zygote_MvNormal, N::Int)
    μ, C = mean(f), cholesky(Symmetric(cov(f)))
    return μ .+ C.U' * randn(rng, length(μ), N)
end

function Distributions.rand(rng::AbstractRNG, f::Zygote_MvNormal{Tx, TY}, N::Int)where {Tx<:AbstractVector{<:Real} , TY <: UpperTriangular}
    μ, U = mean(f), f.Σy
    return μ .+ U' * randn(rng, length(μ), N)
end

Distributions.rand(f::Zygote_MvNormal, N::Int) = rand(Random.GLOBAL_RNG, f, N)
Distributions.rand(rng::AbstractRNG, f::Zygote_MvNormal) = vec(rand(rng, f, 1))
Distributions.rand(f::Zygote_MvNormal) = vec(rand(f, 1))




"""
    dropnames(namedtuple, names )

## Arguments
    - namedtuple::NamedTuple
    - names::Tuple{Vararg{Symbol}}


# Drop one element from a NamedTuple.

"""
function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}})
      keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
      return NamedTuple{keepnames}(namedtuple)
end



"""
    resample(w, num_particles)

## Arguments
    - w::AbstractVector{<:Real}:    Resampling weights
    - num_particles::Integer:       Amount of samples

# Taken from Turing. These functions are well Tested!

"""

function resample(w::AbstractVector{<:Real}, num_particles::Integer=length(w))
    return resample_systematic(w, num_particles)
end


function resample_systematic(weights::AbstractVector{<:Real}, n::Integer)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")
    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand())
    # find all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")
            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end
        # save the next sample
        samples[i] = sample
        # update `u`
        u += one(u)
    end
    return samples
end


"""
    clip_gradient(x, lower, upper)

## Arguments

    - x:      A value, vector or matrix
    - lower:  Lower value for the clipping
    - upper:  Upper value for the clipping

"""
function clip_gradient(x, lower, upper)
    return min.(max.(x,lower), upper)
end

clip_gradient(x, bounds) = clip_gradient(x, - bounds, bounds)


"""
    ess(Ws::AbstractVector)

## Arguments
    - Ws::AbstractVector{<:Real}:       Vector of weights

Compute the effective sample size
"""
function ess(Ws::AbstractVector{<:Real})
    return sum(Ws)^2/sum(Ws.^2)
end
