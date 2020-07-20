"""

    BaseAlgorithm

## Fields
    - alg::AbstractLSAlgorithm:     Latentstates Algorithm

Base struct storing information on the inference model and the latent states algorithm.

"""

struct BaseLSAlgorithm{I<:InferenceModel, ALG<:AbstractLSAlgorithm}
    alg::ALG
end


function BaseLSAlgorithm(alg::AbstractLSAlgorithm, ahp::AbstractHPAlgorithm)
    infmdl = get_inferencemodel(ahp)
    check_compatibility(alg, infmdl) # This is important to reduce errors
    BaseLSAlgorithm{typeof(infmdl),typeof(alg)}(alg)
end




mutable struct LSState{A}
    X::A
end

struct LSSampler{I,A, S} <: AbstractLSSampler where {I<:InferenceModel, A<:AbstractLSAlgorithm, S}
    alg::BaseLSAlgorithm{I,A}
    state::S
    inputs::NamedTuple
end

# We can use the AdvancedPF Sampler!
@inline function Sampler(infalg::BaseLSAlgorithm{I,A}, inputs::NamedTuple) where {
    I<:InferenceModel, A<:AbstractLSPGAlgorithm
}
    s = LSState(zeros(get_latent_type(),inputs[:sizes][1], inputs[:sizes][2]+1, 1 ,inputs[:sizes][3]))
    return LSSampler{I, A, typeof(s)}(infalg, s, inputs)
end
@inline function Sampler(infalg::BaseLSAlgorithm{I,A}, inputs::NamedTuple) where {
    I<:InferenceModel, A<:AbstractLSSMCAlgorithm
}
    s = LSState(zeros(get_latent_type() ,inputs[:sizes][1], inputs[:sizes][2]+1, infalg.alg.n_particles, inputs[:sizes][3]))
    return LSSampler{I, A, typeof(s)}(infalg, s, inputs)
end

# This function returns a model!
function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    sampler::LSSampler,
    N::Integer,
    hp::HyperParams,
    additional_args::NamedTuple;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)

    bmdl = BaseLSModel(sampler.alg, hp, additional_args, sampler.inputs)
    for i in 1:sampler.inputs[:sizes][3]
        AbstractMCMC.sample_init!(
            rng,
            bmdl,
            sampler,
            N;
            i = i,
            verbose=verbose,
            resume_from=resume_from,
            kwargs...
            )
        # Perform the first step! This is important because we have already sampled the hyperparamters using sampe from prior.
    end
    return bmdl, (X = deepcopy(sampler.state.X),)
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    mdl::AbstractModel,
    sampler::LSSampler,
    N::Integer,
    hp::HyperParams,
    additional_args::NamedTuple;
    kwargs...
)
    # We iterate over all the parallel sequences. This could be done using Thread.@threads macro
    for i = 1:sampler.inputs[:sizes][3]
        # Extract parameters from the new_param argument
        params = extend_args(additional_args, sampler, i)
        # Update the arugments
        update_args!(sampler.alg, mdl, hp, params, i)
    end
    ts = Vector{NamedTuple}()
    # This could be done in parallel!!!
    for i =1:sampler.inputs[:sizes][3]
        # Sample latent states.
        push!(ts, AbstractMCMC.step!(rng, mdl, sampler, N; i=i, kwargs...))
    end
    # For SMC sampler, we only store the first five ones for tracking.
    return ConstTransition(tstonamedtuple(ts)), (X= deepcopy(sampler.state.X),)

end



"""
     extend_args(args, s, i)

## Arguments
     - args::NamedTuple
     - s::AbstractHPSampler
     - i::Integer

# This method is used to extend the arguments passed to the latent state sampler.


"""

function extend_args(args::NamedTuple, s::LSSampler{I,A}, i::Integer) where {I<:InferenceModel,A <: AbstractLSPGAlgorithm }
    return merge(args, (xref = s.state.X[:,:,i], ))
end

function extend_args(args::NamedTuple, s::LSSampler{I,A}, i::Integer) where {I<:InferenceModel,A <:AbstractLSSMCAlgorithm}
    return args
end

"""
    tstonamedtuple(ts)

## Arguments
    - ts::AbstractVector{<:NamedTuple}:     the transition for each chain

Flattens to vector of NamedTuples to a single NamedTuple.

"""
tstonamedtuple(ts::AbstractVector{<:NamedTuple}) = length(ts) === 1 ? ts[1] :  NamedTuple{Tuple([Symbol("chn_$i"*"_"*string(key))  for i =1:length(ts) for key in keys(ts[i])])}(Tuple([ti  for t in ts for ti in t]))




"""

The step functions specified for the distinct PF algorithms. This could be done in a single function.


"""



function AbstractMCMC.step!(rng::AbstractRNG, mdl::AbstractModel, spl::LSSampler{I, A}, N::Integer; i::Integer=1, kwargs...) where {I<:InferenceModel,A<: AbstractLSSMCAlgorithm}
	X, W_s, Info, _ ,_= AbstractMCMC.step!(spl.alg, spl.alg.alg.n_particles, mdl.hyperparams[1], mdl.args[i]...; mdl.additional_args[i]...)

	spl.state.X[:,:,:,i] = X[:,:, :]
	return Info
end

function AbstractMCMC.step!(rng::AbstractRNG, mdl::AbstractModel, spl::LSSampler{I, A}, N::Integer; i::Integer=1, kwargs...) where {I<:InferenceModel,A<: AbstractLSPGAlgorithm}
	#xref = mdl.args[i][:xref]
	X, W_s, Info, _ , _= AbstractMCMC.step!(spl.alg, spl.alg.alg.n_particles, mdl.hyperparams[1], mdl.args[i]...; mdl.additional_args[i]...)
	idx = resample(W_s, 1)
	spl.state.X[:,:,:,i] = X[:,:, idx]
	return Info
end

function AbstractMCMC.sample_init!(rng::AbstractRNG, mdl::AbstractModel, spl::LSSampler{I, A}, N::Integer; i::Integer=1, kwargs...) where {I<:InferenceModel,A<: AbstractLSSMCAlgorithm}
	X, W_s, Info, _ ,_ = AbstractMCMC.step!(get_init_alg(spl.alg), spl.alg.alg.n_particles, mdl.hyperparams[1], mdl.args[i]...; mdl.additional_args[i]...)
	spl.state.X[:,:,:,i] = X[:,:, :]
	return Info
end


function AbstractMCMC.sample_init!(rng::AbstractRNG, mdl::AbstractModel, spl::LSSampler{I, A}, N::Integer; i::Integer=1, kwargs...) where {I<:InferenceModel,A<: AbstractLSPGAlgorithm}
	X, W_s, Info, _ ,_= AbstractMCMC.step!(get_init_alg(spl.alg), spl.alg.alg.n_particles, mdl.hyperparams[1], mdl.args[i]...; mdl.additional_args[i]...)
	idx = resample(W_s, 1)
	spl.state.X[:,:,:,i] = X[:,:, idx]
	return Info
end


###############################
#
### BaseAlgorithm
#
###############################



"""
	LogPdfs

## Fields
	- Sequentiallp::Vector:      Used for ancestor sampling. This is log p ( x_{1:t-1}| x_0 )
	- Joinglp::Vector:		     Joint log pdf for ancestor sampling. This is log p ( x_{1:t-1},xref{t:T}| x_0 )
	- Weightlp::Vector:			 Used for the weights, usually log p(y_t|x_t)
	- marginallp::Float64:		 Estimate of the margian log likelihood

Imporant log probaiblities used throughout the sampling.

"""


mutable struct LogPdfs
	Sequentiallp::Vector
	Jointlp::Vector
	Weightlp::Vector
	marginallp::Float64
end
LogPdfs(n_particles::Integer) = LogPdfs(zeros(n_particles),zeros(n_particles), zeros(n_particles), 0.0)

"""
	updatelogps!(lpf::LogPdfs, idx::Vector{<:Integer})

Simple function used to modify the cahnges after resampling


"""
function reorderlpfs!(lpf::LogPdfs, idx::Vector{<:Integer})
	lpf.Sequentiallp[:] = lpf.Sequentiallp[idx]
	lpf.Jointlp[:] = lpf.Jointlp[idx]
	lpf.Weightlp = lpf.Weightlp[idx]
end



"""
    SMC(n_particles)

## Arguments

    - n_particles: The amount of particles.

The SMC sampler


"""


struct SMC <: AbstractLSSMCAlgorithm
    n_particles::Integer
end



"""
    PGAS(n_particles)

## Arguments

    - n_particles: The amount of particles.


PGAS Implementation.

Original paper: Particle Gibbs with Ancestor Sampling (Lindsten, 2014)

"""

struct PGAS <: AbstractLSPGAlgorithm
    n_particles::Integer
end


"""
    FAAPF(n_particles)

## Arguments

    - n_particles: The amount of particles.


 The Fully Adapted Auxiliary particle Filter implementatoin.
 Oritinal Paper: Filtering via Simulation: Auxiliary Particle Filters, (Pitt, 1999))

"""

struct FAAPF <: AbstractLSPGFAAlgoirthm
    n_particles::Integer
end



"""
	AutoRegressiveKernel
## Arguments
	- ϵ::Float64:		The epsilon used for the updates

Turing has not implemented the autoregressive kernel by Neal. This is a workaround.

# insert quote here...
"""

struct AutoRegressiveKernel
	ϵ::Float64
end
AutoRegressiveKernel() = AutoRegressiveKernel(0.5)




"""
    MCMCFAAPF(n_particles, inf, nrepetitions, evaluate_gaussian)

## Arguments
    - n_particle::Integer:                          The amount of particles
    - inf::Turing.Inference.InferenceAlgorithm:     The Turing Inference Algorithm
    - nrepetitons::Integer:                         The amount of repetitions during each step.
    - compute_mll::Bool:                            If the emission is Gaussian, we can compute the unbiased estimate of the Marginal log likelihood.

MCMC-FA-APF Implementation.

Original paper: On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)
"""


struct MCMCFAAPF{A} <: AbstractLSPGFAAlgoirthm
    n_particles::Integer
	kernel_alg::A
	n_iter::Integer
    compute_mll::Bool
end
# By default, we use the ESS sampler
MCMCFAAPF(n_particles::Integer) = MCMCFAAPF{typeof(ESS())}( n_particles, ESS(),2, true)
MCMCFAAPF(n_particles::Integer; kernel_alg =ESS(), n_iter=2, compute_mll = true) = MCMCFAAPF{typeof(kernel_alg)}( n_particles, kernel_alg, n_iter, compute_mll)
