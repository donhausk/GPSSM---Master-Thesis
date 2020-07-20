

"""

    FullGPInferece

The full GP


"""


struct FullGPInference <: InferenceModel end

"""

    SparseInference

The sparese GP

"""

struct SparseInference <: MarkovianInferenceModel end

get_additional_args_keys(::Type{<:SparseInference}) = (:mu, :Sigma,  :Z_Ind, )


"""

    HybridVIInference

The sparese GP resulting from a VI approximation

"""
struct HybridVIInference <: MarkovianInferenceModel end

get_additional_args_keys(::Type{<:HybridVIInference}) = (:Z_Ind, :Mu1, :Mu2)


"""

    PredictionInference

A fixed GP, which is used for inference

"""

struct PredictionInference<:MarkovianInferenceModel end

get_additional_constant_args_keys(::Type{<:PredictionInference}) = (:trainedtransition,)
"""
    extract_args(T, nt)

## Arguments
    - T::Type{<:InferenceModel}:    The inference Type
    - nt::NamedTuple:               The Namedtuple


Returns a NamedTuple containing only the additional args specified by T

"""

function extract_additional_args(T::Type{<:InferenceModel}, nt::NamedTuple)
    keys = get_additional_args_keys(T)
    vals = [nt[v] for v in keys]
    return NamedTuple{keys}(Tuple(vals))
end

function extract_additional_constant_args(T::Type{<:InferenceModel}, nt::NamedTuple)
    keys = get_additional_constant_args_keys(T)
    vals = [nt[v] for v in keys]
    return NamedTuple{keys}(Tuple(vals))
end



"""

    get_additional_args_keys

##Arguments
    -T::Type{<:InferenceModel}: The type of the infernce model

Returns the additional arguments used by the particle filter which are updated everystep.
Every of these args is stored in as a vector!!.


"""
get_additional_args_keys(::Type{<:InferenceModel}) = Tuple([])



"""

    get_additional_constant_args_keys

##Arguments
    -T::Type{<:InferenceModel}: The type of the infernce model

Returns the additional arguments used by the particle filter which are considered as constant.


"""
get_additional_constant_args_keys(::Type{<:InferenceModel}) = Tuple([])





struct BaseLSModel{NT, NTA, HP} <: AbstractBaseLSModel where {NT<:NamedTuple,NTA <:NamedTuple, HP<:HyperParams}
	args::Vector{NT} # Observations, Controls and lssize
	additional_args::Vector{NTA} # Additional args such as indcuing points etc.
	hyperparams::Vector{HP}
end


"""
	BaseLSModel(mdl, alg, args, inputs)

## Arguments
	- alg::AbstractLSAlgorithm:		The Latentstate Algorithm
	- hp::HyperParams:				The Hyper Parameter struct
	- args::NamedTuple				Arugments containting additional parameters such as inducing points
	- inputs::NamedTuple:			Inputs of the sampler ( Observations, Sizes, Controls)

# This is the base model struct which is always instantiated. It contains only a vector of NamedTuples. These are
# used as inputs for the step functions of the specific latent state algorithms.
#
#
# Depending on the AbstractModel the args have to be built in a different way to, which is done by overloading the BaseLSModel function
# and the update_args! function.

Attention, the hyper parameters are stored in vectors in order to manipulate them while keeping type stabiliy!
"""


function BaseLSModel(
	alg::BaseLSAlgorithm,
	hp::HyperParams,
    args::NamedTuple,
	inputs::NamedTuple
)
	additional_args = []
	baseargs = []
	# Multi Sequence Training, a different set for every sequence.
	for j =1:inputs[:sizes][3]
		push!(baseargs, (Y =  inputs[:observations][:,:,j],
		       U =  inputs[:controls][:,:,j],
		       lssize= inputs[:sizes]))

	    push!(additional_args, build_model_args(inputs, alg, args, j))
	end
	#all have the same type
    return BaseLSModel(baseargs, additional_args, [hp])
end



"""
    build_model_args(mdl,inputs, alg::A, args, i::Int=1)

## Arguments
   - mdl::AbstractLSModel:       The LS Model.
   - inputs::NamedTuple:         The inputs of the sampler (observations, controls, sizes etc).
   - alg<:AbstractLSAlgorithm    The LS Algorithm.
   - args::NamedTuple:           NamedTuple containting the Hyper parameters etc.
   - i::Int:                     The index of the multi squence training chain.


Depending on the LS Model, we have to build the arguments in a different way to call the step! function.
"""

@inline function build_model_args( inputs::NamedTuple, alg::BaseLSAlgorithm{I,A}, args::NamedTuple, i::Int=1) where {I<:InferenceModel, A<: AbstractLSAlgorithm}
		merge(extract_additional_args(I, args), extract_additional_constant_args(I, args))
end

@inline function build_model_args( inputs::NamedTuple, alg::BaseLSAlgorithm{I,A}, args::NamedTuple, i::Int=1) where {I<:InferenceModel, A<: AbstractLSAlgorithm}
	merge((xref=zeros((inputs[:sizes][1], inputs[:sizes][2]+1)), ), extract_additional_args(I, args), extract_additional_constant_args(I, args)) # Attention, we need to modify this to inlcude XT

end


"""
	update_args!(mdl::BaseLSModel, new_args, i::Int=1)

## Arguments
	- mdl<:BaseLSModel:		The BaseLSModel
	- new_hp:				New HyperParams
	- new_args:				New additional arguments
	- i::Int:				Indicates that we are sampling from the i-th chain. This is importatn for MS sampling.

# Update the arguments for the step function of the latent state sampler. Note that i stands for the i'th sequence to allow multi sequence training.

"""

# Could we make this generated?
function update_args!(alg::BaseLSAlgorithm{I,A}, mdl::BaseLSModel, new_hp::HyperParams, new_args::NamedTuple, i::Int=1) where {I<:InferenceModel, A<: AbstractLSAlgorithm}
	mdl.hyperparams[1] = new_hp
	update_algspecific_args!(alg, mdl, new_hp, new_args, i)
	additional_args = extract_additional_args(I, new_args)

	# copy all extra arguemnts, which are stored in keys as assumed.
	for key in keys(additional_args)
		mdl.additional_args[i][key][:] = new_args[key][:]
	end
	return mdl
end

function update_algspecific_args!(alg::BaseLSAlgorithm{I,A}, mdl::BaseLSModel, new_hp::HyperParams, new_args::NamedTuple, i::Int=1) where {I<:InferenceModel, A<: AbstractLSSMCAlgorithm}
	mdl
end
function update_algspecific_args!(alg::BaseLSAlgorithm{I,A}, mdl::BaseLSModel, new_hp::HyperParams, new_args::NamedTuple, i::Int=1) where {I<: InferenceModel, A<: AbstractLSPGAlgorithm}
	mdl.additional_args[i][:xref][:] = new_args[:xref][:]
end




"""

	PGASInstance{H, A, S, E}

## Fields
	- hyperparams:		Hyperparameters
	- xref:				Reference trajectory
	- storage:			A storage instance containing for example the cholesky decompositions or the inducing points
	- emission:			An instance containing information on the Emission model.

"""

struct ParticleFilterInstance{ H <: HyperParams, A<:Union{Nothing,AbstractArray}, S<:PFStorage, EM<:EmissionType} <: AbstractPFInstance
	hyperparams::H
	xref::A
	storage::S
	emission::EM
end



"""
	ParticleFilterInstance(alg::AbstractLSAlgorithm, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple, hyperparams::HyperParams; kwargs...)

## Arguments
    - alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
    - n_particles::Int              The amount of particles
	- Y::AbstractMatrix:			Observations
	- U::AbstractMatrix:			Control inputs
	- lssize::Tuple:				Sizes of the particle filter
	- hyperparams::HyperParams:		An instance of the hyper parameters.
    - kwargs...                     Additional Arguments as a named tuple

Returns a subtype of a ParticleFilterInstance.


"""


function ParticleFilterInstance(alg::BaseLSAlgorithm, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams;xref::M=nothing, XT::T=Float64, kwargs...) where {M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}
	# get the storage instance
	storage = get_storage(alg, n_particles, Y, U, lssize, hyperparams; xref=xref,  XT=XT,  kwargs...)
	# and the extras instance
	emission = get_emission(alg, n_particles, Y, U, lssize, hyperparams; xref=xref,  XT=XT,  kwargs...)
	ParticleFilterInstance(hyperparams, xref, storage, emission)
end





"""
	init_storage!(pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			ParticleFilterInstance
	- Xtraj:		Trajectories
	- n_particles:	Amount of particles
	- Y:			Observations
	- U:			Controls
	- lssize:		Sizes of the Timeseries (dim, T, nsubsequences)
	- hyperparams:	Hyper parameter struct
	- lpfs:			Log pdfs


Initial step for the storage.
"""

function init_pfinstance!(alg::BaseLSAlgorithm, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams , lpfs::LogPdfs)
	#nothing
end




"""
	step_pfinstance!(pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			ParticleFilterInstance
	- Xtraj:		Trajectories
	- n_particles:	Amount of particles
	- Y:			Observations
	- U:			Controls
	- lssize:		Sizes of the Timeseries (dim, T, nsubsequences)
	- hyperparams:	Hyper parameter struct
	- lpfs:			Log pdfs
	- i:			Iteration i= t-1

This function updates the pf instance every time step.

"""


function step_pfinstance!(alg::BaseLSAlgorithm, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )
	# do nothing
end

"""
	updatepfinstance!(alg, pf, indices, i)

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			ParticleFilterInstance
	- lssize::Tuple The specifications of the time series (dim xT x n_sequence)
	- Indices		Vector of new ancestor indices
	- i:			Iteration i= t-1

Updates the pfinstance struct after resampling.

"""
function reorderstorage!(alg::BaseLSAlgorithm, pf::ParticleFilterInstance,lssize::Tuple, indices::Vector{<:Integer}, i::Integer)
	# do nothing
end

"""
	DefaultStorage

An empty storage struct


"""

struct DefaultStorage <:PFStorage end



"""
	get_storage(alg, n_particles, Y, U, lssize, hyperparams; xref,  XT,  kwargs...)

## Arguments
- alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
	- n_particles::Int              The amount of particles
	- Y::AbstractMatrix:			Observations
	- U::AbstractMatrix:			Control inputs
	- lssize::Tuple:				Sizes of the particle filter
	- hyperparams::HyperParams:		An instance of the hyper parameters.
	- xref	                     	Reference Trajectory
	- XT							Type of latent states


"""






function get_storage(alg::BaseLSAlgorithm, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...) where {M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}
	return DefaultStorage()
end




## We do not need to implement the finalise_pf function because all samples have equal weight at the end!



"""

	GaussianEM

# Indicates a gaussian Emission with p(y_t|x_t) = N(Ax_t +b; S). In addition, C is the inverse of S in order to save computation.

"""

struct GaussianEM{MA,V,MS,MV} <: EmissionType
	A::MA
	b::V
	S::MS
	C::MV
end

"""

	TuringMCMCKernel

# Indicates a general Emission. We only store the transition kernel leaving p(x_t|y_t, x_{0:t-1}) invariant in form of a Turing sampler and a inference algorithm.
# Note that the Eval is a GAussianEM, which allows us to compute the marginal log pdf

"""
struct GeneralEMTuringKernel{M,A,E<:Union{Nothing, GaussianEM}} <: EmissionType
	model::M
	alg::A
	nrep::Integer
	Eval::E
end


"""

	AutoregressiveMCMCKernel

# Indicates a general Emission. We only store the transition kernel leaving p(x_t|y_t, x_{0:t-1}) invariant in form of a Turing sampler and a inference algorithm.
# Note that the Eval is a GAussianEM, which allows us to compute the marginal log pdf

"""
struct GeneralEMAutoregressiveKernel{A,E<:Union{Nothing, GaussianEM}} <: EmissionType
	ϵ::Float64
	hp::A
	Eval::E
end


"""
	DefaultEmission

A Default emission, which contains no information.

"""

struct DefaultEmission <: EmissionType end

"""
	get_emission_extra(alg::AbstractLSAlgorithm, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	   hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...) where {M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}



"""




"""
	get_emission(alg, n_particles, Y, U, lssize, hyperparams; xref,  XT,  kwargs...)

## Arguments
- alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
	- n_particles::Int              The amount of particles
	- Y::AbstractMatrix:			Observations
	- U::AbstractMatrix:			Control inputs
	- lssize::Tuple:				Sizes of the particle filter
	- hyperparams::HyperParams:		An instance of the hyper parameters.
	- xref	                     	Reference Trajectory
	- XT							Type of latent states



Returns an Emissionstruct which is a subtype of EmissionType. This contains information on the emission in order to perform the
fully adapted particle filtering and the mcmcapf particle filtering.

"""


function get_emission(alg::BaseLSAlgorithm, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
   hyperparams::HyperParams;xref::M=nothing,  XT::T=Float64, kwargs...) where {M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}
	return DefaultEmission()
end


##########################################################

### FAAPF Implementation

##########################################################


function get_emission(alg::BaseLSAlgorithm{I,A}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...) where { I<:InferenceModel, A<: FAAPF, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}

	#Depending on the input argument
	params =  get_gaussian_em(hyperparams.gpinstance, hyperparams.args_θy) # Extract the Gaussian parameters.

	if typeof(params[:S]) <: AbstractVector
		emissiont = GaussianEM(params[:A],params[:b],
			Diagonal(params[:S]),Symmetric(Diagonal(inv.(params[:S]))))
	else
		emissiont = GaussianEM(params[:A],params[:b],
			params[:S],inv(params[:S]))
	end

	return emissiont
end



##########################################################

### MCMCFAAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!

# Paper On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)



function get_emission(basealg::BaseLSAlgorithm{I,A}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...) where { I<:InferenceModel,  A2 <:Turing.Inference.InferenceAlgorithm ,A<: MCMCFAAPF{A2}, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}

	# Get the Turing model, which leaves N(x_t; μ_t, Σ_t) p(y_t| x_t ) invariant
	emodel = mcmcapfkernel(zeros(lssize[1]),ones(lssize[1]), Y[:,1], hyperparams)

	# If we have a gaussian emission...
	if basealg.alg.compute_mll
		#Depending on the input argument
		params =  get_gaussian_em(hyperparams.gpinstance, hyperparams.args_θy) # Extract the Gaussian parameters.

		if typeof(params[:S]) <: AbstractVector
			emissiont = GaussianEM(params[:A],params[:b],
				Diagonal(params[:S]),Symmetric(Diagonal(inv.(params[:S]))))
		else
			emissiont = GaussianEM(params[:A],params[:b],
				params[:S],inv(params[:S]))
		end
	else
		emissiont = nothing
	end
	# create emssion
	GeneralEMTuringKernel(emodel,basealg.alg.kernel_alg, basealg.alg.n_iter, emissiont)
end





function get_emission(basealg::BaseLSAlgorithm{I,A}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...) where { I<:InferenceModel,  A2 <: AutoRegressiveKernel ,A<: MCMCFAAPF{A2}, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}

	# Get the Turing model, which leaves N(x_t; μ_t, Σ_t) p(y_t| x_t ) invariant
	# If we have a gaussian emission...
	if basealg.alg.compute_mll
		#Depending on the input argument
		params =  get_gaussian_em(hyperparams.gpinstance, hyperparams.args_θy) # Extract the Gaussian parameters.

		if typeof(params[:S]) <: AbstractVector
			emissiont = GaussianEM(params[:A],params[:b],
				Diagonal(params[:S]),Symmetric(Diagonal(inv.(params[:S]))))
		else
			emissiont = GaussianEM(params[:A],params[:b],
				params[:S],inv(params[:S]))
		end
	else
		emissiont = nothing
	end
	# create emssion
	GeneralEMAutoregressiveKernel(basealg.alg.kernel_alg.ϵ, hyperparams, emissiont)
end





"""
	 propagate_single_particle

## Arguments
	- xtm:			The current state
	- am: 			The current ancestor index
	- mu:			The predictive means
	- sigma: 		The predicitve covariances
	- mcmc_trans:	Specifies the transition kernel for the latent states updates
	- Y:			The current observations.

We build a symmetric kernel. First, we resample the x_t with a kernel leaving p(x_t| x_{0:t-1}^am, y_t) invariant
then we sample anew proportional to p(x_t| x_{0:t-1}^i) and then we sample x_t' from a kernel leaving again
p(x_t'| x_{0:t-1}^anew, y_t) invariant. If the used kernel is symmetric, the overall transition kernel is symmetric and
therefore the procedure satisfies the requirements in: On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)

"""



function propagate_single_particle(xtm::AbstractVector, am::Integer,  mu::AbstractMatrix, sigma::AbstractMatrix, mcmc_trans::GeneralEMTuringKernel, Y::AbstractVector)
	mcmc_trans.model.args[:yt][:] = Y[:]

	# Sample a_{t-1}
	# Compute p(x_t|x_{t-1}^i)

	# Sample x_t
	mcmc_trans.model.args[:mu][:] = mu[:,am]
	mcmc_trans.model.args[:sigma][:] = sigma[:,am]

	niter = mcmc_trans.nrep
	# We generate a new Turing model to guarantee no mistakes. Unfortunaltely, this has to be made...
	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
	esampler = Turing.Sampler(mcmc_trans.alg, emodel, TrainableSelector)
	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, niter)
	Turing
	# Set current state ( This trick was proposed on the Turing webpage somewhere in the Issue section.)
	esampler.state.vi.metadata.x.vals[:] = xtm[:]
	# Run the sampler leaving p(x_t|x_{t-1}^am, y_t) invariant
	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
	for i =2:niter
		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
	end
	# Extract new state
	xnew = esampler.state.vi.metadata.x.vals


	Lps = get_logps(mu, sigma, xnew)
	# Resample index
	anew = resample(softmax!(Lps),1)[1]


	# Sample x_t
	mcmc_trans.model.args[:mu][:] = mu[:,anew]
	mcmc_trans.model.args[:sigma][:] = sigma[:,anew]

	niter = mcmc_trans.nrep
	# We generate a new Turing model to guarantee no mistakes. Unfortunaltely, this has to be made...
	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
	esampler = Turing.Sampler(mcmc_trans.alg, emodel, TrainableSelector)
	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, niter)
	Turing
	# Set current state ( This trick was proposed on the Turing webpage somewhere in the Issue section.)
	esampler.state.vi.metadata.x.vals[:] = xtm[:]
	# Run the sampler leaving p(x_t|x_{t-1}^am, y_t) invariant
	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
	for i =2:niter
		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
	end
	# Extract new state
	xnew2 = esampler.state.vi.metadata.x.vals


	# Sample again a_{t-1}
	# Compute p(x_t|x_{t-1}^i)
	Lps = get_logps(mu, sigma, xnew2)
	# Resample index



	# Compute p(x_t|x_{t-1}^i) for ancestor weights, and return xnew2, anew and the transition probability.
	return xnew2, anew, Lps[anew]
end


function propagate_single_particle(xtm::V, am::Integer,  muv::M, sigma::M, mcmc_trans::GeneralEMAutoregressiveKernel, Y) where {V<:AbstractVector, F <: AbstractVector, M <: AbstractMatrix}

	# Sample a_{t-1}
	# Compute p(x_t|x_{t-1}^i)

	# Sample x_t
	mu = muv[:,am]
	L = Diagonal(sqrt.(sigma[:,am]))

	xnewprop = mu + sqrt(1- mcmc_trans.ϵ^2) * (xtm - mu) + mcmc_trans.ϵ .* L * rand(Normal(), length(xtm))

	distxy, yargs = get_distxy_and_args( mcmc_trans.hp.gpinstance, vec(xtm), mcmc_trans.hp.args_θy...)

	pold =logpdf(distxy(yargs...), Y)::Float64
	distxy, yargs = get_distxy_and_args( mcmc_trans.hp.gpinstance, vec(xnewprop), mcmc_trans.hp.args_θy...)

	pnew =logpdf(distxy(yargs...), Y)::Float64

	r = rand(Uniform(0,1))
	accept = r <= exp(pnew-  pold) ? true : false
	if accept
		xtm = xnewprop

	end


	Lps = get_logps(muv, sigma, vec(xtm))
	# Resample index
	anew = resample(softmax!(Lps),1)[1]



	mu = muv[:,anew]
	L = Diagonal(sqrt.(sigma[:,anew]))

	xnewprop = mu + sqrt(1- mcmc_trans.ϵ^2) * (xtm - mu) + mcmc_trans.ϵ .* L * rand(Normal(), length(xtm))

	distxy, yargs = get_distxy_and_args( mcmc_trans.hp.gpinstance, vec(xtm), mcmc_trans.hp.args_θy...)

	pold =logpdf(distxy(yargs...), Y)::Float64
	distxy, yargs = get_distxy_and_args( mcmc_trans.hp.gpinstance, vec(xnewprop), mcmc_trans.hp.args_θy...)

	pnew =logpdf(distxy(yargs...), Y)::Float64

	r = rand(Uniform(0,1))
	accept = r <= exp(pnew -pold ) ? true : false
	if accept
		xtm = xnewprop
	end


	# Sample again a_{t-1}
	# Compute p(x_t|x_{t-1}^i)
	Lps = get_logps(muv, sigma, vec(xtm))
	# Resample index


	# Compute p(x_t|x_{t-1}^i) for ancestor weights, and return xnew2, anew and the transition probability.
	return xtm, anew, Lps[anew]
end



"""

Turing Models form MCMC step used by MCMCAPF sampler

"""

# Kernel leaving p(x_t| y_t, x_{0:t-1}) ∝ N(x_t; mu, sigma)*p(y_t| x_t) invariant.
Turing.@model mcmcapfkernel(mu, sigma, yt, hyperparams) = begin
    diagelements = sigma
    Σ = Diagonal(diagelements)
    x ~ MvNormal(mu, Σ) #  x is drawn from a MvNormal

    distxy, yargs = get_distxy_and_args(hyperparams.gpinstance, x, hyperparams.args_θy...)
    yt ~ distxy(yargs...)
end
