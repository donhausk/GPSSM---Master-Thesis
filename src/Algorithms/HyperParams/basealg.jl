


############################################################
# Hyper Parameter Algorithms
############################################################

#
#
# General Note
#
# Flux optimiser only allow gradient descent algorithms. Therefore, when mentioning a Flux optimiser,
# gradient descent is automatically converted into gradient ascent by taking the negation of the gradient.
#
#
#


"""

Every Hyper Parameter Algorithm has the following structure:

struct Algname{I} <: AbstractHPAlgorithm
    arguments....
    function Algname(...) = Algname{<:InferenceModel}(...)
end

The inference modle is then used to define the specific Sampler.

"""

"""
    get_inferencemodel(alg::AbstractHPAlgorithm)
## Arguments
    - alg:  The Hyper paramtere inference model

Returns a instance of the Inference Model used by the specific hyper parameter algorithm.

"""
get_inferencemodel(alg::AbstractHPAlgorithm) = error("No Infrence model is specified for this algorithm.")







"""
    ConstantHP(hp,additional_parameters,  trainedtransition)

## Arguments
    - hp::HyperParams:             The pretrained hyperparams
    - additional_parameters:       The pretrained additional_parameters
    - trainedtransition:           An instance of TrainedTransition struct


# Set the hyper parameters to be constant.

"""


struct ConstantHP{HP, NT, T} <: AbstractHPAlgorithm where{hyperparams<:HyperParams,  NT <: NamedTuple, T<: TrainedTransition}
    hyperparams::HP
    additiononal_args::NT
    trainedtrans::T
    # function ConstantHP(hyperparams::HyperParams, additiononal_args::NamedTuple, trainedtransition::TrainedTransition)
    #     new{PredictionInference, typeof(hyperparams), typeof(additiononal_args), typeof(trainedtrans)}(hyperparams, additiononal_args, trainedtransition)
    # end
end

function ConstantHP(instance::GPSSMInstance, args::NamedTuple, tranistion::TrainedTransition) where {I<:InferenceModel, A<:AbstractLSAlgorithm}
    hp = restore_from_namedtuple(instance, args)
    return ConstantHP(hp, (trainedtransition = tranistion, ), tranistion)
end


get_inferencemodel(::ConstantHP) = PredictionInference()



"""
	generate_grad_f(vi, hmp, Xin)

## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.
	- Xin::AbstractArray: 						The latent states. Either dim x T+1 x 1 xn_chains or dim x T+1 x n_particles x n_chains
	- kwargs::NamedTuple:						Additional arguments like Mu1, Mu2 for the hybrid vi case.


The standard way of how to generate a vector of gradient functions used for differentiation in the frequentist approaches.

"""

function generate_grad_f(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel, Xin::AbstractArray{<:Real, 4}; kwargs...)
	# This is not type stable
	Xtemp = deepcopy(Xin) # We need to copy this!
	space = _getspace(vi, TrainableSelector)

	# A vector of return functions.
	ls_lpfs = []

	@views for j = 1: size(Xin)[4]
		Y = hpm.inputs.Y[:,:,j]
		U = hpm.inputs.U[:,:,j]
		for i = 1:size(Xin)[3]
			push!(ls_lpfs, get_lpf(vi, hpm, space, Xin[:,:,i,j], Y, U ; kwargs...))
		end
	end

	return ls_lpfs
end

"""
	get_lpf(vi, hmp, X, Y, U)

## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.
	- spec::Val{<:NamedTuple}:					Containing information which variables should be trained.
	- X::AbstractMatrix: 						The latent states of dimesnion dim x T+1
	- Y::AbstractMatrix:						The observations of dimension dimy xT
	- U::AbstractMatrix: 						The control variables of dimension dimcxT+1
	- kwargs::NamedTuple:						Additional arguments like Mu1, Mu2 for the hybrid vi case.

Returns the logpdf estimate for a single latent states trajectory X. In the end, we average over all of them.


"""

function get_lpf(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel, space::Val, X::AbstractMatrix, Y::AbstractMatrix, U::AbstractMatrix; kwargs...)
	error("This function must be specified!")
end

"""
	extract_hyper_params(vi, hmp)

## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.

Extract the kerenl, mean, Q, and args_θy from the Vi instance

"""





"""
    get_grad_and_lpf(Fs, θ)

## Arguments
    - Fs:           List of functions computing the lpf estimate
    - θ:            The position where the gradients should be evaluated.
    - Ws:           List of Weights.
    - compute_grad: Whether to compute the gradients.
Compute the gradients and the Log Pdf averaged over Fs multiplied by the weights.

"""



function get_grad_and_lpf(Fs::AbstractVector, θ::AbstractVector; Ws::Union{Nothing,AbstractVector} = nothing, compute_grad::Bool=true)
    grad = zeros(size(θ)) # Gradient has same shape as θ
    val = 0.0 # The evaluation of Fs at θ
    # The weights
    if Ws === nothing
        Ws = ones(length(Fs)) .* 1.0/length(Fs)
    end

    for (c,f) in enumerate(Fs)
        lenf = length(f)
        for el in f
            if compute_grad
                #lp = el(θ)
                #gr = ForwardDiff.gradient(el, θ)
                #lp, gr= Zygote.pullback(el, θ)
                # We average over f and take the weighted sum over Ws
                #grad +=  (1/lenf * Ws[c]) .* gr
                lp, gr= Zygote.pullback(el, θ)

                # We average over f and take the weighted sum over Ws
                grad +=  (1/lenf * Ws[c]) .* gr(1)[1]
            else
                lp = el(θ)
            end
            # We average over f and take the weighted sum over Ws
            val += Ws[c] * lp/lenf
        end
    end
    return val, grad
end

"""

    update_diffs!(spl::AbstractHPOptimiser, mdl::AbstractHPModel, X::AbstractArray,  iteration::Integer)

## Arguments
    - spl::AbstractHPOptimiser:         The sampler
    - mdl::AbstractHPModel:             The HP Model
    - X::AbstractArray:                 The new states
    - iteration::Integer                The current itertation
Applys the lag to the diff storage and returns the new function ans weights.

"""


function update_diffs!(spl::AbstractHPOptimiser, mdl::AbstractHPModel, X::AbstractArray, iteration::Integer; kwargs...)
    # Get a new gradient generating function with the current states
    new_f  = generate_grad_f(spl.state.vi, mdl, X; kwargs...)
    # Add it to the set of gradient generating functions
    push!(spl.state.diffs, new_f)

    # Get γ_k of the EM algorithm
    ng = spl.alg.get_gamma(iteration)

    # Update weights
    if iteration == 1
        push!(spl.state.Ws, log(ng))
    else
        spl.state.Ws[:] = spl.state.Ws .+ log(1.0-ng) # We use the log numerical stability reasons
        push!(spl.state.Ws, log(ng))
    end

    # If we use a lag, we only consider the last spl.alg.lag elements
    if spl.alg.lag >0 && length(spl.state.diffs)>=spl.alg.lag
        spl.state.Ws = spl.state.Ws[end-spl.alg.lag+1:end]
        spl.state.diffs = spl.state.diffs[end-spl.alg.lag+1:end]
    end

    Ws = softmax!(copy(spl.state.Ws))
    Fs = spl.state.diffs
    return Fs, Ws
end
