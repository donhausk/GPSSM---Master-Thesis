

###

# Shared among all Sparse  GP implementatoins:

###

"""
	 PrecditionFPExtras{T, E}

## Fields
	- transition:		An instance of trained transitions


"""

struct PrecditionStorage{T} <: PFStorage
	transition::T
end



function get_storage(alg::BaseLSAlgorithm{I}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, trainedtransition::TrainedTransition, kwargs...) where {I <: PredictionInference, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}
	return PrecditionStorage(trainedtransition)
end


function get_prior_transition!(alg::BaseLSAlgorithm{I}, pf::ParticleFilterInstance, Xtraj::AbstractArray,
    n_particles::Int,  Y::AbstractArray, U::AbstractArray,  lssize::Tuple, hyperparams::HyperParams, i::Integer) where I <: PredictionInference

	Xt = @view Xtraj[:,i,:]
	mus = zeros(size(Xt)[1], size(Xt)[2])
    Sigmas = zeros(size(Xt)[1], size(Xt)[2])
    @views for p = 1:size(Xt)[2]
        mut, st = predict_avg(vcat(Xt[:,p], U[:, i]), pf.storage.transition)
        mus[:,p] = mut
        Sigmas[:,p] = st
    end
    return mus, Sigmas
end


#########################################################

### PGAS Implementatoin

##########################################################

# We do not need to change anything here!



##########################################################

### FAAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!






##########################################################

### MCMCAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!

# Paper On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)



##########################################################

### SMC Implementatoin

##########################################################


# We do not need to change anything here!
