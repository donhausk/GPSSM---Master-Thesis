"""
	DefaultHPModel

## Fields
	- spce::GPSpec: 			An instance of the GPSpec
	- inputs::NamedTuple:		The inputs of the GP.
	- instance::GPSSMInstance:  The instance of the GPSSM model.
The default HP Model.


"""

struct DefaultHPModel{A, I, S, NT}<: AbstractHPModel where {A<:AbstractHPAlgorithm,  I<:GPSSMInstance, S<:GPSpec, NT<: NamedTuple}
	spec::S
	inputs::NT
	gpinstance::I
end


"""
	get_hp_model(hp, alg, inputs)

## Arguments
	- hp::GPSpec:				 The GP-Specification struct
	- alg::AbstractHPAlgorithm:	 The Hyper Parameter Algorithm
	- inputs::NamedTuple:		 The inputs of the sampling function, i.e. (observations, controls, sizes)
	- gpinstance::GPSSMInstance: The instance of the GPSSM model.


Returns the Hyper Parameter model
"""

function get_hp_model(
    hp::GPSpec,
	alg::AbstractHPAlgorithm,
	inputs::NamedTuple,
	gpinstance::GPSSMInstance

)
	# Extract base arguements

	lsspec = inputs[:sizes]
	Y = inputs[:observations]
	U = inputs[:controls]
	@assert lsspec[2]+1 == size(U)[2] && lsspec[3] == size(U)[3] " [GPSSM] The size of the controls must have (n_controls, N+1, K) with N observations and K chains"
	@assert lsspec[2] == size(Y)[2] && lsspec[3] == size(Y)[3] " [GPSSM] The size of the observations must have (n_out, N, K) with N observations and K chains"
	DefaultHPModel{typeof(alg), typeof(gpinstance), typeof(hp), typeof( (Y=Y,U=U,lsspec = lsspec))}(hp, (Y=Y,U=U,lsspec = lsspec), gpinstance)
end



"""
	get_hyperparams_from_vi(vi, hpm, inputs)

## Arguments
	- vi::VarInfo:				The VarInfo struct containing all the Hyper Parameters
	- hpm::AbstractHPFBModel:	The Hyper Parameter Model

Extract the hyper parameters from the VarInfo struct and returns including a NamedTuple with extra parameters.

"""

function get_hyperparams_from_vi(vi::VarInfo{<:NamedTuple}, hpm::AbstractHPModel)
    args_θx0 = get_distθx0_args(vi, hpm.spec)
    kernel_args = get_kernel_args(vi, hpm.spec)
    mean_args = get_mean_args(vi, hpm.spec)
	# Not very nice way of doing this...
	Q = vi.metadata.Q.vals
    args_θy = get_distθy_args(vi, hpm.spec)
	hps = HyperParams(kernel_args, mean_args , Q, args_θy , args_θx0, hpm.gpinstance)

    return hps, get_additional_args_from_vi(vi, hpm)
end



"""
	get_additional_args_from_vi(vi, hpm, inputs)

## Arguments
	- vi::VarInfo:				The VarInfo struct containing all the Hyper Parameters
	- hpm::AbstractHPFBModel:	The Hyper Parameter Model

Returns a NamedTuple with extra parameters.

"""

function get_additional_args_from_vi(vi::VarInfo{<:NamedTuple}, hpm::AbstractHPModel)
	return NamedTuple()
end



"""
	get_additional_args_from_spl(vi, hpm, inputs)

## Arguments
	- spl::AbstractHPSampler:	Hyper Parameter Samplers
	- hpm::AbstractHPFBModel:	The Hyper Parameter Model

Returns a NamedTuple with extra parameters.

"""

function get_additional_args_from_spl(spl::AbstractHPSampler, hpm::AbstractHPModel)
	return NamedTuple()
end




"""
	get_hyperparams(spl, hpm)

## Arguments
	- spl::AbstractHPSampler:	Hyper Parameter Samplers
	- hpm::AbstractHPFBModel:	The Hyper Parameter Model

This is only a intermediate function extracting the VarInfo instance form the sampler.

"""
function get_hyperparams(spl::AbstractHPSampler, hpm::AbstractHPModel)
	hyperparams, additional_args = get_hyperparams_from_vi(spl.state.vi, hpm)
	return hyperparams, merge(additional_args, get_additional_args_from_spl(spl, hpm))
end





function extract_hyper_params(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel)
	k_args = get_kernel_args(vi, hpm.spec)
	m_args = get_mean_args(vi, hpm.spec)
	kernels = get_kernels(hpm.gpinstance, k_args)
	mean_fs = get_means(hpm.gpinstance, m_args)
	Q = get_Q(vi)
	d0_args = get_distθx0_args(vi,hpm.spec)
	dy_args = get_distθy_args(vi, hpm.spec)

	return kernels, mean_fs, Q,d0_args, dy_args
end
