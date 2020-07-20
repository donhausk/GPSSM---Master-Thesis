
"""
 	HybridVIAlgorithm(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule ,schedulerate)

## Arguments
    - get_gamma::Function = (i) -> 1/i:             The γ_i used to update Q
    - optimiser= Flux.Descent(0.1):                 The optimiser for the hyper parameters specified using Flux
    - optimisermu1= Flux.Descent(0.1):              The optimiser for the update Mu1 specified using Flux
    - optimisermu2= Flux.Descent(0.1):              The optimiser for the update Mu2 specified using Flux
    - lag::Int = 1:                                 The lag of the algorithm, originally set to one, which is the algorithm proposed by Frigola.
    - burnin::Int = 0:                              The amount of burnin steps until the optimiser starts.
    - lrschedule::Int = 0:                          The lrschduler for the gradient maximisation, lrschedulue is the frequence the lr rate is multiplied by schedulerate
    - schedulerate::Float64 = 1.0:                  The schedulerate, hence, lrnew = lr * schedulerate
	- n_grad::Int = 0:								Amount of samples used for the gradient compuation. 0 means we use all.

The standard EM algorithm, where either a full maximisation step is performed when optimization_type
is set to "maximisation" or only a sinlge greadient step when set to  "gradient".


The VI algorithm proposed by Frigola. The different optimisers are all Flux Optimiser.

Original paper: Variational Gaussian Process State-Space Models (Frigola, 2014)

"""

struct HybridVIAlgorithm <: AbstractVIAlgorithm
    get_gamma::Function
    optimiserhp # Flux optimiser for the Hyper Parameter where Gradient Descent -> Gradient Ascent
    optimisermu1 # Flux optimiser for Mu1
    optimisermu2 # Flux optimiser for Mu2
    lag::Int # In case of PG as LS sampler, a lag defines how many of the last states are used for the greadient computation.
    burnin::Int # Self explaining
    lrschedule::Int # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
    schdulerate::Float64  # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
	n_grad::Int # We only use n_grad samples for the gradient computations

end
HybridVIAlgorithm(;n_grad = 0,  get_gamma::Function = (i) -> 1/i, optimiser= Flux.Descent(0.1),  optimisermu1= Flux.Descent(0.1),  optimisermu2= Flux.Descent(0.1),lag::Int = 1, burnin::Int = 0,lrschedule::Int = 0, schedulerate::Float64 = 1.0)= HybridVIAlgorithm( get_gamma, optimiser,optimisermu1, optimisermu2, lag, burnin, lrschedule ,schedulerate, n_grad) # No scheduling

get_inferencemodel(::HybridVIAlgorithm) = HybridVIInference()




mutable struct VIState
	vi::VarInfo
	Mu1::AbstractMatrix{<:Real}
	Mu2::AbstractArray{<:Real,3}
	diffs::Vector{Vector{Function}}
	Ws::Vector{<:Real}
	Xvals::Vector
end

struct VIOptimiser{ALG} <:AbstractHPVIOptimiser where {ALG <:HybridVIAlgorithm}
	alg::ALG
	state::VIState
	inputs::NamedTuple
end

VIState(alg::AbstractVIAlgorithm) = VIState(VarInfo(), zeros(GPSSM.get_latent_type(), 0,0), zeros(get_latent_type(),0,0,0),  [], Vector{typeof(alg.get_gamma(1))}(), [])
function Sampler(mdl::DefaultHPModel, inf::INF, inputs::NamedTuple) where INF<:AbstractVIAlgorithm
	spl =  VIOptimiser{INF}(inf,  VIState(inf), inputs)
	varinfo = generate_vi(mdl.spec)
	spl.state.vi = varinfo
	return spl
end


function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::DefaultHPModel,
    spl::VIOptimiser,
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)

	X_Init = get_X_Init(model.gpinstance, spl.inputs[:sizes])
	# This is a shitty initialization...
	Mu1, Mu2 = get_mu_init(spl.state.vi, model)

	spl.state.Mu1 = Mu1
	spl.state.Mu2 = Mu2

	@views for i = 1:spl.inputs[:sizes][1]
		spl.state.Mu2[:,:,i] = Hermitian(spl.state.Mu2[:,:,i]) # Overcome numerical errors
	end


	return  get_hyperparams(spl, model)
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    mdl::DefaultHPModel,
    spl::VIOptimiser,
    N::Integer,
    states::NT;
	iteration,
    kwargs...
) where {NT<:NamedTuple}

	#@views  if iteration > spl.alg.burnin
	@views begin

		push!(spl.state.Xvals, states[:X])
		if spl.alg.lag >0 && length(spl.state.Xvals)>=spl.alg.lag
	        spl.state.Xvals = spl.state.Xvals[end-spl.alg.lag+1:end]
	    end
		Mu1, Mu2 = compute_Mus(spl.state.vi,mdl, cat(spl.state.Xvals...; dims=3)) # Gradient descen -grad => gradient ascent
	end

	link!(spl.state.vi, TrainableSelector)
	# get the diffs functions
	newstates = spl.alg.n_grad == 0 ? states[:X] : @view states[:X][:,:,randperm(size(states[:X])[3])[1:spl.alg.n_grad],:]

	Fs, Ws = update_diffs!(spl, mdl, newstates, iteration; Mu1 = spl.state.Mu1, Mu2 = spl.state.Mu2)

	# Extract trainable hyper parameters.
	θ = spl.state.vi[TrainableSelector]
	if length(θ) > 0
		if iteration > spl.alg.burnin
			_, grad = get_grad_and_lpf(Fs, θ; Ws = Ws)
			new_x =  Flux.Optimise.update!(spl.alg.optimiserhp, θ, -clip_gradient(grad, DEFAULT_GRADIENT_CLIPPING)) # We want gradient ascent...
			spl.state.vi[TrainableSelector] = new_x
		end
	end
	invlink!(spl.state.vi, TrainableSelector)


	#@views if iteration > spl.alg.burnin
	@views begin
		spl.state.Mu1 = Flux.Optimise.update!(spl.alg.optimisermu1, spl.state.Mu1, spl.state.Mu1 - Mu1)  # Gradient descent -grad => gradient ascent
		spl.state.Mu2 =Flux.Optimise.update!(spl.alg.optimisermu2, spl.state.Mu2, spl.state.Mu2 - Mu2)  # Gradient descent -grad => gradient ascent
		for i = 1:spl.inputs[:sizes][1]
			spl.state.Mu2[:,:,i] = Hermitian(spl.state.Mu2[:,:,i]) # Overcome numerical errors
		end
	end
	#end

	link!(spl.state.vi, TrainableSelector)
	if length(θ) > 0
		θ = spl.state.vi[TrainableSelector]
		elbo,_ = get_grad_and_lpf(Fs, θ; Ws = Ws, compute_grad=false)
	else
		elbo = 0.0
	end
	invlink!(spl.state.vi, TrainableSelector)
	nt = dropnames(tonamedtuple(spl.state.vi), (:Mu1, :Mu2, :Z_Ind))


	return ConstTransition(merge(nt,(elbo = elbo, ))), deepcopy(get_hyperparams(spl, mdl))...
end


get_additional_args_from_spl(spl::VIOptimiser, mdl::AbstractHPModel) = (Mu1 = spl.state.Mu1, Mu2 = spl.state.Mu2)



"""

We need to overwrite this!

"""

function get_additional_args_from_vi(vi::VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:HybridVIAlgorithm})
	Z_v = vi.metadata.Z_Ind.vals
	dims = hpm.inputs.lsspec[1]+size(hpm.inputs[:U])[1]
	Z_Ind = reshape(Z_v, (dims,:))

    return (Z_Ind = Z_Ind,)
end



"""
	get_mu_init(vi::DynamicPPL.VarInfo{NT}, hpm::DefaultHPModel, X::AbstractArray)

## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.


Computes the initial Mu1, Mu2 for the HybridVI approximation. This is the formula evaluated without any latent states X.

"""


function get_mu_init(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:HybridVIAlgorithm})
	k_args = get_kernel_args(vi, hpm.spec)
	m_args = get_mean_args(vi, hpm.spec)
	kernels = get_kernels(hpm.gpinstance, k_args)
	mean_fs = get_means(hpm.gpinstance, m_args)
	Qn = @varname Q
	Q = vi[Qn]
	Z_Ind_n = @varname Z_Ind
	Z_v = vi[Z_Ind_n]
	dims = hpm.inputs.lsspec[1]+size(hpm.inputs[:U])[1]
	Z_Ind = reshape(Z_v, (dims, :))

	Mu1 = zeros(get_latent_type(),size(Z_Ind)[2], hpm.inputs.lsspec[1])
	Mu2 = zeros(get_latent_type(),size(Z_Ind)[2], size(Z_Ind)[2], hpm.inputs.lsspec[1])

	# This is simply the prior!!
	@views for k = 1:hpm.inputs.lsspec[1]
		Cholz = cholesky(Kxx(hpm.gpinstance, kernels[k], Z_Ind)).U
		Mu1[:,k] = Kinv_A(Cholz, mx(hpm.gpinstance, mean_fs[k], Z_Ind))
		Mu2[:,:,k] = CholInv(Cholz)
	end

	Mu2 =Mu2 .* -0.5
	return Mu1,Mu2
end


"""
	compute_Mus(vi::DynamicPPL.VarInfo{NT}, hpm::DefaultHPModel{<:HybridVIAlgorithm}, X::AbstractArray)

## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.
	- Xin::AbstractArray: 						The latent states. Either dim x T+1 x 1 xn_chains or dim x T+1 x n_particles x n_chains


Compute the new values Mu1, Mu2 given the sampled states according to Frigolas algorithm.


"""



function compute_Mus(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:HybridVIAlgorithm}, X::AbstractArray)


	# Extract the hyper paramters
	kernels, mean_fs, Q, _, _ =  extract_hyper_params(vi, hpm)

	# Extract the additional args, these are linked! Thas why we can manipulate them!
	Z_Ind = get_additional_args_from_vi(vi,hpm)[:Z_Ind]


	Mu1 = zeros(get_latent_type(),size(Z_Ind)[2], hpm.inputs.lsspec[1])
	Mu2 = zeros(get_latent_type(),size(Z_Ind)[2], size(Z_Ind)[2], hpm.inputs.lsspec[1])
	@views for k = 1:size(X)[1]
		Cholz = cholesky(Kxx(hpm.gpinstance, kernels[k], Z_Ind)).U
		for i = 1:size(X)[4]
			for j = 1:size(X)[3]
				xtu = vcat(X[:,1:end-1, j, i], hpm.inputs[:U][:, 1:end-1, i])
				for h = 1:size(X)[2]-1
					Ktz = Kxxt(hpm.gpinstance, kernels[k],Z_Ind, xtu[:,h:h])
					At1 = get_At(Cholz, Ktz')

					Mu1[:,k] +=   vec(At1' * (X[k:k,h+1,j,i] -  mx(hpm.gpinstance, mean_fs[k],xtu[:,h:h]))) # Scalar product
					Mu2[:,:,k] +=  At1'* At1
				end
			end
		end

		# Average as we take the expectation
		Mu1[:,k] = Mu1[:,k] .* (1.0/(size(X)[4]*size(X)[3]* Q[k]))
		Mu2[:,:,k] = Mu2[:,:,k] .* (1.0/(size(X)[4]*size(X)[3]* Q[k]))

		# Add mz^T Kz^-1
		Mu1[:,k] += vec(Kinv_A(Cholz, mx(hpm.gpinstance, mean_fs[k],Z_Ind)))
		# Add Kzz^{-1}
		Mu2[:,:,k] += CholInv(Cholz)
	end


	Mu2 = Mu2 .* -0.5
	return Mu1, Mu2
end







"""

Compute the gradient and the value of the ELBO with respect to the trainable Hyper Parameters.

"""
# We want to avoid the gradients.
@nograd get_Σ_μ_VI_ng(Mu1, Mu2) = get_Σ_μ_VI(Mu1,Mu2)

function get_lpf(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:HybridVIAlgorithm}, space::Val, X::AbstractMatrix, Y::AbstractMatrix, U::AbstractMatrix; Mu1::AbstractMatrix{<:Real}, Mu2::AbstractArray{<:Real,3})
	lpf = function(θ::AbstractVector{<:Real})
		lp = 0.0
		@views begin
			new_vi = NewVarInfo(vi, space, θ) # Create new VI with trainable Variables θ

			# Extract the hyper paramters.
			kernels, mean_fs, Q,d0_args, dy_args =  extract_hyper_params(new_vi, hpm)
			# Extract the additional args, these are linked! Thas why we can manipulate them!
			Z_Ind = get_additional_args_from_vi(new_vi,hpm)[:Z_Ind]



			# The GP prior of the Inducing Points.
			Σup = map((dim) ->cholesky(Kxx(hpm.gpinstance, kernels[dim],Z_Ind)).U, 1:hpm.inputs.lsspec[1])
			μup = map((dim) ->mx(hpm.gpinstance, mean_fs[dim],Z_Ind), 1:hpm.inputs.lsspec[1])

			# Mean and cholesky of the covariance of the inducing points
			μuq, Σuq = get_Σ_μ_VI_ng(Mu1, Mu2)

			# KL Divergence

			# http://mlg.eng.cam.ac.uk/teaching/4f13/1920/gaussian%20and%20matrix%20equations.pdf
			for i = 1:size(X)[1]
				lp -= KL(μuq[i], μup[i], Σuq[i], Σup[i])
			end



			# log p(x0)
			distx0, xargs = get_distx0_and_args(hpm.gpinstance, d0_args...)
			xfull = nograd_get_index(X, (:,1))
	    	lp += logpdf(distx0(xargs...), xfull)



			# Compute the choleksy decomposition of Kzz
			# Transitions
			lptrans = 0.0
			for k = 1:size(X)[1]
				kernels_k = kernels[k]
				means_k = mean_fs[k]
				Q_k = Q[k]
				Sigma_q_k = Σuq[k]
				mu_q_k = μuq[k]

				Cholz = cholesky(Kxx(hpm.gpinstance, kernels_k, Z_Ind)).U


				for h = 1:size(X)[2]-1
					xtu = vcat(nograd_get_index(X, (:,h:h)), nograd_get_index(U,  (:,h:h)))

					Ktz = Kxxt(hpm.gpinstance, kernels_k, Z_Ind, xtu)
					Ktt = Kxx(hpm.gpinstance, kernels_k, xtu)

					# -1/2 * Q^{-1} trace( Bt-1 + At-1 Σ At-1^T
					qterm = QInv_B_A_Σ_A(Sigma_q_k, Cholz, Ktz', Ktt, Q_k)
					lptrans +=  qterm
					At1 = get_At(Cholz, Ktz')
					# mt-1 + At_1  μ
					mpred = mx(hpm.gpinstance, means_k, xtu)+ At1 *mu_q_k
					# log N(mt-1 + At-1 μ, Q)
					xtp1 = nograd_get_index(X,(k:k, h+1))
					lptrans = lptrans + logpdf(Zygote_MvNormal(mpred, ones(1,1)*Q_k), xtp1 )
				end
			end

			lp += lptrans

			# Observations
			lpobs = 0.0

			# log p(y_t| x_t)
			for h = 1:size(X)[2]-1
				xt = nograd_get_index(X, (:, h+1))
				yt = nograd_get_index(Y , (:, h))
				distxy, yargs = get_distxy_and_args(hpm.gpinstance, xt, dy_args...)
				# Now we have to compute the ancestor weights for the next observation...
				lpobs = lpobs + logpdf(distxy(yargs...),yt )
			end
			lp += lpobs
		end
		return lp
	end
end
