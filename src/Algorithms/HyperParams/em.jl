

"""
 	EMAlgorithm(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule ,schedulerate)

## Arguments
    - get_gamma::Function = (i) -> 1/i:             The γ_i used to update Q
    - optimiser= Flux.Descent(0.1):                 The optimiser, either a Flux or a Optim optimiser
    - lag::Int = -1:                                The lag of the EM algorithm (hence we only consider the lag -last steps for the computation of Q)
    - burnin::Int = 0:                              The amount of burnin steps until the optimiser starts.
    - optimization_type::String = "gradient":       The Optimisation type, either "gardient" or "maximisation"
    - lrschedule::Int = 0:                          The lrschduler for the gradient maximisation, lrschedulue is the frequence the lr rate is multiplied by schedulerate
    - schedulerate::Float64 = 1.0:                  The schedulerate, hence, lrnew = lr * schedulerate

The standard EM algorithm, where either a full maximisation step is performed when optimization_type
is set to "maximisation" or only a sinlge greadient step when set to  "gradient".

Original paper: Identification of Gaussian Process State-Space Models with Particle Stochastic Approximation EM (Frigola, 2014)

"""



struct EMAlgorithm <: AbstractEMAlgorithm
    get_gamma::Function # Gamma for computing the surrogate Q function
    optimiser # Either a Flux optimiser like ADAM or a Optim optimiser like (L-)BFGS
    lag::Int # How many of the states do we use for the opitization. -1 Means all.
    burnin::Int # How many burin in samples are we takeing.
    optimization_type::String # Optimization Type, "gradient" or "maximisation"
    lrschedule::Int # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
    schdulerate::Float64 # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
    # function EMAlgorithm(get_gamma::Function, optimiser, lag::Int, burnin::Int, optimization_type::String, lrschedule::Int, schdulerate::Float64)
    #     new{FullGPInference}(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule, schdulerate)
    # end
end
EMAlgorithm(;get_gamma::Function = (i) -> 1/i, optimiser= Flux.Descent(0.1), lag::Int = -1, burnin::Int = 0, optimization_type::String = "gradient", lrschedule::Int = 0, schedulerate::Float64 = 1.0)= EMAlgorithm(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule ,schedulerate) # No scheduling

get_inferencemodel(::EMAlgorithm) = FullGPInference()




"""
 	SparseEMAlgorithm(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule ,schedulerate)

## Arguments
    - get_gamma::Function = (i) -> 1/i:             The γ_i used to update Q
    - optimiser= Flux.Descent(0.1):                 The optimiser, either a Flux or a Optim optimiser
    - lag::Int = -1:                                The lag of the EM algorithm (hence we only consider the lag -last steps for the computation of Q)
    - burnin::Int = 0:                              The amount of burnin steps until the optimiser starts.
    - optimization_type::String = "gradient":       The Optimisation type, either "gardient" or "maximisation"
    - lrschedule::Int = 0:                          The lrschduler for the gradient maximisation, lrschedulue is the frequence the lr rate is multiplied by schedulerate
    - schedulerate::Float64 = 1.0:                  The schedulerate, hence, lrnew = lr * schedulerate


The standard EM algorithm, where either a full maximisation step is performed when optimization_type
is set to "maximisation" or only a sinlge greadient step when set to  "gradient".

No paper so far about this algorithm
"""

struct SparseEMAlgorithm{OIND} <: AbstractEMAlgorithm
    get_gamma::Function # Gamma for computing the surrogate Q function.
    optimiser # Either a Flux optimiser like ADAM or a Optim optimiser like (L-)BFGS.
    lag::Int # How many of the states do we use for the opitization. -1 Means all.
    burnin::Int # How many burin in samples are we takeing.
    optimization_type # Optimization Type, "gradient" or "maximisation".
    lrschedule::Int # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
    schdulerate::Float64  # Lr scheduling for Gradient maximisation. In case of full maximisation, this is neglected.
	optimiser_ind::OIND
end
SparseEMAlgorithm(;get_gamma::Function = (i) -> 1/i, optimiser_ind = Flux.Descent(0.05), optimiser= Flux.ADAM(0.1), lag::Int = -1, burnin::Int = 0, optimization_type::String = "gradient", lrschedule::Int = 0, schedulerate::Float64 = 1.0)= SparseEMAlgorithm(get_gamma, optimiser, lag, burnin, optimization_type, lrschedule ,schedulerate, optimiser_ind) # No scheduling

get_inferencemodel(::SparseEMAlgorithm) = SparseInference()







"""
	EMState

## Fields
	- vi::VarInfo
	- diffs:Vector{Functions}:	A vector of functions which compute the gradient and the value of p(x_{0:T}^k, y_{1:T}|θ)
	- Ws::Vector{Float64}:		A vector containing the weights of the ELBO algortihm

# Stores a VarInfo instance containing all the hyper parameters. This struct is mutuble

"""

mutable struct EMState{V<: DynamicPPL.VarInfo{<:NamedTuple}}
	vi::V
	diffs::AbstractVector
	Ws::AbstractVector{<:Real}
end


"""

	EMState(alg,spec,inputs)

## Arguments
	- alg::AbstractHPEMOptimiser:				The EM algorithm
	- spec::GPSpec:								The GPSpec struct
	- inputs::NamedTuple:						The inputs of the gp
Returns the specific state

"""

EMState(alg::AbstractEMAlgorithm, spec::GPSpec, inputs::NamedTuple) =  EMState(generate_vi(spec),  [], Vector{typeof(alg.get_gamma(1))}())
function EMState(alg::SparseEMAlgorithm, spec::GPSpec, inputs::NamedTuple)
	keysmu = [Symbol("μ_$i") for i = 1:inputs[:sizes][1]]
	keyssigma = [Symbol("Σ_$i") for i = 1:inputs[:sizes][1]]
	valsmu = [zeros( size(spec.Z_Ind)[2]) for _ in 1:inputs[:sizes][1]]
	valssigma = [UpperTriangular(Matrix(Diagonal(( ones(size(spec.Z_Ind)[2]))))) for _ in 1:inputs[:sizes][1]]
	trainmu = [ alg.optimiser_ind == nothing ?  "T" : "F" for _ in 1:inputs[:sizes][1]]
	trainsigma = [ alg.optimiser_ind == nothing ?  "TP" : "F" for _ in 1:inputs[:sizes][1]] # We store here the choleksy decomposition and hence do not need to make it "TP"
	vi = generate_vi(spec, additional_params =  NamedTuple{Tuple(vcat(keysmu, keyssigma))}(Tuple(vcat(valsmu, valssigma))), train_additional_params =NamedTuple{Tuple(vcat(keysmu, keyssigma))}(Tuple(vcat(trainmu, trainsigma))))
	EMState(vi, [],  Vector{typeof(alg.get_gamma(1))}() )
end



"""
	EMOptimiser

## Arguments
	- alg<:EMAlgorithm: 		The EMalgorithm instance
	- state::EMState			A state instance
	- inputs::NamedTuple		The inputs of the sampler (controls, observations, sizes)


"""


struct EMOptimiser{ALG, S} <:AbstractHPEMOptimiser where {ALG <:AbstractEMAlgorithm, S <: EMState}
	alg::ALG
	state::S
	inputs::NamedTuple
end

function Sampler(mdl::AbstractHPModel, alg::ALG, inputs::NamedTuple) where ALG <: AbstractEMAlgorithm
	state =  EMState(alg, mdl.spec, inputs)
	EMOptimiser{ALG, typeof(state)}(alg, state ,inputs)
end

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::AbstractHPModel,
    spl::EMOptimiser,
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)

	return get_hyperparams(spl, model)
end


function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::AbstractHPModel,
    spl::EMOptimiser{<:SparseEMAlgorithm},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)

	set_sparse_pior!(spl.state.vi, model)
	return get_hyperparams(spl, model)

end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    mdl::AbstractHPModel,
    spl::EMOptimiser,
    N::Integer,
    states::NT;
	iteration,
    kwargs...
) where {NT<:NamedTuple}

	# Lr scheduling if we use gradient descent
	if spl.alg.lrschedule >0 && spl.alg.optimization_type == "gradient"
		if iteration%spl.alg.lrschedule == 0
			spl.alg.optimiser.eta *= spl.alg.schdulerate
		end
	end
	#@views if iteration > spl.alg.burnin

	update_inducingpoints(spl.alg, spl.state.vi, mdl, states[:X], iteration)
	#end

	# We optimize on the linked hyper parameters to take care of positive definiteness!
	link!(spl.state.vi, TrainableSelector)


	# get the diffs functions
	Fs, Ws = update_diffs!(spl, mdl, states[:X], iteration)

	# Extract trainable hyper parameters.
	θ = spl.state.vi[TrainableSelector]
	if length(θ) > 0
		if spl.alg.optimization_type == "gradient"
			# We do this only if we are after the burnin phase
			if iteration > spl.alg.burnin
				_, grad = get_grad_and_lpf(Fs, θ; Ws = Ws)
				new_θ =  Flux.Optimise.update!(spl.alg.optimiser, θ, -clip_gradient(grad, DEFAULT_GRADIENT_CLIPPING)) # We use gradient ascent...
				spl.state.vi[TrainableSelector] = new_θ # Only update the TrainableSelector hyper parameters.
			end

		elseif  spl.alg.optimization_type == "maximisation"
			# We make a full optimization step
			if iteration >spl.alg.burnin
				# Define functions for optimiser.
				fun = (θm) -> begin # Computes the surrogateQ
					surrogateQ, _ = get_grad_and_lpf(Fs, θm; Ws = Ws, compute_grad = false)
					# We use a minimizer therefore, we have to take the -
					return -1 .* surrogateQ
				end
				# Computes the graident ∇surrogateQ
				gfun = (θm) -> begin
					_, grad = get_grad_and_lpf(Fs, θm; Ws = Ws)
					# We use a minimizer therefore, we have to take the -
					return  -1 .* grad
				end

				#  Run the optimiser, where we have again a minimizer
				#  Note that spl.alg.optmizer is an optmizer from the optim package.
				res = optimize(fun, gfun, convert.(Float64, θ), spl.alg.optimiser; inplace=false) # We need Float64
				spl.state.vi[TrainableSelector] = res.minimizer
			end
		else
			error("The optimisation type of the EM sampler must be either maximisation or gradient!")
		end

		# Compute the new surrogateQ for tracking!
		surrogateQ, _ = get_grad_and_lpf(Fs,spl.state.vi[TrainableSelector]; Ws = Ws, compute_grad = false)
	else
		# We do not train anything here....
		surrogateQ =  0.0
	end
	# Invlink again.
	invlink!(spl.state.vi, TrainableSelector)
	nt = dropnames(tonamedtuple(spl.state.vi), Tuple([Symbol("μ_$i") for i = 1:spl.inputs[:sizes][1]]))
	nt = dropnames(nt, Tuple([Symbol("Σ_$i") for i = 1:spl.inputs[:sizes][1]]))
	nt = dropnames(nt, (:Z_Ind, ))
	return ConstTransition(merge(nt , (surrogateQ = surrogateQ,))), deepcopy(get_hyperparams(spl, mdl))...
end

update_inducingpoints(alg::AbstractEMAlgorithm,vi::DynamicPPL.VarInfo, mdl::AbstractModel, X::AbstractArray, iteration::Integer) = vi





###

# The full log pdf. This returns log p(x_{0:T}, y_{1:T}| θ)

###


function get_lpf(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:EMAlgorithm}, space::Val, X::AbstractMatrix, Y::AbstractMatrix, U::AbstractMatrix)
	lpf = function(θ::AbstractVector{<:Real})
		lp = 0.0
		@views begin
			# The new Var Vinfo ( Similar to Turings HMC implementaton.)
			new_vi = NewVarInfo(vi, space, θ)

			# Extract the hyper paramters.
			kernels, mean_fs, Q,d0_args, dy_args =  extract_hyper_params(new_vi, hpm)

			# The identity matrix of size TxT
			Id = get_identity_mat(hpm.inputs.lsspec[2])


			# This is p(x_0)!
		    distx0, xargs = get_distx0_and_args(hpm.gpinstance, d0_args...)
			# Get the first elementss
			x0 = nograd_get_index(X, (:,1,))
	        lp += logpdf(distx0(xargs...), x0)


			# Now log p(x_{1:T}| x_0).
			# Attention, we use a trick here, namely that X has the pointwise logpdf like the Multivariate Normal distribution,
			# with K = K_{0:T-1} + I *Q
			logpjoint = 0.0
			# These are x_{0:T-1} concatenated with U
			xandu = vcat(nograd_get_index(X, (:,1:hpm.inputs.lsspec[2])), nograd_get_index(U,  (:,1:hpm.inputs.lsspec[2])))
			for i in 1:hpm.inputs.lsspec[1]
				xout = nograd_get_index(X, (i,2:hpm.inputs.lsspec[2]+1))
				logpjoint += logpdf(Zygote_MvNormal(mx(hpm.gpinstance, mean_fs[i], xandu), Symmetric(Kxx(hpm.gpinstance, kernels[i], xandu) .+ Id .* Q[i]) ),xout)
			end

			lp += logpjoint

			# The emission.
			lpem = 0.0
			for i in 1:hpm.inputs.lsspec[2]
				Xem = nograd_get_index(X, (:, i+1))
				# Attention, Y_t = Y[i], X_t = X[i+1] because X also contains x_0
				Yem = nograd_get_index(Y, (:,i))

				distxy, yargs = get_distxy_and_args(hpm.gpinstance, Xem, dy_args...)
				lpem += logpdf(distxy(yargs...),Yem)
			end

			lp += lpem
		end
		return lp
	end
end


####################

# Sparse GP

####################



"""

Returns the inducing points, the means and the cholesky decompositions of the inducing points


"""

function get_additional_args_from_vi(vi::VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:SparseEMAlgorithm})
	Z_v = vi.metadata.Z_Ind.vals
	dims = hpm.inputs.lsspec[1]+size(hpm.inputs[:U])[1]
	Z_Ind = reshape(Z_v, (dims,:))

	Cv = map(i -> UpperTriangular(vi[varname("Σ_$i")]), 1:hpm.inputs.lsspec[1])
	μv = map(i ->  vi[varname("μ_$i")], 1:hpm.inputs.lsspec[1])

	#Cv = map(i -> cholesky(Symmetric(reshape(get_metadata_from_vi(vi,"Σ_", i).vals, (size(Z_Ind)[2], size(Z_Ind)[2])))), 1:hpm.inputs.lsspec[1])
	#μv = map(i ->  get_metadata_from_vi(vi,"μ_", i).vals, 1:hpm.inputs.lsspec[1])

    return (Z_Ind = Z_Ind, mu =μv, Sigma=Cv)
end




"""
	set_sparse_pior!(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:SparseEMAlgorithm})


## Arguments
	- vi::DynamicPPL.VarInfo{<:NamedTuple}:		The varinfo struct containing information on the hyper paramters.
	- hpm::DefaultHPModel:						An instance of a DefaultHPModel.


Sets the initial mean and covariance of the inducing points to be the prior.

"""


function set_sparse_pior!(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:SparseEMAlgorithm})

	# Extract the hyper paramters
	kernels, mean_fs, Q, _, _ =  extract_hyper_params(vi, hpm)

	# Extract the additional args, these are linked! Thas why we can manipulate them!
	Z_Ind, _, _ = get_additional_args_from_vi(vi,hpm)

	# This is simply the prior!!
	@views for k = 1:hpm.inputs.lsspec[1]
		vi.metadata[Symbol("μ_$k")].vals[:] =  mx(hpm.gpinstance, mean_fs[k], Z_Ind)
		vi.metadata[Symbol("Σ_$k")].vals[:] =  vec(cholesky(Symmetric(Kxx(hpm.gpinstance, kernels[k], Z_Ind))).U)
	end
end

## Documentation will follow...

update_inducingpoints(alg::SparseEMAlgorithm{Nothing},vi::DynamicPPL.VarInfo, mdl::AbstractModel, X::AbstractArray, iteration::Integer) = vi

function update_inducingpoints(alg::SparseEMAlgorithm,vi::DynamicPPL.VarInfo, hpm::AbstractModel, XFull::AbstractArray, iteration::Integer)
	# The new Var Vinfo ( Similar to Turings HMC implementaton.)

	# Extract the hyper paramters.
	kernels, mean_fs, Q,d0_args, dy_args =  extract_hyper_params(vi, hpm)
	# Extract the additional args, these are linked! Thas why we can manipulate them!
	Z_Ind, μ, CΣ = get_additional_args_from_vi(vi,hpm)

	# The identity matrix of size TxT
	Id = get_identity_mat(hpm.inputs.lsspec[2])
	# Now log p(x_{1:T}| x_0).
	# Attention, we use a trick here, namely that X has the pointwise logpdf like the Multivariate Normal distribution,
	# with K = K_{0:T-1} + I *Q
	μ_new, Σ_new = [[] for _ in 1:hpm.inputs.lsspec[1]], [[] for _ in 1:hpm.inputs.lsspec[1]]

	@views for i = 1:size(XFull)[3]
		for j = 1:size(XFull)[4]
			X = XFull[:,:,i,j]
			U = hpm.inputs[:U][:,:,j]
			Y = hpm.inputs[:Y][:,:,j]
			# These are x_{0:T-1} concatenated with U
			xandu = vcat(nograd_get_index(X, (:,1:hpm.inputs.lsspec[2])), nograd_get_index(U,  (:,1:hpm.inputs.lsspec[2])))

			# The sparse elbo as lower bound of the joing log p.
			for k in 1:hpm.inputs.lsspec[1]
				# Unfortunately, we need to make these get indices!
				xout = nograd_get_index(X, (k,2:hpm.inputs.lsspec[2]+1))
				# We want to avoid multiple times calling get index
				kernels_k = kernels[k]
				means_k = mean_fs[k]
				Q_k = Q[k]
				# Compute the covariance and mean functions
				Kxz = Kxxt(hpm.gpinstance, kernels_k, xandu, Z_Ind)
				Cholz = cholesky(Kxx(hpm.gpinstance, kernels_k, Z_Ind)).U
				mz = mx(hpm.gpinstance, means_k, Z_Ind)
				mxfull = mx(hpm.gpinstance, means_k, xandu)
				# Compute the elbo

				μ_ik, Σ_ik =  sparseopt(Cholz, Kxz,  Q_k, xout, mxfull, mz)
				push!(μ_new[k], μ_ik)
				push!(Σ_new[k], Σ_ik)
			end
		end
	end


	@views for k in 1:hpm.inputs.lsspec[1]
		μ_k = mean(μ_new[k])
		Σ_k= Symmetric(mean(Σ_new[k]))

		if iteration == 1
			μ_n =  μ_k
			CΣ_n = cholesky(Σ_k).U
		else
			μ_n = Flux.Optimise.update!(alg.optimiser_ind, μ[k],μ[k] - μ_k )
			Sigmak = CΣ[k]' * CΣ[k]
			CΣ_n = cholesky(Symmetric(Flux.Optimise.update!(alg.optimiser_ind, Sigmak , Sigmak - Σ_k ))).U
		end
		vi.metadata[Symbol("Σ_$k")].vals[:] = vec(CΣ_n)
		vi.metadata[Symbol("μ_$k")].vals[:] = vec(μ_n)
	end

	return vi
end


function get_lpf(vi::DynamicPPL.VarInfo{<:NamedTuple}, hpm::DefaultHPModel{<:SparseEMAlgorithm}, space::Val, X::AbstractMatrix, Y::AbstractMatrix, U::AbstractMatrix; kwargs...)
	lpf = function(θ::AbstractVector{<:Real})
		lp = 0.0
		@views begin
			# The new Var Vinfo ( Similar to Turings HMC implementaton.)
			new_vi = NewVarInfo(vi, space, θ)

			# Extract the hyper paramters.
			kernels, mean_fs, Q,d0_args, dy_args =  extract_hyper_params(new_vi, hpm)
			# Extract the additional args, these are linked! Thas why we can manipulate them!
			Z_Ind, μ, Σ = get_additional_args_from_vi(new_vi,hpm)

			# The identity matrix of size TxT
			Id = get_identity_mat(hpm.inputs.lsspec[2])

		    d0_args = get_distθx0_args(new_vi,hpm.spec)

			# This is log p(x_0)!
		    distx0, xargs = get_distx0_and_args(hpm.gpinstance, d0_args...)
			# Get the first elementss
			x0 = nograd_get_index(X, (:,1,))
			# Add it to the log p
	        lp += logpdf(distx0(xargs...), x0)


			# Now log p(x_{1:T}| x_0).
			# Attention, we use a trick here, namely that X has the pointwise logpdf like the Multivariate Normal distribution,
			# with K = K_{0:T-1} + I *Q
			logpjoint = 0.0
			# These are x_{0:T-1} concatenated with U
			xandu = vcat(nograd_get_index(X, (:,1:hpm.inputs.lsspec[2])), nograd_get_index(U,  (:,1:hpm.inputs.lsspec[2])))

			# The sparse elbo as lower bound of the joing log p.
			for k in 1:hpm.inputs.lsspec[1]
				# Unfortunately, we need to make these get indices!
				xout = nograd_get_index(X, (k,2:hpm.inputs.lsspec[2]+1))
				# We want to avoid multiple times calling get index
				kernels_k = kernels[k]
				means_k = mean_fs[k]
				Q_k = Q[k]
				Sigma_q_k = Σ[k]
				mu_q_k = μ[k]
				# Compute the covariance and mean functions
				DiagKxxe = DiagKxx(hpm.gpinstance, kernels_k, xandu)
				Kxz = Kxxt(hpm.gpinstance, kernels_k, xandu, Z_Ind)
				Cholz = cholesky(Kxx(hpm.gpinstance, kernels_k, Z_Ind)).U
				mz = mx(hpm.gpinstance, means_k, Z_Ind)
				mxfull = mx(hpm.gpinstance, means_k, xandu)
				# Compute the elbo
				logpjoint +=  elbo(mu_q_k, Sigma_q_k, Cholz, DiagKxxe, Kxz,  Q_k, xout, mxfull, mz)
			end
			lp += logpjoint

			# The emission.
			dy_args = get_distθy_args(new_vi, hpm.spec)
			lpem = 0.0
			for i in 1:hpm.inputs.lsspec[2]
				Xem = nograd_get_index(X, (:, i+1))
				# Attention, Y_t = Y[i], X_t = X[i+1] because X also contains x_0
				Yem = nograd_get_index(Y, (:,i))

				distxy, yargs = get_distxy_and_args(hpm.gpinstance, Xem, dy_args...)
				lpem += logpdf(distxy(yargs...),Yem)
			end

			lp += lpem
		end
		return lp
	end
end
