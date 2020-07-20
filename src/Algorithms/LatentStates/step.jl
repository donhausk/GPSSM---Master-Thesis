

####

# In order to verify the algorithm it is important to view x_{0:T} as x_{1:T+1} and i = t
# Every iteration, we sample t+1

###


"""
	step!( alg, n_particles, hyperparams, Y, U, lssize; kwargs...)

## Arguments

	- alg::AbstractLSAlgorithm:		The Latent States Algorithm
	- n_particles::Integer:			The amount of particles
	- hyperparams::HyperParams:		The Hyper Parameter struct
	- Y::AbstractMatrix:			The observations, which is a Matrix! We only sample one sequence in the step function.
	- U::AbstractMatrix:			The controls, which is a Matrix! We only sample one sequence in the step function.
	- lssize::Tuple:				The dimension of the Timeseries (dim, T, n_sequence)
	- kwargs...						Some additional arguments such as the inducing points or the reference trajectory


"""

function AbstractMCMC.step!(
	alg::BaseLSAlgorithm,
	n_particles::Integer,
	hyperparams::HyperParams,
	Y::AbstractMatrix,
	U::AbstractMatrix,
	lssize::Tuple;
	kwargs...
)

	#This is an instance storing the kernels, mean functions cholesky decompositions etc..
	pf = ParticleFilterInstance(alg, n_particles, Y, U, lssize, hyperparams; kwargs...)
	# Our samples
	X = Array{get_latent_type(),3}(undef,lssize[1], lssize[2]+1, n_particles)
	As = Array{Int,2}(undef, n_particles, lssize[2]+1)
	As[:,1] = 1:n_particles

	# This is solely used as a statistics!
	Trackweights = zeros(2, n_particles, lssize[2])
	# The log pdfs which we need
	lpfs = LogPdfs(n_particles)


	# Perform an initial step
	indices = initial_step!(alg, pf, X, As, n_particles, lpfs, Y, U, lssize, hyperparams)


	#Extract trajectories
	Xtraj = get_traj(As, X, 1)

	# Initialise the storage and the log pdfs.
	init_pfinstance!(alg, pf, Xtraj, n_particles, Y, U, lssize, hyperparams, lpfs)
	init_lpfs!(alg, pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs )


	@views @inbounds for i =1:lssize[2]
		# Get the tranistion distributions for p(x_t|x_{1:t}^p) = N(μs, Σs)
		# Xtraj are the trajectories up to i = t
		μt, Σt = get_prior_transition!(alg, pf, Xtraj, n_particles, Y, U, lssize, hyperparams, i)

		# Update the log pdfs at the start using the transition densities.
		update_start_lpfs!(alg, pf, Xtraj, μt, Σt, n_particles,Y, U, lssize, hyperparams, lpfs, i )

		# Note that resampling is not done independently from sampling the next particles but jointly!
		# In this step, we automatically set X[:,i+1,:] and return the ancestor indices
		indices, tw = propagate_particles!(alg, pf, μt, Σt,  X, n_particles, Y, U, lssize, hyperparams, lpfs, i )

		Trackweights[:,:,i] = tw'


		# Reorder the indices, this is important for the copy cholesky step. This automatically changes the indices!!!
		mvs = reorder_indicies!(indices)
		# we need to adapt to the changes!
		X[:,i+1,:] = X[:,i+1, mvs]
		reorderlpfs!(lpfs, mvs)
		# We need to restructure the pf instace based on the sampled states.
		reorderstorage!(alg, pf, lssize, indices, i)

		# Update the ancestors
	    set_as!(As, indices, i)

		# Get new trajectories
		Xtraj =  get_traj(As, X, i+1)



		# Make a step for the pf instance, i.e. compute new cholesky decompositions etc.
		step_pfinstance!(alg, pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

		# Compute the new log pdfs!
		update_end_lpfs!(alg, pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )
	end

	Xtraj =  get_traj(As, X, lssize[2]+1)
	Xr, Ws = finalise_pf(alg, pf, Xtraj, lpfs)
	return Xr, Ws, (logpdf = lpfs.marginallp,), (X, As), Trackweights# The last return argument is solely used for statistics

end





"""
    initial_step!(alg, pf, X, As,n_particles, LogPdfs, Y, U, lssize, hyperparams)

## Arguments
    - alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
    - pf::AbstractPFInstance    AbstractPFInstance
    - X::AbstractArray              The latent states
    - As::AbstractArray             The ancestor indicies
    - n_particles::Int              The amount of particles
    - lpfs::LogPdfs                 LogPdfs instance
  	- Y::AbstractMatrix:			Observations
  	- U::AbstractMatrix:			Control inputs
  	- lssize::Tuple:				Sizes of the particle filter
  	- hyperparams::HyperParams:		An instance of the hyper parameters.

Initial step

"""
function initial_step!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, X::AbstractArray, As::AbstractArray ,
  n_particles::Int, lpfs::LogPdfs, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams)
  error("This function must be specified")
end



"""
    get_transition_dist!(alg, pf, X, As,n_particles, LogPdfs, Y, U, lssize, hyperparams)

## Arguments
    - alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
    - pf::AbstractPFInstance    AbstractPFInstance
    - Xtraj::AbstractArray          Trajectories
    - n_particles::Int              The amount of particles
    - Y::AbstractMatrix:			Observations
    - U::AbstractMatrix:			Control inputs
  	- lssize::Tuple:				Sizes of the particle filter
  	- hyperparams::HyperParams:	   	An instance of the hyper parameters.
    - i::Integer:                   i = t iteration

Return the transition under the GP prior

"""
function get_prior_transition!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, Xtraj::AbstractArray,
    n_particles::Int, Y::AbstractArray, U::AbstractArray,  lssize::Tuple, hyperparams::HyperParams, i::Integer)
    error("This function must be specified")
end



"""
    get_transition_dist!(alg, pf, X, As,n_particles, LogPdfs, Y, U, lssize, hyperparams)

## Arguments
    - alg::AbstractLSAlgorithm:     An instance of the LS Algorithm
    - pf::AbstractPFInstance    AbstractPFInstance
    - μt::AbstractArray             The means of the prior  xt = N(μt, Σt), Structure of μt: dimension x particles
    - Σt::AbstractArray             The covariances of the prior xt = N(μt, Σt), Structure of Σt: dimension x particles
    - Xtraj::AbstractArray          Trajectories
    - n_particles::Int              The amount of particles
  	- Y::AbstractMatrix:			      Observations
  	- U::AbstractMatrix:			      Control inputs
  	- lssize::Tuple:				        Sizes of the particle filter
  	- hyperparams::HyperParams:	   	An instance of the hyper parameters.
    - i::Integer:                   i = t iteration
    - lpfs::LogPdfs                 LogPdfs instance

Propagate the particles.

"""
function propagate_particles!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, μt::AbstractArray, Σt::AbstractArray,  X::AbstractArray,
   n_particles::Int, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )
   error("This function must be specified!")
end



"""
	init_lpfs!(pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			AbstractPFInstance
	- Xtraj:		Trajectories
	- n_particles:	Amount of particles
	- Y:			Observations
	- U:			Controls
	- lssize:		Sizes of the Timeseries (dim, T, nsubsequences)
	- hyperparams:	Hyper parameter struct
	- lpfs:			Log pdfs
	- i:			Iteration i= t-1


Initial step for the log pdfs
"""


function init_lpfs!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs )
	#nothing
end


"""
	update_start_lpfs!(pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			AbstractPFInstance
	- Xtraj:		Trajectories
	- μ:			The prior means
	- Σ:			The prior transitions
	- n_particles:	Amount of particles
	- Y:			Observations
	- U:			Controls
	- lssize:		Sizes of the Timeseries (dim, T, nsubsequences)
	- hyperparams:	Hyper parameter struct
	- lpfs:			Log pdfs
	- i:			Iteration i= t-1


Updates the log pdfs at the begining of the iteration!

"""


function update_start_lpfs!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, Xtraj::AbstractArray, μ::AbstractMatrix, Σ::AbstractMatrix, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )
	# nothing - usually we do nothing
end


"""

In the Markovian case, we can compute the joing log pdf as follows:

"""


function update_start_lpfs!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, Xtraj::AbstractArray, μ::AbstractMatrix, Σ::AbstractMatrix, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {I<: MarkovianInferenceModel, A<: AbstractLSPGAlgorithm}

	# Reset weights
	lpfs.Jointlp[:] = lpfs.Jointlp[:] .* 0.0
	for p = 1:n_particles
		# The transition probability.
		lpfs.Jointlp[p] += logpdf(MvNormal(μ[:,p], Diagonal(Σ[:,p])), pf.xref[:,i+1])
	end
	# This is the correct way to compute the ancestor weights!!
	lpfs.Jointlp[:] = lpfs.Jointlp[:] + lpfs.Sequentiallp[:]
end




"""
	update_end_lpfs!(pf, Xtraj, n_particles,Y, U, lssize, hyperparams, lpfs, i )

## Arguments
	- alg:			AbstractLSAlgorithm
	- pf:			AbstractPFInstance
	- Xtraj:		Trajectories
	- n_particles:	Amount of particles
	- Y:			Observations
	- U:			Controls
	- lssize:		Sizes of the Timeseries (dim, T, nsubsequences)
	- hyperparams:	Hyper parameter struct
	- lpfs:			Log pdfs
	- i:			Iteration i= t-1


Updates the log pdfs at the end of the iteration! By default, this is the standard way of how to compute the weights!

"""
function update_end_lpfs!(alg::BaseLSAlgorithm, pf::AbstractPFInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )
	# nothing
end


function update_end_lpfs!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {I<:InferenceModel, A<:Union{SMC, PGAS}}
	# Standard way of how to compute the log pdfs for the SMC and PGAS algorithms
	# Compute Weights = p(y_t+1/xt+1) where y_t+1 = Y[:,t]
	@views begin
		# Reset weights
		lpfs.Weightlp[:] =  lpfs.Weightlp[:]  .* 0.0
		for p = 1:n_particles
			xt = Xtraj[:,i+1,p]

			distxy, yargs = get_distxy_and_args(pf.hyperparams.gpinstance, vec(xt), pf.hyperparams.args_θy...)
			# Now we have to compute the ancestor weights for the next observation...
			logpobs = logpdf(distxy(yargs...), Y[:, i])::Float64
			lpfs.Weightlp[p] += logpobs
		end
		lpfs.marginallp += StatsFuns.logsumexp(lpfs.Weightlp) - log(n_particles)
	end


end


"""

  function finalise_pf(pf, Xtraj, lpfs)

## Arguments
    - alg:    Latent states Algorithm
    - pf:     Particle Filter Instance
    - Xtraj:  Trajectories
    - lpfs:   LogPdfs

 # Perform last sampling step if needed.

"""

function finalise_pf(alg::BaseLSAlgorithm, pf::AbstractPFInstance, Xtraj::AbstractArray, lpfs::LogPdfs)
    return Xtraj, softmax!(lpfs.Weightlp)
end

function finalise_pf(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance,  Xtraj::AbstractArray, lpfs::LogPdfs) where {A<:Union{PGAS,SMC}, I<:InferenceModel}
	# We need to resample a last time because we want all sample to have equal weights.
	# This makes the further progress easier.
	idx = resample(softmax!(lpfs.Weightlp), size(Xtraj)[3])
	lpfs.Weightlp[:] .= 0.0
	traj = @view Xtraj[:,:, idx]
    return traj, softmax!(lpfs.Weightlp)
end




"""
	get_traj

Returns a matrix containing the trajectories defined by As.

"""


@inline function get_traj(As::Matrix{I}, Xs::Array{XT,3}, i::Integer) where {I <: Integer, XT <: Real}

    xtraj = Array{Float64,3}(undef ,size(Xs)[1], i, size(Xs)[3])
    @views for iter in 1:i  # Ugly way, may find a better one
        xtraj[:,iter,:] = Xs[:, iter, As[:,iter]]
    end
    return xtraj
end



"""
	set_as!

Extends the trajectories by one steps. Idx are the ancestor trajectories.

"""

@inline function set_as!(As::Matrix{I}, idx::Vector{I}, i::Integer) where I<:Integer
    As_c = deepcopy(As)
    for k = 1:i
        As[:,k] = As_c[idx,k]
    end
    As[:,i+1] = 1:length(idx) # We
end



"""
	reorder_indicies!(indx)
## Arguments
	- indx: Indices

Reorders the indices such that we do not need to copy much.

"""


@inline function reorder_indicies!(indx)
	num_children = Vector{Vector{Int}}([[] for _ in 1:length(indx)])

	for (k,p) in enumerate(indx)
		push!(num_children[p], k)
	end
	already_set = []
	moves = Vector{Int}(undef,length(indx)) # Important to track move of the partcles
	# It does not matter which your ancestors are.
	for (k,v) in enumerate(num_children)
		if length(v) > 0
			indx[k] = k # We have a child here
			moves[k] = v[1]
			push!(already_set,k)
		end
	end
	counter = 1
	for (k,v) in enumerate(num_children)
		while length(v) >1
			@assert counter <= length(indx) "Something went wrong with the restructuring of the indicies"
			if !(counter in already_set)
				indx[counter] = k
				moves[counter] = v[2]
				v = v[2:end] # we can do this!
			end
		 	counter +=1
		end
	end
	return moves # These are important for the MCMC-APF method
end









"""
	get_logps(μ::M, Σ::S, xt)

## Arguments
	-	μ:	Matrix of means
	-	Σ:	Matrix or Vector of Variances
	-	xt:	Vector with respect to which the log pdf should be computed.

Compute the Logpdf of the Normal Distribution for μ and Σ Vectors.


"""

@inline function get_logps(μ::AbstractMatrix, Σ::AbstractMatrix, xt::AbstractVector)
	[logpdf(MvNormal(μ[:,i], Diagonal(Σ[:,i])),xt) for i = 1:size(μ)[2]]
end
@inline function get_logps(μ::AbstractMatrix, Σ::AbstractVector, xt::AbstractVector)
	[logpdf(MvNormal(μ[:,i], Diagonal(Σ)),xt) for i = 1:size(μ)[2]]
end


"""
	get_pyxt1(em, mu, sigma, y_t)
## Arguments
	- mcmc_trans::GaussianEM:	An instance of a Gaussian Emission type
	- mu::AbstractVector		Mean of p(x_t| x_{0:t-1})
	- sigma::AbstractMatrix		Variance of  p(x_t| x_{0:t-1})
	- y_t::AbstractVector		The observation at time t

Computes logpdf of p(y_t| x_{0:t-1}), where p(y_t| x_{0:t-1}) = N(y_t; A_t μ_t + b_t, A_t^T Σ_t A_t + S).

"""

function get_pyxt1(em::GaussianEM, mu::AbstractMatrix, sigma::AbstractMatrix, y_t::AbstractVector)
	n_particles = size(mu)[2]
 	return ([logpdf(MvNormal(em.A * mu[:,p] +em.b ,Symmetric(em.S + em.A * Diagonal(sigma[:,p])* (em.A'))), y_t)  for p =1:n_particles])
end




"""
	get_fa_pred(em, μt, Σt, n_particles)

## Arguments
	- emission::GaussianEM:	An instance of a Gaussian Emission type
	- μt::AbstractMatrix		AbstractArray of p(x_t| x_{0:t-1})
	- Σt::AbstractMatrix		AbstractArray of  p(x_t| x_{0:t-1})
	- y_t::AbstractVector		The observation at time t

Computes logpdf of p(x_t|y_t, x_{0:t-1}) following  Appendix B1 in  https://creates.au.dk/fileadmin/site_files/filer_oekonomi/subsites/creates/Seminar_Papers/2010/APF_Jectx_2010.pdf
Recall that  pf.emission.C  = inv(pf.emission.S)
"""


function get_fa_pred(emission::GaussianEM, μt::AbstractMatrix, Σt::AbstractMatrix, y_t::AbstractVector)
	n_particles = size(μt)[2]

	Sstar = [Symmetric(inv(Diagonal(Symmetric((1 ./Σt[:,p]) .+ Symmetric(emission.A' * emission.C * emission.A))))) for p =1:n_particles]
	At = [Sstar[p]* Diagonal(1 ./Σt[:,p]) for p = 1:n_particles]
	bt = [Sstar[p] * emission.A'*emission.C*(y_t .- emission.b) for p = 1:n_particles]
	μtem = [At[p] * μt[:,p] + bt[p] for p = 1:n_particles]
	return μtem, Sstar
end
