

###

# Shared among all Sparse  GP implementatoins:

###


"""
	 SparseStorage{K, M, MZ, E}

## Fields
	- Cholzz:			The Cholesky decompositions of K_zz
	- CholzAz:			The Cholesky decompositions of Kzz^-1 Σ Kzz^-1
	- mz:				Kzz^{-1} (u - m_z)
	- ZInd:				The inducing points z


"""


struct SparseStorage{K, KA, M, MZ} <: PFStorage
	Cholzz::Vector{K}
	Σ::Vector{KA}
	mz::Vector{M}
	ZInd::MZ
end



function get_storage(alg::BaseLSAlgorithm{I}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, Z_Ind::AbstractMatrix, mu::AbstractVector,Sigma::AbstractVector,  kwargs...) where {I <: SparseInference, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}

	Cholzzut = @views [cholesky(Kxx(hyperparams.gpinstance, hyperparams.kernels[dim], Z_Ind)).U for dim =1:lssize[1]] # This is the cholesky decomposition of Kzz
	Cholzz = Vector{typeof(Cholzzut[1])}(Cholzzut)

	#CholSigma = Sigma
	#CholzAzut = @views [Ainv_K_Ainv(Sigma[k], Cholzz[k]) for k = 1:lssize[1]]
	#CholzAz = Vector{typeof(CholzAzut[1])}(CholzAzut)

	# This is Kzz^{-1} U_Ind !
	mzut = @views [Kinv_A(Cholzz[i], mu[i] .- mx(hyperparams.gpinstance, hyperparams.mean_fs[i], Z_Ind)) for i = 1:lssize[1]]
	mz = Vector{typeof(mzut[1])}(mzut)

	return SparseStorage(Cholzz, Sigma, mz, Z_Ind)
end


function get_prior_transition!(alg::BaseLSAlgorithm{I}, pf::ParticleFilterInstance, Xtraj::AbstractArray,
    n_particles::Int,  Y::AbstractArray, U::AbstractArray,  lssize::Tuple, hyperparams::HyperParams, i::Integer) where {I<:SparseInference}
	mus = zeros(lssize[1], n_particles)
	sigmas = zeros(lssize[1], n_particles)

	@views for p = 1:n_particles
		#last step
		xt = Xtraj[:,i:i,p]
		xtu = reshape(vcat(xt, U[:,i]),(:,1))
		for dim = 1:lssize[1] #ok
			mt1 = mx(pf.hyperparams.gpinstance, pf.hyperparams.mean_fs[dim], xtu)
			Ktz = Kxxt(pf.hyperparams.gpinstance, pf.hyperparams.kernels[dim],pf.storage.ZInd, xtu)
			Kttq = Kxx(pf.hyperparams.gpinstance, pf.hyperparams.kernels[dim], xtu) .+ pf.hyperparams.Q[dim]

			mus[dim:dim, p] = get_transition_mean(I,Ktz, mt1,  pf.storage.mz[dim] )
			sigmas[dim:dim,p:p] = get_transition_cov(I, Kttq, Ktz,  pf.storage.Cholzz[dim], pf.storage.Σ[dim])
		end
	end
	return mus, sigmas
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
