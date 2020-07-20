

###

# Shared among all Sparse  GP implementatoins:

###



"""
	 SparseStorage{K, M, MZ, E}

## Fields
	- Cholzz:			The Cholesky decompositions of K_zz
	- mz:				Kzz^{-1} (u - m_z)
	- ZInd:				The inducing points z


"""

struct HybridVIExtras{K, K2, M, MZ} <: PFStorage
	Cholzz::Vector{K}
	Cholq::Vector{K2}
	mz::Vector{M}
	ZInd::MZ
end



function get_storage(alg::BaseLSAlgorithm{I}, n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, Z_Ind::AbstractMatrix, Mu1::AbstractArray, Mu2::AbstractArray, kwargs...) where {I <: HybridVIInference, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}

	μuq, Σuq = get_Σ_μ_VI(Mu1, Mu2)

	Kzzvut = [cholesky(Kxx(hyperparams.gpinstance, hyperparams.kernels[dim],Z_Ind)).U for dim =1:lssize[1]]
	Cholzz = Vector{typeof(Kzzvut[1])}(Kzzvut)
	Muut = [Kinv_A(Cholzz[dim], μuq[dim]) for dim =1:lssize[1]]
	Mus = Vector{typeof(Muut[1])}(Muut)
	return HybridVIExtras(Cholzz, Σuq, Mus, Z_Ind)
end


function get_prior_transition!(alg::BaseLSAlgorithm{I}, pf::ParticleFilterInstance, Xtraj::AbstractArray,
    n_particles::Int,  Y::AbstractArray, U::AbstractArray,  lssize::Tuple, hyperparams::HyperParams, i::Integer) where {I<:HybridVIInference}

	mus = zeros(lssize[1], n_particles)
	sigmas = zeros(lssize[1], n_particles)

	@views for p = 1:n_particles
		#last step
		xt = Xtraj[:,i:i,p]
		xtu = reshape(vcat(xt, U[:,i]),(:,1))
		for dim = 1:lssize[1] #ok
			mt1 = mx(pf.hyperparams.gpinstance, pf.hyperparams.mean_fs[dim], xtu)
			Ktz = Kxxt(pf.hyperparams.gpinstance, pf.hyperparams.kernels[dim],pf.storage.ZInd, xtu)
			# mx + AT* μ
			mus[dim:dim, p] = get_transition_mean(I,  Ktz, mt1, pf.storage.mz[dim])
			# This is simply Q!
			sigmas[dim,p] = pf.hyperparams.Q[dim]
		end
	end
	return mus, sigmas
end

"""

	We overwrite this as Hybrid VI has a very special emission!

"""

function update_end_lpfs!(alg::BaseLSAlgorithm{<:HybridVIInference, <:Union{PGAS, SMC, MCMCFAAPF}}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )
	# Standard way of how to compute the log pdfs for the SMC and PGAS algorithms
	@views begin
		# Reset weights
		lpfs.Weightlp[:] =  lpfs.Weightlp[:]  .* 0.0
		for p = 1:n_particles

			xt = Xtraj[:,i+1:i+1,p]
			distxy, yargs = get_distxy_and_args(pf.hyperparams.gpinstance, vec(xt), pf.hyperparams.args_θy...)
			# Now we have to compute the ancestor weights for the next observation...
			logpobs = logpdf(distxy(yargs...), Y[:, i])::Float64
			lpfs.Weightlp[p] += logpobs
			if i !==  lssize[2]
				xtandu = vcat(xt, U[:,i+1:i+1])
				for k = 1:lssize[1]
					Ktz = Kxxt(hyperparams.gpinstance, hyperparams.kernels[k], xtandu, pf.storage.ZInd)
					Ktt = Kxx(hyperparams.gpinstance, hyperparams.kernels[k], xtandu)
					lpfs.Weightlp[p] +=  QInv_B_A_Σ_A(pf.storage.Cholq[k], pf.storage.Cholzz[k], Ktz, Ktt, pf.hyperparams.Q[k])
				end
			end
		end
	end
end


function init_lpfs!(alg::BaseLSAlgorithm{<:HybridVIInference}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer
	,Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs )
	@views begin
		for p = 1:n_particles
			xtandu = vcat(Xtraj[:,1:1,p], U[:,1:1])
			for k = 1:lssize[1]
				Ktz = Kxxt(hyperparams.gpinstance, hyperparams.kernels[k], xtandu, pf.storage.ZInd)
				Ktt = Kxx(hyperparams.gpinstance, hyperparams.kernels[k], xtandu)
				lpfs.Weightlp[p] +=  QInv_B_A_Σ_A(pf.storage.Cholq[k], pf.storage.Cholzz[k], Ktz, Ktt, pf.hyperparams.Q[k])
			end
		end
	end
end



#########################################################

### PGAS Implementatoin

##########################################################



##########################################################

### FAAPF Implementation

##########################################################


## There is no FAAPF implementation!





##########################################################

### MCMCAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!

# Paper On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)












##########################################################

### SMC Implementatoin

##########################################################
