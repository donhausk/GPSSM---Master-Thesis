

###

# Shared among all Full GP implementatoins:

###



"""
	FGPStorage{XT}

## Fields
	- Chols:		The Cholesky decompositions of K_{0:t-2}+I*Q
	- CholsPGAS:	The Cholesky decomposition of K_{0:t-1, t:T-ref} +I*Q
	- Ktt_save:		The covariance matrix K_{t-1, 0:t-1}
	- Means_save: 	The means of m_0:t-1}


For the full gp, we need to store the cholesky decompositions!!
"""

struct FGPStorage{XT} <: PFStorage
    Chols::Matrix{UpperTriangular{XT, Array{XT,2}}}
    CholsPGAS::Matrix{UpperTriangular{XT, Array{XT,2}}}
    Ktt_save::Matrix{Matrix{XT}}
    Means_save::Matrix{Vector{XT}}
end

# Initialisation
function FGPStorage(XT::T, n_particles::Integer, lssize::Tuple  ) where {T<:Type{<:Real}}
	# We store the cholesky decompositinos and the computations of the kernel in order to reduce computational demands.
	Chols = Matrix{UpperTriangular{XT, Array{XT,2}}}(undef,lssize[1], n_particles)
	CholsPGAS = Matrix{UpperTriangular{XT, Array{XT,2}}}(undef, lssize[1], n_particles)
	Ktt_save = Matrix{Matrix{XT}}(undef, lssize[1], n_particles)
	Means_save = Matrix{Vector{XT}}(undef, lssize[1], n_particles)
	# We will always reuse them in order to save computation
	for j = 1:n_particles
		for i =1:lssize[1]
			Chols[i,j] = UpperTriangular{XT, Array{XT,2}}(Matrix{XT}(undef, lssize[2],lssize[2]))
			Ktt_save[i,j] = Matrix{XT}(undef, 1, lssize[2]+1)
			Means_save[i,j] = Vector{XT}(undef, lssize[2]+1)
		end
	end

	FGPStorage{XT}(Chols,CholsPGAS, Ktt_save, Means_save)
end


function get_storage(alg::BaseLSAlgorithm{I,A},n_particles::Integer, Y::AbstractMatrix, U::AbstractMatrix, lssize::Tuple,
	hyperparams::HyperParams ;xref::M=nothing,  XT::T=Float64, kwargs...)  where {I<:FullGPInference, A<:AbstractLSAlgorithm, M<: Union{AbstractArray,Nothing}, T<: Type{<:Real}}
	return FGPStorage(XT, n_particles, lssize)
end

# The prior transitions. Note that we update the Cholesky decompositoins directly here. This is not nice but it is easier.

function get_prior_transition!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, Xtraj::AbstractArray,
    n_particles::Int,  Y::AbstractArray, U::AbstractArray,  lssize::Tuple, hyperparams::HyperParams, i::Integer) where  {I<:FullGPInference, A<:AbstractLSAlgorithm}
	μts = zeros(lssize[1], n_particles)
    Sigmas = zeros(lssize[1], n_particles)
    @views for p = 1:n_particles
        for dim = 1:lssize[1]
            el = getindex(pf.storage.Ktt_save,dim, p)[:,1:i]
            Kttq =  el[:, end:end] .+ pf.hyperparams.Q[dim]  # We need to add pf.hyperparams.Q here.
            if i==1
				# for i = 1, this is straigt forward.
                Sigmas[dim,p]=  Kttq[1,1]
                μts[dim,p]= getindex(pf.storage.Means_save, dim, p)[i]
            else
				#Extract the Cholesky decomposition form the storage ( i.e K_{1:t-1} +I*pf.hyperparams.Q) because i = t)
				chol = UpperTriangular(pf.storage.Chols[dim, p][1:i-1,1:i-1] )
				# Compute the conditional mean and variance.
                Ktt0 = getindex(pf.storage.Ktt_save,dim, p)[:, 1:i-1]
				# compute the conditional mean anc covariance
                mt = get_transition_mean(I, getindex(pf.storage.Means_save,dim, p)[i:i], getindex(pf.storage.Means_save,dim, p)[1:i-1], Xtraj[dim, 2:end, p], Ktt0, chol)
                Σt = get_transition_cov(I, Ktt0, Kttq, chol)
                μts[dim, p] = mt[1] # This is one dimensional!
                Sigmas[dim, p] = Σt[1,1] # This is one dimensional!
            end
        end
    end
    return μts, Sigmas
end




# We need to update the cholesky decompositions and the storage!
# Further, we compute the Joint LogPdf here.
function init_pfinstance!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer,
	Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs) where  {I<:FullGPInference, A<:AbstractLSPGAlgorithm}

	@views for i =1:n_particles
		xandufull = hcat(vcat(Xtraj[:,1,i], U[:,1]), vcat(pf.xref[:, 2:end-1],U[:, 2:end-1])) #Full trajectory
		for dim=1:lssize[1]
			# get kernel and mean
			kernel = pf.hyperparams.kernels[dim]
			mean_f = pf.hyperparams.mean_fs[dim]
			# compute Kernel output
			Kttfull = Kxx(pf.hyperparams.gpinstance, kernel, Array(xandufull))
			# Save first computation
			pf.storage.Ktt_save[dim,i][1:1, 1:1] = Kttfull[1:1, 1:1]

			# Compute cholesky decomposition, note taht this has complexity O(T^3), however, this has to be done only once!
			chol = cholesky(Hermitian(Kttfull .+ Diagonal(ones(lssize[2])) .* pf.hyperparams.Q[dim]))
			# Save Cholesky decomposition.
			pf.storage.CholsPGAS[dim,i] = chol.U
			#compute menas and store them
			mfull =  mx(pf.hyperparams.gpinstance, mean_f, xandufull)
			pf.storage.Means_save[dim,i][1:1] = mfull[1:1]
			xfull = pf.xref[dim, 2:end]

			# Compute JointLogPdf for ancestor sampling!
			logjointp = logpdf(Zygote_MvNormal(mfull, chol.U), xfull)
			lpfs.Jointlp[i] =  lpfs.Jointlp[i] + logjointp
		end
	end
end





# We need to update the cholesky decompositions and the storage!
# Further, we compute the Joint LogPdf here.
function step_pfinstance!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer,
	Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where  {I<:FullGPInference, A<:AbstractLSPGAlgorithm}

	lpfs.Jointlp[:] =  lpfs.Jointlp[:]  .* 0.0

	@views for p in 1:n_particles
		#This are computed during every step.

		# xandufull is x_0:t - p th trajectory concatenated with x_t+1:T reference trajector<
        xandufull = hcat(vcat(Xtraj[:,:,p], U[:, 1:i+1]), vcat(pf.xref[:, i+2:end-1],U[:, i+2:end-1]))
        for dim = 1:lssize[1]
			# We update the cholesky decompositions!
			if i != lssize[2]
				# keep in mind that we compute this for the next step!
				if i == 1
					chol = cholesky( getindex(pf.storage.Ktt_save, dim, p)[:,1:1].+ pf.hyperparams.Q[dim]).U
					pf.storage.Chols[dim, p][1:1,1:1] = chol[:,:]
				else # Else, we need to do the cholesky update to save computation
					chol = UpperTriangular(pf.storage.Chols[dim, p][1:i-1,1:i-1] )# This is the previous cholesky determinant
					# Update choesky.
					chol = update_chol(chol, getindex(pf.storage.Ktt_save,dim, p)[:,i:i] .+ pf.hyperparams.Q[dim], getindex(pf.storage.Ktt_save,dim, p)[:,1:i-1])
					pf.storage.Chols[dim, p][1:i,1:i] = chol[:,:]
				end
			end

            #  Compute the new covariances and store them because we will need them again.
            Kttfull = Kxxt(pf.hyperparams.gpinstance, pf.hyperparams.kernels[dim], xandufull[:,i+1:i+1], xandufull) #This has shape 1xN
            pf.storage.Ktt_save[dim, p][1:1, 1:i+1] = Kttfull[:, 1:i+1]
            mfull =  mx(pf.hyperparams.gpinstance, pf.hyperparams.mean_fs[dim], xandufull)
            pf.storage.Means_save[dim, p][1:i+1] = mfull[1:i+1]

			#  We do not need this if t = T
            if i != lssize[2]
				# keep in mind that we compute this for the next step!

				#Update Cholesky - We exchange the i+1 row with Kttfull = Kxxt(kernels[dim], xandufull[:,i+1:i+1], xandufull)
                cholesky_up_down_date!(pf.storage.CholsPGAS[dim, p], Kttfull, reshape(pf.hyperparams.Q[dim:dim], (1,1)), i+1)
				#Compute new ancestor weights
				# We do this here in order to save computation !!
				 xfull = vcat(Xtraj[dim,2:end, p], pf.xref[dim, i+2:end])


				lp = logpdf(Zygote_MvNormal(mfull, pf.storage.CholsPGAS[dim, p]), xfull)
                lpfs.Jointlp[p] = lpfs.Jointlp[p] + lp  	#  log p(x_{1:t}, x'_{t+1:T}|x_0)
            end
        end

    end
end



"""
	copy_cholesky!(CholsPGAS, Chols, Ktt_save, Means_save, indx, dim, i)

## Arguments
	- CholsPGAS::Matrix{UpperTriangular{T, Array{T,2}}}:		The Cholesky decompositions of K_{0:t, t+1:T-ref}+ I*Q
	- Chols::Matrix{UpperTriangular{T, Array{T,2}}}:			The Cholesky decompositions of K_{0:t} + I*Q
	- Ktt_save::Matrix{Matrix{T}}:								K_{t,0:t}
 	- Means_save::Matrix{Vector{T}}:							Means of x_{0:t}
	- indx::Vector{<:Integer}:									The new ancestor samples
	- dim::Integer:												Dimension
	- i::Integer:												i = t
"""

function copy_cholesky!(CholsPGAS::Matrix{UpperTriangular{T, Array{T,2}}}, Chols::Matrix{UpperTriangular{T, Array{T,2}}},
		Ktt_save::Matrix{Matrix{T}}, Means_save::Matrix{Vector{T}}, indx::Vector{<:Integer}, dim::Integer, i::Integer) where T<:Real
	# Compute number of children for every particle
	num_children = Vector{Vector{Int}}([[] for _ in 1:length(indx)])

	for (k,p) in enumerate(indx)
        push!(num_children[p], k)
    end
	# Some placeholder
    @views for p in 1:length(indx)
        ni = length(num_children[p])
        if ni > 0
			# If we have several childs, we need to dublicate the storage, this is where the bottle neck exists.
            @inbounds for j in 1:ni
				# This is the reason why we have to reorder! If p == num_children[p][j] we do not need to reorder
				# Further, due to reordernig the indices, we have a guarantee that this works and does not lead to a double copying confict ( i.e. 2 => 3, 3= >2)
				if p !== num_children[p][j] # Then we have to copy!
					for k = 1:dim
		                CholsPGAS[k, num_children[p][j]][:,:] =  	CholsPGAS[k, p][:,:]
		                Ktt_save[k, num_children[p][j]][1:1,1:i] =  Ktt_save[k, p][1:1, 1:i]
		                Means_save[k, num_children[p][j]][1:i] =  	Means_save[k, p][1:i]
						# Attention, we only need to make this if i > 1.
		                if i > 1
		                   Chols[k, num_children[p][j]][1:i-1, 1:i-1] = Chols[k, p][1:i-1, 1:i-1]
		                end
		            end
				end
			end
        end
    end
end


# Update the Cholesky decompositions
function reorderstorage!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, lssize::Tuple, indx::Vector{<:Integer}, i::Integer) where {I<:FullGPInference, A<:AbstractLSPGAlgorithm}
	copy_cholesky!(pf.storage.CholsPGAS, pf.storage.Chols, pf.storage.Ktt_save, pf.storage.Means_save, indx, lssize[1], i)
end




##########################################################

### PGAS Implementatoin

##########################################################

# We do not need to change anything here!


##########################################################

### FAAPF Implementation

##########################################################

# We do not need to change anything here!

##########################################################

### MCMCAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!

# Paper On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)

# We do not need to change anything here!

##########################################################

### SMC Implementatoin

##########################################################

# We need to update the cholesky decompositions and the storage!
# Further, we compute the Joint LogPdf here.
function init_pfinstance!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer,
	Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs) where  {A<: SMC, I <: FullGPInference}
	@views @inbounds for i = 1:n_particles
		# Compte Initial ancestor weigths and store the computations for the next steps.
		xandufull = hcat(vcat(reshape(Xtraj[:,1,i], (:,1)), U[:,1:1])) #Full trajectory
		for dim=1:lssize[1]
			# get kernel and mean
			kernel = pf.hyperparams.kernels[dim]
			mean_f = pf.hyperparams.mean_fs[dim]
			# compute Kernel output
			Kttfull = Kxx(pf.hyperparams.gpinstance, kernel, Array(xandufull))
			# Save first computation
			pf.storage.Ktt_save[dim,i][1:1, 1:1] = Kttfull[1:1, 1:1]
			#compute menas and store them
			mfull =  mx(pf.hyperparams.gpinstance, mean_f, xandufull)
			pf.storage.Means_save[dim,i][1:1] = mfull[1:1]
		end
	end
end

# We need to update the cholesky decompositions and the storage!
# Further, we compute the Joint LogPdf here.
function step_pfinstance!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, Xtraj::AbstractArray, n_particles::Integer,
	Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {A<: SMC, I <: FullGPInference}
		# xandufull is x_0:t - p th trajectory concatenated with x_t+1:T reference trajector<
	@views for p = 1:n_particles
	    xandufull = hcat(vcat(Xtraj[:,:,p], U[:, 1:i+1]))
	    for dim = 1:lssize[1]
			# We update the cholesky decompositions!
			if i != lssize[2]
				# keep in mind that we compute this for the next step!
				if i == 1
					chol = cholesky( getindex(pf.storage.Ktt_save, dim, p)[:,1:1].+ pf.hyperparams.Q[dim]).U
					pf.storage.Chols[dim, p][1:1,1:1] = chol[:,:]
				else # Else, we need to do the cholesky update to save computation
					chol = UpperTriangular(pf.storage.Chols[dim, p][1:i-1,1:i-1] )# This is the previous cholesky determinant
					# Update choesky.
					chol = update_chol(chol, getindex(pf.storage.Ktt_save,dim, p)[:,i:i] .+ pf.hyperparams.Q[dim], getindex(pf.storage.Ktt_save,dim, p)[:,1:i-1])
					pf.storage.Chols[dim, p][1:i,1:i] = chol[:,:]
				end
			end

	        #  Compute the new covariances and store them because we will need them again.
	        Kttfull = Kxxt(pf.hyperparams.gpinstance, pf.hyperparams.kernels[dim], xandufull[:,i+1:i+1], xandufull) #This has shape 1xN
	        pf.storage.Ktt_save[dim, p][1:1, 1:i+1] = Kttfull[:, 1:i+1]
	        mfull =  mx(pf.hyperparams.gpinstance, pf.hyperparams.mean_fs[dim], xandufull)
	        pf.storage.Means_save[dim, p][1:i+1] = mfull[1:i+1]

	    end
	end
end




# Update the Cholesky decompositions
# We do not need to update CholPGAS!!
function reorderstorage!(alg::BaseLSAlgorithm{I,A}, pf::ParticleFilterInstance, lssize::Tuple, indx::Vector{<:Integer}, i::Integer) where {A<: SMC, I <: FullGPInference}
	# Compute number of children for every particle
	num_children = Vector{Vector{Int}}([[] for _ in 1:length(indx)])

	for (k,p) in enumerate(indx)
        push!(num_children[p], k)
    end
	# Some placeholder
    @views for p in 1:length(indx)
        ni = length(num_children[p])
        if ni > 0
			# If we have several childs, we need to dublicate the storage, this is where the bottle neck exists.
            @inbounds for j in 1:ni
				# This is the reason why we have to reorder! If p == num_children[p][j] we do not need to reorder
				# Further, due to reordernig the indices, we have a guarantee that this works and does not lead to a double copying confict ( i.e. 2 => 3, 3= >2)
				if p !== num_children[p][j] # Then we have to copy!
					for k = 1:lssize[1]
		                pf.storage.Ktt_save[k, num_children[p][j]][1:1,1:i] =  	pf.storage.Ktt_save[k, p][1:1, 1:i]
		                pf.storage. Means_save[k, num_children[p][j]][1:i] =  	pf.storage.Means_save[k, p][1:i]
						# Attention, we only need to make this if i > 1.
		                if i > 1
		                   	pf.storage.Chols[k, num_children[p][j]][1:i-1, 1:i-1] = pf.storage.Chols[k, p][1:i-1, 1:i-1]
		                end
		            end
				end
			end
        end
    end
end
