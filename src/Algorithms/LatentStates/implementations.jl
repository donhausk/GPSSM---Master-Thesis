
############################################################
# Latent States Algorithms

# The structure is as follows. For any algorithm there exists an generat variant ( SMC, PGAS, MCMCAPF) etc.
# Further, for any setting of the GP, there exists a specific Algorithm i.e. SMCVI, SMCPGAS etc.

############################################################



###

# Implementations shared among many different algorithms!

###



##########################################################

### SMC Implementatoin

##########################################################


## We do not need any emission and therefore can use the default emission.





# Standard implementation for the initial step.
function initial_step!(alg::BaseLSAlgorithm{<:InferenceModel, <:AbstractLSSMCAlgorithm}, pf::AbstractPFInstance, X::AbstractArray, As::AbstractArray ,
	n_particles::Int, lpfs::LogPdfs, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams)
	@views @inbounds for i = 1:n_particles
		# Sample from p(x_0)
		distx0, xargs = get_distx0_and_args(pf.hyperparams.gpinstance, pf.hyperparams.args_θx0...)
		# Not that it is initially fine for ALL algorithms to set b_0 = N since we draw the samples independently!
		# We also do not need to store the LogPdfs since we are sampling from the prior...
		# This would be different in case someone wants to implemnt this using proposal distributions
		# We also do not need to store the Sequentiallp because this will be cancelled during the ancestor weights computation
		x0 = rand(distx0(xargs...))
		X[:,1, i] = x0
	end
end


# This is the important function!

function propagate_particles!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, μt::AbstractArray, Σt::AbstractArray,  X::AbstractArray,
   		n_particles::Int, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {A<: SMC, I <: InferenceModel}


	# In a first step, we resample!
	#Compute the weights from the observations w_i
	Ws = softmax!(copy(lpfs.Weightlp))

	# Resample ancestor index
	# Resample the remaining particles.
	indx = resample(Ws,n_particles)
	# update lpfs
	# This is a bit an overhead but it makes it easier for us to have a clean structure which can also be used for MCMCAPF
	reorderlpfs!(lpfs, indx)
	# Structure of μt: dimension x particles
	μt[:,:] = μt[:,indx]
	Σt[:,:] = Σt[:,indx]

	# Now, compute next particles!
	# Iterate over dimensions
	for p in 1:n_particles
		lp = 0.0
		xt = zeros(get_latent_type(), lssize[1])
		for dim in 1:lssize[1]
			mt = μt[dim, p]
			Σtel = Σt[dim, p]

			# One dimensional!
			# Sample new state.
			distxt = Normal(mt,sqrt(Σtel))
			tmp = rand(distxt)
			lp += logpdf(distxt,tmp)
			xt[dim] = tmp
		end
		#Set reference trajectory
		# Add new state
		X[:, i+1, p] = xt
		# Update our SequentialLogPdf
		# This is not relevant...
		#lpfs.Sequentiallp[p] += lp #  p(x_{1:t}|x_0) =  p(x_{1:t-1}|x_0)*  p(x_{t}| p(x_{0:t-1})
	end
	# update the marginal log likelihood estimate
	return indx, cat(Ws, ones(n_particles);dims = 2)
end



###################################


### Particle Gibbs with Ancestor Sampling


###################################





# Standard implementation for the intiial_step
function initial_step!(alg::BaseLSAlgorithm{<:InferenceModel, <:AbstractLSPGAlgorithm}, pf::AbstractPFInstance, X::AbstractArray, As::AbstractArray ,
	n_particles::Int, lpfs::LogPdfs, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams)
	@views @inbounds for i = 1:n_particles
		# Sample from p(x_0)
		distx0, xargs = get_distx0_and_args(pf.hyperparams.gpinstance, pf.hyperparams.args_θx0...)
		# Not that it is initially fine for ALL algorithms to set b_0 = N since we draw the samples independently!
		# We also do not need to store the LogPdfs since we are sampling from the prior...
		# This would be different in case someone wants to implemnt this using proposal distributions
		# We also do not need to store the Sequentiallp because this will be cancelled during the ancestor weights computation
		if i != n_particles
			x0 = rand(distx0(xargs...))
		else
			x0 = pf.xref[:,1]
		end
		X[:,1, i] = x0
	end
end


function propagate_particles!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, μt::AbstractArray, Σt::AbstractArray,  X::AbstractArray,
   		n_particles::Int, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {A<: PGAS, I <: InferenceModel}

	@views begin
		# In a first step, we resample!
		# Compute ancestor weights !
		#AW = JointLogPdf .+ LogPdf .- SequentialLogPdf
		AW = lpfs.Jointlp .- lpfs.Sequentiallp

		#Compute the weights from the observations w_i
		Ws = softmax!(copy(lpfs.Weightlp))
		#Unnormalised ancestor weigths
		unnormaw = softmax!(AW) .* Ws

		# Resample ancestor index
		an = resample(unnormaw ./ sum(unnormaw),1)[1]

		# Resample the remaining particles.
		icds = resample(Ws,n_particles-1)
		indx = push!(icds,an)
		# update lpfs
		# This is a bit an overhead but it makes it easier for us to have a clean structure which can also be used for MCMCAPF
		reorderlpfs!(lpfs, indx)
		# Structure of μt: dimension x particles
		μt[:,:] = μt[:,indx]
		Σt[:,:] = Σt[:,indx]

		# Now, compute next particles!
		# Iterate over dimensions
		for p in 1:n_particles
			lp = 0.0
			xt = zeros(get_latent_type(), lssize[1])
			if p == n_particles
				xt = pf.xref[:, i+1]
			end
			for dim in 1:lssize[1]
				mt = μt[dim, p]
				Σtel = Σt[dim, p]

				if p == n_particles
					# One dimensional!
					# Only compute the tranistion probability
					lp += logpdf(Normal(mt,sqrt(Σtel)), xt[dim]) # Attention std vs variance!!!
				else
					# One dimensional!
					# Sample new state.
					distxt = Normal(mt,sqrt(Σtel))
					tmp = rand(distxt)
					lp += logpdf(distxt,tmp)
					xt[dim] = tmp
				end
			end
			#Set reference trajectory

			# Add new state
			X[:, i+1, p] = xt

			# Update our SequentialLogPdf
			lpfs.Sequentiallp[p] += lp #  p(x_{1:t}|x_0) =  p(x_{1:t-1}|x_0)*  p(x_{t}| p(x_{0:t-1})
		end
	end
	return indx, cat(Ws,unnormaw./ sum(unnormaw); dims=2)
end






##########################################################

### FAAPF

##########################################################





function propagate_particles!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, μt::AbstractArray, Σt::AbstractArray,  X::AbstractArray,
   		n_particles::Int, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer )where {A<: FAAPF, I <: InferenceModel}

	@views begin
		μtem, Sstar = get_fa_pred(pf.emission, μt, Σt, Y[:,i])

		# p(y_t | x_{0:t-1}) used for resamplign.
		Lps = get_pyxt1(pf.emission, μt, Σt, Y[:,i])
		# the estimate of the marginal log likelihood =log(1/n_particle ∑_{i = 1}^n_particles w_i) = log(1/n_particle ∑_{i = 1}^n_particles p(y_t| x_{0:t-1}^i))
	 	lpfs.marginallp += StatsFuns.logsumexp(Lps) - log(n_particles)
		# Resmample
		# Ancestor samping
		# Compute ancestor weights !
		#AW = JointLogPdf .- SequentialLogPdf
		AW = softmax!(lpfs.Jointlp .- lpfs.Sequentiallp)
		#We do not need to add p(y_t|x_t) because we are in the fully adapted case!

		#This are computed during every step.

		as = resample(AW,1)[1]# This was the bug!!!


		# Resample the rest
		indices =  resample(softmax!(copy(Lps)), n_particles-1)
		push!(indices, as)

		# New Sequentiallp, we keep track of this.
		# Reorder them!
		reorderlpfs!(lpfs, indices)

		for p = 1:n_particles-1
			xnew = rand(MvNormal(vec(μtem[indices[p]]), Sstar[indices[p]])) # Sample new particle proportional to p(x_t| y_t, x_{0:t-1})
			lpfs.Sequentiallp[p] += logpdf(MvNormal(μt[:,indices[p]], Diagonal(Σt[:,indices[p]])), xnew) # Increase the SequentialLogPdf
			X[:, i+1, p] = xnew # Update new state.
		end

		# Fix N-th particle
		X[:, i+1, n_particles] = pf.xref[:,i+1]
		# Increase the SequentialLogPdf for last particle
		lpfs.Sequentiallp[n_particles] += logpdf(MvNormal(μt[:,indices[n_particles]], Diagonal(Σt[:,indices[n_particles]])), pf.xref[:,i+1])
	end
	return indices,  cat(softmax!(copy(lpfs.Weightlp)),AW; dims=2)
end



##########################################################

### MCMCAPF Implementation

##########################################################


## We do not need to implement the finalise_pf function because all samples have equal weight at the end!
# Paper On embedded hidden Markov models and particle Markov chain Monte Carlo methods (Finke, 2016)


# Turing has no autoregrssive update....





function propagate_particles!(alg::BaseLSAlgorithm{I,A}, pf::AbstractPFInstance, μt::AbstractArray, Σt::AbstractArray,  X::AbstractArray,
   		n_particles::Int, Y::AbstractArray, U::AbstractArray, lssize::Tuple, hyperparams::HyperParams, lpfs::LogPdfs, i::Integer ) where {A<: MCMCFAAPF, I <: InferenceModel}
	@views begin
		# Note: xt, is the reference trajectory at time t

		# Ancestor sampling. Ws ∝ log p(x_{t:T}| x_{0:t-1}^i), note that the weights are constant.

		# This is only relevant to compute the log likelihood!!
		if pf.emission.Eval !== nothing

			Lps = get_pyxt1(pf.emission.Eval, μt, Σt, Y[:,i])
			mlog = StatsFuns.logsumexp(Lps) - log(n_particles)
		else
			mlog = 0.0
		end


		# the estimate of the marginal log likelihood =log(1/n_particle ∑_{i = 1}^n_particles w_i) = log(1/n_particle ∑_{i = 1}^n_particles p(y_t| x_{0:t-1}^i))
		lpfs.marginallp += mlog
		# Resmample
		# Ancestor samping
		# Compute ancestor weights !
		#AW = JointLogPdf .- SequentialLogPdf
		AW = softmax!(lpfs.Jointlp .- lpfs.Sequentiallp)
		#We do not need to add p(y_t|x_t) because we are in the fully adapted case

		#This are computed during every step.


		idcstart = resample(AW,1)[1]

		# First Sample the initial l value uniformly.
		lstart =  resample(1/n_particles .* ones(n_particles),1)[1]

		# Set the observation to Y
		# Some Temporal Log Pdf term storing the transition probabiliyt. This is used in the latter on for the computation of the next Ancestor weights.
		SequentialLogPdfTmp = zeros(size(lpfs.Sequentiallp))

		# Probability of receiving xt for the ancestor trajectory.

		SequentialLogPdfTmp[lstart] = get_logps(μt, Σt, pf.xref[:,i+1])[idcstart]

		# Incices
		indices = zeros(Int,n_particles)
		#Set ancestor index
		indices[lstart]= idcstart
		# Set reference trajectory
		X[:,i+1, lstart] =  pf.xref[:,i+1]

		# Iterate trough 1... lstart-1
		if lstart !== 1
			xtm = pf.xref[:,i+1] # Set initial value
			am = idcstart # Set initial index
			for p = 1:lstart-1
				# Propagate particle
				xtm, am, lp = propagate_single_particle(xtm,am, μt, Σt, pf.emission, Y[:,i]) # We use symmetric kernels!!!

				# Set transition probability, this will be needed for the next ancestor sampling.
				SequentialLogPdfTmp[lstart-p] = lp
				# Set indices
				indices[lstart-p] = am
				# Set particle
				X[:, i+1, lstart-p] = xtm
			end
		end
		# Same as before, we use symmetric kernel and therefore do not distinguish between forward and backward iteration.
		if lstart !== n_particles
			xtm = pf.xref[:,i+1]
			am = idcstart
			for p = lstart+1:n_particles
				xtm, am, lp = propagate_single_particle(xtm,am,μt, Σt, pf.emission, Y[:,i]) # We use symmetric kernels!!!
				SequentialLogPdfTmp[p] += lp
				indices[p] = am
				X[:, i+1, p] = xtm
			end
		end

		# Update SequentialLogPdf
		lpfs.Sequentiallp[:] = lpfs.Sequentiallp[indices] .+ SequentialLogPdfTmp
	end
	return indices,  cat(softmax!(copy(lpfs.Weightlp)), AW; dims=2)
end







"""
    get_init_alg(alg)

## Arguments
    - alg::AbstractLSAlgorithm:     The latent states algorithm
    - ahp::AbstractHPAlgorithm:     The Hp algorithm

Check if the two algorithms are compatible

"""
check_compatibility(alg::AbstractLSAlgorithm, inf::I)  where{I<:InferenceModel} = error("The latent states sampler is not compatible with the hyper parameter sampler!")
check_compatibility(alg::Union{PGAS, SMC}, inf::I)  where {I<:InferenceModel}  = true # Any methods are compatiblecheck_compatibility(alg::Union{PGAS, SMC}, ahp::M{I})  where {I<:InferenceModel, M<:AbstractHPAlgorithm}  = true # Any methods are compatible
check_compatibility(alg::Union{FAAPF}, inf::I)  where {I<:Union{FullGPInference, SparseInference, PredictionInference}}  = true # Any methods are compatible
check_compatibility(alg::Union{MCMCFAAPF},  inf::I)  where {I<:Union{FullGPInference, SparseInference, PredictionInference}}  = true # Any methods are compatible
check_compatibility(alg::Union{MCMCFAAPF},  inf::I)  where {I<:Union{HybridVIInference}}  = alg.compute_mll ? error("The marginal log likelihood can not be computed in for the Hybrid VI algorithm ") : true # Any methods are compatible


"""
    get_init_alg(alg)

## Arguments
    - alg::BaseLSAlgorithm

Returns the algorithm used for the initial step. Note that Particle Gibbs algorithm require an initial trajectory.

"""
get_init_alg(basealg::BaseLSAlgorithm{I,A}) where { A<:AbstractLSAlgorithm, I<:InferenceModel} = basealg

get_init_alg(basealg::BaseLSAlgorithm{I,A} ) where { A<:AbstractLSPGAlgorithm, I<:InferenceModel} = BaseLSAlgorithm{I, SMC}(SMC(basealg.alg.n_particles))



#
# """
#
# Same story as for the propagate_single_particle but for the VI model.
#
# """
#
# function propagate_particle(mcmc_trans::VIEM, xtm, mu, Q, am)
#
# 	mcmc_trans.model.args[:mu][:] = mu[:,am]
# 	niter = mcmc_trans.nrepetitions
#
# 	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
#
# 	esampler = Turing.Sampler(mcmc_trans.alg, emodel)
# 	esampler.state.vi.metadata.x.vals[:] = xtm[:]
#
# 	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, mcmc_trans.nrepetitons) # This must be made unfortunately...
# 	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
# 	for i =2:2
# 		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
# 	end
# 	xnew = esampler.state.vi.metadata.x.vals
#
#
# 	Lps = get_logps(mu, Q, xnew)
# 	anew = resample(softmax!(Lps),1)[1]
# 	# remake this
#
# 	mcmc_trans.model.args[:mu][:] = mu[:,anew]
# 	mcmc_trans.model.args[:sigma][:] = sigma[:,anew]
#
# 	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
#
#
# 	esampler = Turing.Sampler(mcmc_trans.alg, emodel)
# 	esampler.state.vi.metadata.x.vals[:] = xnew[:]
#
# 	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, mcmc_trans.nrepetitons) # This must be made unfortunately...
# 	#esampler = mcmc_trans.sampler
# 	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
# 	for i =2:2
# 		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
# 	end
# 	xnew2 = esampler.state.vi.metadata.x.vals
#
# 	return xnew2, anew
# end




#
# """
#
# VI needs three different models, one for t =1, one for t = N and one for the rest.
#
# """
#
# Turing.@model mcmcapfvi_t1(args_θx0, u, kernels, mean_fs, Z_Ind, Σu, Kzzv, lssize, Q) = begin
#     distx0, xargs = get_distx0_and_args(args_θx0[1]...)
#     x ~ distx0(xargs...)
#     xtu = cat(vcat(x, u), dims=2)
#     for dim = 1:lssize[1] #ok
#         Ktz = Kxxt(kernels[dim],Z_Ind, xtu)
#         At1 = Kzzv[dim]\((Ktz' /Kzzv[dim])')
#         v = (Ktz' /Kzzv[dim])
#         Bt1 = Kxx(kernels[dim], xtu) - v*v'
#         # We use that Q[dim:dim is one dimensional]
#         v2 = ((Σu[dim])' *At1)
#         lp = @logpdf
#         lp += -0.5*(tr(1/Q[dim]*(Bt1 + v2'*v2))) # We are going to need this....
#     end
# end
#
# Turing.@model mcmcapfvi_tn(mu, sigma, y, args_θy) = begin
#     x ~ MvNormal(mu,  Diagonal(sigma)) # We need to store everything in arrays...
#     distxy, yargs = get_distxy_and_args(x, args_θy[1]...)
#     # Now we have to compute the ancestor weights for the next observation...
#     y ~ distxy(yargs...)
# end
#
#
# Turing.@model mcmcapfvi(mu, sigma, args_θy, y, u, kernels, mean_fs, Z_Ind, Σu, Kzzv, lssize, Q) = begin
#     x ~ MvNormal(mu, Diagonal(sigma)) # We need to store everything in arrays...
#     xtu = cat(vcat(x, u), dims=2)
#     for dim = 1:lssize[1] #ok
#         Ktz = Kxxt(kernels[dim],Z_Ind, xtu)
#         At1 = Kzzv[dim]\((Ktz' /Kzzv[dim])')
#         v = (Ktz' /Kzzv[dim])
#         Bt1 = Kxx(kernels[dim], xtu) - v*v'
#         # We use that Q[dim:dim is one dimensional]
#         v2 = ((Σu[dim])' *At1)
#         lp = @logpdf
#         lp += -0.5*(tr(1/Q[dim]*(Bt1 + v2'*v2))) # We are going to need this....
#     end
#     distxy, yargs = get_distxy_and_args(x, args_θy[1]...)
#     # Now we have to compute the ancestor weights for the next observation...
#     y ~ distxy(yargs...)
# end
#
#
#
# """
#
# Only for the last step, which must be threated differently.
#
#
# """
#
#
#
# function  propagate_particle_sn(mcmc_trans, xtm,mu, Q, am)
# 	mcmc_trans.modelsl.args[:mu][:] = mu[:,am]
#
# 	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
#
#
# 	esampler = Turing.Sampler(mcmc_trans.alg, emodel)
# 	esampler.state.vi.metadata.x.vals[:] = xtm[:]
#
# 	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, 5) # This must be made unfortunately...
#
# 	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
# 	for i =2:5
# 		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
# 	end
# 	xnew = esampler.state.vi.metadata.x.vals
#
#
# 	Lps = get_logps(mu, Q, xnew)
# 	anew = resample(softmax!(Lps),1)[1]
#
#
# 	mcmc_trans.model.args[:mu][:] = mu[:,anew]
#
# 	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
#
#
# 	esampler = Turing.Sampler(mcmc_trans.alg, emodel)
# 	esampler.state.vi.metadata.x.vals[:] = xnew[:]
#
# 	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, 5) # This must be made unfortunately...
# 	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
# 	for i =2:5
# 		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
# 	end
# 	xnew2 = esampler.state.vi.metadata.x.vals
#
#
# 	return xnew2, anew
# end
#
#
# """
#
# Only for the first step, which must be threated differently.
#
#
# """
#
# function propagate_particle_s1(mcmc_trans,xtm)
#
# 	emodel = mcmc_trans.model.modelgen.f(mcmc_trans.model.args...)
#
# 	esampler = Turing.Sampler(mcmc_trans.alg, emodel)
# 	esampler.state.vi.metadata.x.vals[:] = xtm[:]
#
# 	AbstractMCMC.sample_init!(Random.GLOBAL_RNG, emodel, esampler, 5) # This must be made unfortunately...
# 	#esampler = mcmc_trans.sampler
# 	ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, 1;)
# 	for i =2:5
# 		ts = AbstractMCMC.step!(Random.GLOBAL_RNG, emodel, esampler, i, ts;)
# 	end
# 	xnew = esampler.state.vi.metadata.x.vals
#
# 	return xnew
# end
