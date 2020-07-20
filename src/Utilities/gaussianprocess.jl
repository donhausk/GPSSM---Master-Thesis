

###################################################
#In case we plan to change the backend            #
###################################################

# Attention the GaussianProcesses library has a bizar way of defining states.
const AbstractKernelType = Stheno.Kernel
const AbstractMeanType = Stheno.MeanFunction

@inline @nograd get_jitter_mat(instance::GPSSMInstance, Kmatrix) = Diagonal(ones(eltype(Kmatrix), size(Kmatrix)[1]) .* get_jitter_level(instance))


#Small correction to allow vectors for the stheno scale
function Stheno.scale(kernel::Stheno.Kernel, var::AbstractVector)
    @assert length(var) == 1 "[GPSSM-Stheno] The variance of the scale must be of length 1!"
    return Stheno.scale(kernel,var[1])
end


"""
     Kxx(instance, kernel, x:)

## Arguments
    - instance::GPSSMInstance:      The GP Instance
    - kernel::AbstractKernelType    Kernel of type AbstractKernelType
    - x::AbstractMatrix:            Matrix with dim x T

# Evaluate Stheno Kernel function, returns the covariance matrix with optionally some jitter.
# Note that the output is of dimension T x T

"""

@inline function Kxx(instance::GPSSMInstance, kernel::AbstractKernelType, x::AbstractMatrix)
    Kmatrix = Stheno.pairwise(kernel,Stheno.ColVecs(x))
    if add_jitter(instance)
        Kmatrix = Kmatrix + get_jitter_mat(instance, Kmatrix)
    end
    return Symmetric(Kmatrix)
end


"""
     DiagKxx(instance, kernel, x:)

## Arguments
- instance::GPSSMInstance:      The GP Instance

    - kernel::AbstractKernelType    Kernel of type AbstractKernelType
    - x::AbstractMatrix:            Matrix with dim x T

Returns diag(Kxx(instance, kernel, x))

"""

@inline function DiagKxx(instance::GPSSMInstance, kernel::AbstractKernelType, x::AbstractMatrix)
    Kvec = Stheno.elementwise(kernel,Stheno.ColVecs(x))

    if add_jitter(instance)
        Kvec = Kvec .+ get_jitter_level(instance)
    end
    return Kvec
end




"""
    Kxxt(instance, kernel, x1, x2)

## Arguments
    - instance::GPSSMInstance:      The GP Instance
    - kernel::AbstractKernelType
    - x1::AbstractMatrix
    - x2::AbstractMatrix

# Same as Kxx but not symmetric.


"""



@inline function Kxxt(instance::GPSSMInstance, kernel::AbstractKernelType, x1::AbstractMatrix, x2::AbstractMatrix) # structure of x1 : N x dim x - where N is the amount of different x-s
    Kmatrix = Stheno.pairwise(kernel, Stheno.ColVecs(x1), Stheno.ColVecs(x2))
end

"""
    mx(insstance, m, x)

## Arguments
    - instance::GPSSMInstance:      The GP Instance
    - m::AbstractMeanType:          The mean function
    - x::AbstractMatrix:            Input Matrix for the mean function

# Evaluate Stheno mean function elementwise.


"""


@inline function mx(instance::GPSSMInstance, m::AbstractMeanType, x::AbstractMatrix)
    return Stheno.elementwise(m,Stheno.ColVecs(x))
end





"""
    get_transition_mean( I, Ktt0t, Ktta, Chol)

## Arguments
	- ::Type{<:InferenceModel:   The inference model
    - xmeanst::AbstractVector:      mean(x_{t-1})
    - xmeans0t::AbstractVector:     mean(x_{0:t-2})
    - xout::AbstractVector:         x_{0:t-2}
    - Ktt0t::AbstractMatrix:        K(x_{t-1},x_{0:t-2})
    - Chol::UpperTriangular:        The choleksy decomposition of K(x_{t-2},x_{t-2}) + I*Q

# The conditional transition mean. Formulas are taken form Frigolas PHD thesis ( equation 3.15 page 30)


"""

@inline function get_transition_mean(
	::Type{<:InferenceModel},
    xmeanst::AbstractVector,
    xmeans0t::AbstractVector,
    xout::AbstractVector,
    Ktt0t::AbstractMatrix,
    Chol::UpperTriangular
)


    v = xout .- xmeans0t

    α = Kinv_A(Chol, v)
    gpmeans = Ktt0t* α
    xmeans_t = xmeanst + vec(gpmeans)
    return vec(xmeans_t)
end

"""
    get_transition_cov(I, Ktt0t, Ktta, Chol)

## Arguments
	- ::Type{<:InferenceModel:   The inference model
    - Ktt0t::AbstractMatrix     This is K(x_{t-1}, x{0:t-2})
    - Ktta::AbstractMatrix       This is K(x_{t-1}, x_{t-1})+ I*Q
    - Chol::UpperTriangular     The choleksy decomposition of K(x_{t-2},x_{t-2}) + I*Q

# The conditional transition covariance. Formulas are taken form Frigolas PHD thesis ( equation 3.15 page 30)


"""

@inline function get_transition_cov(
	::Type{<:InferenceModel},
    Ktt0t::AbstractMatrix,
    Kttq::AbstractMatrix,
    Chol::UpperTriangular
)
    Cov = Kttq - A_Kinv_A(Chol, Ktt0t')
    return Cov
end




"""
    get_transition_mean(I, Ktz,mti,  Mz)
## Arguments
	- ::Type{<:InferenceModel:   The inference model
    - Ktz::AbstractMatrix:      This corresponds to K_{x_t, Z_Ind}
	- mti::AbstractVector:      This is the mean at time i, (i.e. m(x_t))
    - Mz::AbstractVector:       This is K_zz^{-1}(U_ind - m(Z_Ind))

# The conditional sparse transition mean. Formulas are taken form Frigolas PHD thesis ( equation 3.15 page 30)


"""
@inline function get_transition_mean( ::Type{<:Union{MarkovianInferenceModel}}, Ktz::AbstractMatrix,mt1::AbstractVector, mz::AbstractVector)
    mt1+ Ktz' *mz
end


"""
    get_transition_cov(I, Kttq::AbstractMatrix, Ktz::AbstractMatrix, Cholzz::UpperTriangular,CholzAz::UpperTriangular)

## Arguments
	- ::Type{<:InferenceModel:   The inference model
	- Kttq::AbstractMatrix:        Ktt + I *Q
	- Ktz::AbstractMatrix:		   This corresponds to K_{x_t, Z_Ind}
    - Cholzz::UpperTriangular:	   The cholesky decomposition of Kzz
    - CholzAz::UpperTriangular:    The cholesky decomposition of Kzz^{-1} Σ Kzz^{-1}

# The sparse covariance function, which is Ktt + Q - Ktz Kzz^{-1} Kzt +  Ktz Kzz^{-1} Σ Kzz^{-1} Kzt

"""
@inline function get_transition_cov( ::Type{<:MarkovianInferenceModel},  Kttq::AbstractMatrix, Ktz::AbstractMatrix, Cholzz::UpperTriangular,Sigma::UpperTriangular)
	v = Kinv_A(Cholzz, Ktz)
	return Kttq -  A_Kinv_A(Cholzz, Ktz) + A_K_A(Sigma,v')
end






"""
    get_prediction_transition_mean(I,  xmstar, mu, Kxstarx)

## Arguments

    - ::Type{<:InferenceModel:   The inference model
    - xmstar::AbstractVector:       mean(x*)
    - mu:                           (K_{0:T-1}+I*Q)^{-1}(x_{1:T} - m_{0:T-1}) or  (K_z+I*Q)^{-1}(u - m_z)
    - Kxstarx::AbstractMatrix:      K(x*,x_{0:T-1}) or  K(x*,x_z)

The prediction mean. This is the same for all inference models.

"""

@inline function get_prediction_transition_mean(
    ::Type{<:InferenceModel},
    xmeanst::AbstractVector,
    mu::AbstractArray,
    Ktt0t::AbstractMatrix,
)
	get_transition_mean(SparseInference, Ktt0t', xmeanst, mu)
end



"""
    get_prediction_transition_cov(I,  Kxstarx, Kxstarxstar, Chol)

## Arguments

    - ::Type{<:InferenceModel:   The inference model
    - Kxstarx::AbstractVector:       K(x*,x_{0:T-1}) or  K(x*,x_z)
    - Kxstarxstarq:                  Kx* + I*Q
    - Chol::AbstractMatrix:          The cholesky decomposition of  K_{0:T-1}+I*Q or K_z+I*Q

The standard prediction covariance.


"""


# Exactly the same as get_prediction_cov!

@inline function get_prediction_transition_cov(
    I::Type{<:InferenceModel},
    Kxstarx::AbstractMatrix,
    Kxstarxstarq::AbstractMatrix,
    Chol::UpperTriangular
)
    get_transition_cov(I, Kxstarx, Kxstarxstarq, Chol)
end

"""
    get_prediction_transition_cov(I,  Kxstarx, Kxstarxstar, Chol)

## Arguments

    - ::Type{<:InferenceModel:     The inference model
	- Kxstarstar::AbstractMatrix:  Ktt + I *Q
	- Kstarz::AbstractMatrix:	   This corresponds to K_{x*, Z_Ind}
    - Cholzz::UpperTriangular:	   The cholesky decomposition of Kzz
    - CholzAz::UpperTriangular:    The cholesky decomposition of Kzz^{-1} Σ Kzz^{-1}

The standard prediction covariance.


"""


# Exactly the same as get_prediction_cov!

@inline function get_prediction_transition_cov(
    I::Type{<:MarkovianInferenceModel},
    Kxstarx::AbstractMatrix,
    Kxstarxstarq::AbstractMatrix,
	Cholzz::UpperTriangular,
	CholzAz::UpperTriangular

)
    get_transition_cov(I, Kxstarxstarq, Kxstarx', Cholzz, CholzAz)
end





"""
  elbo_optimal(Kxx::AbstractMatrix, Kxz::AbstractMatrix, Cholz::UpperTriangular, Q::Real, y::AbstractVector, mx::AbstractVector)

## Arguments
    - Kxx::AbstractMatrix:      A vector containing the diagonal elements of K_{0:T-1}
    - DiagKxxx::AbstractMatrix:  This is K_{0:T-1},Z
    - Cholz::UpperTriangular:   The cholesky decomposition of Kzz
    - Q::Real:                  The process noise which is one dimensional for the single output gp.
    - y::AbstractVector:        The observations, in ourcase this is X^dim_{2:T}
    - mx::AbstractVector:       The means mx_{0:T-1}

The elbo when the variational approximation is the optimal approximation.

Paper:  M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function elbo_optimal(DiagKxxx::AbstractVector, Kxz::AbstractMatrix, Cholz::UpperTriangular, Q::Real, y::AbstractVector, mx::AbstractVector)
	# Compute the cholesky decomposition of (Kzz + Kzx Kxz / Q)
	Chol_Σ_opt = K_CTCQ_Chol(Cholz, Kxz, Q)
 	return elbo_optimal(DiagKxxx, Kxz, Cholz, Q, y, mx, Chol_Σ_opt)
end
function elbo_optimal(DiagKxxx::AbstractVector, Kxz::AbstractMatrix, Cholz::UpperTriangular, Q::Real, y::AbstractVector, mx::AbstractVector, Chol_Σ_opt::UpperTriangular)


	# First, we comput the trace term!
	# -1/(2*Q) *(tr(Kxx) - tr(Kxz kzz^{-1}) Kxz))
	# tr(Kxz kzz^{-1}) Kxz) = tr(Kxz( U^T U )^{-1} Kzx) = tr( Kxz U^{-1} ( Kxz U^{-1})^T) = Frobenius norm^2 of Kxz U^{-1} ( can be computed in T m^2)
	traceterm = -1/(2*Q) * (sum(DiagKxxx) - squared_frobenius_norm((Kxz / Cholz)'))


	# Let M =  I*Q + Kxz Kzz^{-1} Kzx
	# Next, we need to compute log( N(y| mx, M)
	# First, we compute the log determinant. This is (Rasmussen gp book, appendix A) |I*Q| |Kzz^{-1}| |Kzz + Kzx Kxz /Q| = |I*Q|/|Kzz| |Chol_Σ_opt|^2
	# Now transferring this to log(det(M)^{-1/2}), we get:
	ldet = logdet(Cholz) - length(y)/2 *log(Q) - logdet(Chol_Σ_opt)

	# Next, we compute the term (y- mx)^T  M^{1} ( y- mx)
	# However, we need to use the formula given in rasmussens gp book, appendix A in order to have O(T*m^2) complexity!!

	v = y .- mx

	# M^{-1} = I*1/Q - I*1/(Q^2) Kxz (Chol_Σ_opt^T* Chol_Σ_opt)^{-1}Kzx
	Kxzv = vec(v' * Kxz)

	expterm = v'*v /Q - 1/(Q^2)*A_Kinv_A(Chol_Σ_opt, Kxzv)
	# only for debugging.

	# This is the logarithm of N(y|mx, M)
	logNy = -1/2*(length(y) * log(2π)+ expterm) + ldet

	# Finally, we have:
	return logNy + traceterm
end




"""
  elbo(μq::AbstractVector, Σq::UpperTriangular, Kxx::AbstractMatrix, Kxz::AbstractMatrix, Cholz::UpperTriangular, Q::Real, y::AbstractVector, mx::AbstractVector)

## Arguments
	- μq::AbstractVector:		The mean of the inducing points.
	- Σq::UpperTriangular:		The covariance matrix of the inducing points
    - DiagKxx::AbstractVector:  The diagonal elements of K_{0:T-1}
    - Kxz::AbstractMatrix:      This is K_{0:T-1},Z
    - Cholz::UpperTriangular:   The cholesky decomposition of Kzz
    - Q::Real:                  The process noise which is one dimensional for the single output gp.
    - y::AbstractVector:        The observations, in ourcase this is X^dim_{2:T}
    - mx::AbstractVector:       The means mx_{0:T-1}
	- mz::AbstractVector:		The means m_Zind

The suboptimal elbo, with μq and Σq any covariance matrix. The formula is derived in the underlying thesis.

Paper:  M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function elbo(μq::AbstractVector, Σq::UpperTriangular ,Cholz::UpperTriangular, DiagKxx::AbstractVector, Kxz::AbstractMatrix,  Q::Real, y::AbstractVector, mx::AbstractVector,  mz::AbstractVector)

	# Get mean and covariance of p(fm | y)
	# The covariance is K Σ K

	Chol_Σ_opt = K_CTCQ_Chol(Cholz, Kxz, Q)
	# The cholesky decomposition is then


	Cholopt = K_ΣInv_K(Chol_Σ_opt, Cholz)

	# In addition, we compute the mean as:
	# mz + 1/Q * Kxx * Σ^{-1} Kxz (y - mx)
	v = Kxz' * ( y .- mx)
	intermediate = Kinv_A(Chol_Σ_opt, v)
	mopt = mz + 1/Q .* (Cholz' * (Cholz * intermediate))
	# Finally, we have the elbo = elbo_optimal - KL term
	elbo_o =  elbo_optimal(DiagKxx, Kxz, Cholz, Q, y, mx, Chol_Σ_opt)
	elbo2 =  KL(μq, mopt, Σq, Cholopt)
	return elbo_o - elbo2

end



"""
  sparseopt(μq::AbstractVector, Σq::UpperTriangular, Kxx::AbstractMatrix, Kxz::AbstractMatrix, Cholz::UpperTriangular, Q::Real, y::AbstractVector, mx::AbstractVector)

## Arguments
    - Kxz::AbstractMatrix:      This is K_{0:T-1},Z
    - Cholz::UpperTriangular:   The cholesky decomposition of Kzz
    - Q::Real:                  The process noise which is one dimensional for the single output gp.
    - y::AbstractVector:        The observations, in ourcase this is X^dim_{2:T}
    - mx::AbstractVector:       The means mx_{0:T-1}
	- mz::AbstractVector:		The means m_Zind

The optimal sparse mean and covariance of the inducing points.

Paper:  M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""

function sparseopt(Cholz::UpperTriangular, Kxz::AbstractMatrix,   Q::Real, y::AbstractVector,mx::AbstractVector,  mz::AbstractVector)

	# Get mean and covariance of p(fm | y)
	# The covariance is K Σ K

	Chol_Σ_opt = K_CTCQ_Chol(Cholz, Kxz, Q)
	# The cholesky decomposition is then
	M = (Cholz/Chol_Σ_opt)
	M2 = Symmetric(M* M')
	Σopt = Symmetric(Cholz' * M2 * Cholz )
	# In addition, we compute the mean as:
	# mz + 1/Q * Kxx * Σ^{-1} Kxz (y - mx)
	v = Kxz' * ( y .- mx)
	intermediate = Kinv_A(Chol_Σ_opt, v)
	mopt = mz + 1/Q .* (Cholz' * (Cholz * intermediate))
	# Finally, we have the elbo = elbo_optimal - KL term
	return  mopt, Σopt
end
"""

	squared_frobenius_norm(A::AbstractMatrix{<:Real})

Returns the sqaure of the frobenius norm

"""

squared_frobenius_norm(A::AbstractMatrix{<:Real}) = sum(A.^2)


"""

	K_CTCQ_Chol(Cholz::UpperTriangular, Kxz::AbstractMatrix, Q::Real)

## Arugments
	- Cholz::UpperTriangular:		The cholesky decomopsition of Kzz
	- Kxz::AbstractMatrix:			Kxz
	- Q::Real:						Q

computes the cholesky decomposition of (Kzz + Kzx Kxz/Q)



"""

function K_CTCQ_Chol(Cholz::UpperTriangular, Kxz::AbstractMatrix, Q::Real)
	return cholesky(Symmetric(Cholz'* Cholz + Kxz' * Kxz ./ Q)).U
end


"""
	K_ΣInv_K(ΣC::UpperTriangular, KC::UpperTriangular)
## Arguments
	- ΣC::UpperTriangular:		The cholesky decomposition of Σ
	- KC::UpperTriangular:		The cholesky decomposition of K

Compute the cholesky of K*Σ^{-1}*K

"""
function K_ΣInv_K(ΣC::UpperTriangular, KC::UpperTriangular)
	M = (KC/ΣC)
	M2 = Symmetric(M* M')


	cholesky(Symmetric(KC' * M2 * KC )).U
end


"""
    A_Kinv_A(Chol::UpperTriangular, A::AbstractMatrix)

## Arguments
    - Chol:     The cholesky decomposition of K nxn
    - A:        A matrix mxn


returns A^T K^{-1} A

"""

function A_Kinv_A(Chol::UpperTriangular, A::Union{AbstractMatrix, AbstractVector})
    v = A'/ Chol
    return  get_sym(v*v')
end
get_sym(a::AbstractMatrix) = Symmetric(a)
get_sym(a::Union{Real, AbstractVector}) = a



"""
    A_K_A(Chol::UpperTriangular, A::AbstractMatrix)

## Arguments
    - Chol:     The cholesky decomposition of K nxn
    - A:        A matrix mxn


returns A K A^T

"""

function A_K_A(Chol::UpperTriangular, A::AbstractMatrix)
    v = Chol* A'
    return  Symmetric(v'*v)
end


"""
    Kinv_A(Chol::UpperTriangular, A::AbstractMatrix)

## Arguments
    - Chol:     The cholesky decomposition of K nxn
    - A:        A vector n or a matrix nxm


returns K^{-1} A

"""

function Kinv_A(Chol::UpperTriangular, A::Union{AbstractMatrix, AbstractVector})
    v = A'/ Chol
    return  Chol\ v'
end


"""
    Ainv_K_Ainv(Chol::UpperTriangular, A::UpperTriangular)

## Arguments
    - Chol:     The cholesky decomposition of K nxn
    - CholA:        The Cholesky decomposition of A


returns the cholesky decomposition of A^{-1} K A^{-1}

"""

function Ainv_K_Ainv(Chol::UpperTriangular, CholA::UpperTriangular)
	# (CA^T CA)^{-1} CK^T CK (CA^T CA) = CA^{-1} CA^{-T} CK^T CK CA^{-1} CA^{-T}
	# = = CA^{-1} ( CK CA^{-1}) ( CK CA^{-1})^T CA^{-T}
	# = (CA^{-1}( CK CA^{-1}) )^T CA^{-1} ( CK CA^{-1}))^T =
	#CKInvA = Chol/CholA
	#AinvCKInvAT =  CholA\(CKInvA')
    #return  cholesky(Symmetric(AinvCKInvAT * (AinvCKInvAT'))).U

	m = A_Kinv_A(Chol,CholA)
    icl = inv(cholesky(m).U)
    v = CholA\icl
    return cholesky(Symmetric(v*(v'))).U


end



"""
	QInv_B_A_Σ_A(Cholz, Ktz, Ktt, Q)
## Arguments
	- Cholq::UpperTriangular:		The cholesky decomposition of the inducing points
	- Cholz::UpperTriangular:		Cholesky decomposition of Kzz
	- Ktz::AbstractMatrix:			Ktz
	- Ktt::AbstractMatrix:			Ktt
	- Q::Real



	# -1/2 * Q^{-1} trace( Bt-1 + At-1 Σ At-1^T)

"""
function QInv_B_A_Σ_A(Cholq::UpperTriangular, Cholz::UpperTriangular, Ktz::AbstractMatrix, Ktt::AbstractMatrix, Q::Real)
	Atm1 = get_At(Cholz, Ktz)
	Btm1 = get_Bt(Cholz, Ktz, Ktt)
	-0.5 *1/Q *sum(diag(Btm1 + A_K_A(Cholq, Atm1)))
end
# Some zygote stuff.. Returns a nxn identity matrix, where A = nxm
@nograd get_IdMat(A::AbstractMatrix) =Matrix{eltype(A)}(I, size(A)[1], size(A)[1])


"""
    CholInv(Chol::UpperTriangular)
## Arguments
    - Chol::UpperTriangular:     The cholesky decomposition.

Returns the inverse of A, where A = Chol^T Chol

"""

function CholInv(Chol::UpperTriangular)
    Cinv = inv(Chol)
    return Symmetric(Cinv * Cinv')
end



"""
    KL(mu::AbstractVector, mz::AbstractVector, Cholu::UpperTriangular, Cholz::UpperTriangular)
## Arguments
    - mu::AbstractVector            The mean of the first mv normal
    - mz::AbstractVector            The mean of the second mv normal
    - Cholu::UpperTriangular        The cholesky decompostion of the first mv normal
    - Cholz::UpperTriangular:       The cholesky decompostion of the second mv normal

Computes the Kullback leibler divergence.
Reference Rassmussen Appendix.
"""

function KL(mu::AbstractVector, mz::AbstractVector, Cholu::UpperTriangular, Cholz::UpperTriangular)
    kl = 0.0
    # Cholz : Cholesky decompostion of Σz and Cholu equivalent...
    # 1/2 logdet(Σz Σu^{-1})

    kl +=  logdet(Cholz)
	kl += -logdet(Cholu)
    # Innermatrix form the trace part = ((mu - mz)*(mu -mz^T) + Σu - Σz)
    innermat = ( mu .- mz)*(mu .- mz)' + Cholu'*Cholu - Cholz'*Cholz
    kl += +1/2 * sum(diag(Kinv_A(Cholz, innermat)))
    return kl
end





"""
 	getSigmaandMu(Mu1::AbstractMatrix,Mu2::AbstractArray{<:Real, 3})

## Arguments
	- Mu1::AbstractMatrix:
	- Mu2::AbstractArray{<:Real, 3}:


Compute the mean and the covariance of the inducing points given Mu1, Mu2. See Frigolas PHD thesis for the derivation.
The formulas are: Sigma = (-2 Mu2)^{-1}, mu = Sigma* Mu1

"""

@nograd get_Σ_μ_VI(Mu1,Mu2) = begin
	Σuq = Vector{UpperTriangular{<:Real}}(undef, size(Mu2)[3])
	μuq =  Vector{Vector{<:Real}}(undef, size(Mu2)[3])

	@views for dim in 1:size(Mu2)[3]

		# The cholesky decomposition is then:
		Su = cholesky(CholInv(cholesky(Hermitian(-2 .* Mu2[:,:,dim])).U)).U
		# Now we have Su' * Su is Σ.
		Σuq[dim] =  Su
		# This is Σ* Mu1
		μuq[dim] =  Su'*(Su* Mu1[:,dim])
	end
	return μuq, Σuq
end

"""
	get_At(Cholz::UpperTriangular, Ktz::AbstractMatrix)

## Arguments
	- Cholz::UpperTriangular:
	- Ktz::AbstractMatrix:

Compute At from Frigolas PHD thesis. At = Ktz Kzz^{-1} = (Kzz^{-1} Kzt)^T

"""
function get_At(Cholz::UpperTriangular, Ktz::AbstractMatrix)
	return Kinv_A(Cholz, Ktz')'
end


"""
	get_Bt(Cholz::UpperTriangular, Ktz::AbstractMatrix, Ktt::AbstractMatrix)

## Arguments
	- Cholz::UpperTriangular:
	- Ktz::AbstractMatrix:
	- Ktt::AbstractMatrix:

Compute Bt from Frigolas PHD thesis. Bt =Ktt - Ktz Kzz^{-1} Kzt

"""
function get_Bt(Cholz::UpperTriangular, Ktz::AbstractMatrix, Ktt::AbstractMatrix)
	return Symmetric(Ktt .- A_Kinv_A(Cholz, Ktz'))
end




"""
     update_chol!(chol, Kttq, Ktt0)

## Arguments

     - chol::UpperTriangular:    The cholesky decomposition of K_{0:t-1}+  I*Q
     - Kttq::AbstractMatrix:     The covariance matrix K_tt + I*Q
     - Ktt0::AbstractMatrix:     The covariance matrix Kt,0:t-1


Returns the updated cholesky decomposition of K_{0:t} + I*Q

"""



## Function for the update of the Cholesky decomposition - only increasing
function update_chol(chol::UpperTriangular, Kttq::AbstractMatrix, Ktt0::AbstractMatrix)
    #error(string(size(chol))*" "*string(size(Ktta))*" "*string(size(Ktt0)))
    S21 = Ktt0/chol
    S22 = cholesky(Kttq - S21*permutedims(S21,(2,1))).U
    S12 =  permutedims(S21,(2,1))
    return UpperTriangular([ chol S12; S21 S22])
end


"""
     cholesky_up_down_date!(chol, Kttful, Q, i)

## Arguments

     - chol::UpperTriangular:       The cholesky decomposition of K_{0:t-1,t:T - reference traj }+  I*Q
     - Kttfull::AbstractMatrix:     The covariance matrix K_t,0:T-1
     - Q::AbstractMatrix:           The process noise
     - i::Int:                      i = t
     - InputDim:                    This would only be relevant if we have a multi output gp.

Exchanges the i-th column and row of chol wiht Ktt + [0,...,0,1,0...0] Q

"""

function cholesky_up_down_date!(chol::UpperTriangular, Kttful::AbstractMatrix, Q::AbstractMatrix, i::Int, InputDim::Int = 1)
    @views if i == 1
        # There is no l11
        # First do the downdate
        ## https://www.ucg.ac.me/skladiste/blog_10701/objava_23569/fajlovi/cholesky.pdf
        S33 = Cholesky(chol[InputDim+1:end, InputDim+1:end],'U',0)
        S23 = chol[1:InputDim, InputDim+1:end]

        for idim in 1:InputDim
            lowrankupdate!(S33, Array(S23[idim,:]))
        end
        # Now we need to do the update

        S22 = cholesky(Kttful[:,1:InputDim]+Q).U
        S23 = permutedims(S22,(2,1))\(Kttful[:,InputDim+1:end])
        for idim in 1:InputDim
            lowrankdowndate!(S33,Array(S23[idim,:]))
        end
        # If we get an error here, we know that something is wrong.
        chol[:,:] =  UpperTriangular([S22 S23; permutedims(S23,(2,1)) S33.U])[:,:]
    else
        # There is no l11
        # First do the downdate
        S11 = chol[1:(i-1)*InputDim,1:(i-1)*InputDim]
        S13 = chol[1:(i-1)*InputDim,i*InputDim+1:end]
        S33 = Cholesky(chol[i*InputDim+1:end,i*InputDim+1:end],'U',0)
        S23 = chol[(i-1)*InputDim+1:i*InputDim,i*InputDim+1:end]

        for idim in 1:InputDim
            lowrankupdate!(S33,Array(S23[idim,:]))
        end
        # Now we need to do the update
        S11 = S11
        A12 = permutedims(Kttful[:,1:(i-1)*InputDim], (2,1))
        A22 = Kttful[:,(i-1)*InputDim+1:i*InputDim] +Q
        A23 = Kttful[:,i*InputDim+1:end]
        S13 = S13
        S12 = permutedims(S11,(2,1))\A12
        S22 = cholesky(A22 - permutedims(S12, (2,1))*S12).U
        S23 = permutedims(S22,(2,1))\(A23 - permutedims(S12, (2,1))*S13)

        for idim in 1:InputDim
            lowrankdowndate!(S33,Array(S23[idim,:]))
        end
        # If we get an error here, we know that something is wrong.
        chol[:,:] = UpperTriangular([S11 S12 S13;
            permutedims(S12, (2,1)) S22 S23;
            permutedims(S13, (2,1)) permutedims(S23, (2,1)) S33.U])[:,:]
    end

end



# """
#     Q_A_Kinv_A(Chol::UpperTriangular, A::AbstractMatrix)
#
# ## Arguments
#     - Chol:     The cholesky decomposition of K nxn
#     - A:        A matrix mxn
#
#
# Returns the inverse of I*Q+ A K^{-1} A^T.
# The formula is taken from Rasmussen appendix A.
#
# """
#
#
#
# function Q_A_Kinv_A(Chol::UpperTriangular, A::AbstractMatrix, Q::Real)
#     Zinv = get_IdMat(A) .* 1/Q
#     Winv = CholInv(Chol)
#     # = (W^{-1} + A^T Z^{-1} A)
#     cholbracket = cholesky(Symmetric(Winv + A'*A .* 1/Q)).U
#     # A *  (W^{-1} + A^T Z^{-1} A)^{-1} A^T
#     ABA = A_Kinv_A(cholbracket, A')
#
#     # Zinv - Zinv ABA Zinv
#     return Symmetric(Zinv - 1/Q^2 .* ABA)
# end
