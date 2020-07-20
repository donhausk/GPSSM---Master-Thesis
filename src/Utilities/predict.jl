

abstract type TransitionExtras end

"""
    TrainedTransition

## Fields
    - I:                The InferenceModel type.
    - kernel_ls:        List of kernels, this list contains several elements for the fully bayesian case.
    - mean_ls:          List of means, this list contains several elements for the fully bayesian case.
    - sizes:            The latent states dimension of the GP.
    - Qs:               List of Qvalues.
    - extras:           Specific attributes for the InfrenceModel


Trained transition for multi step ahead predicition.

"""


struct TrainedTransition{I<: InferenceModel, GI <: GPSSMInstance, KL, ML, T, L, E<: TransitionExtras}
    instance::GI
    kernel_ls::KL
    mean_ls::ML
    dim::T
    Qs::L
    extras::E
end



"""
    predict(XandUstar, instacnce::GPSSMInstance,
        spec::AbstractGPSpec,
        alghp::AbstractHPAlgorithm,
        pfalg::AbstractLSAlgorithm,
        inputs_train::NamedTuple,
        args::Vector{<:NamedTuple};
        wo_burnin, kwargs... )

## Arguments

    - XandUstar::A:                     The inputs for the predictions
    - instacnce::GPSSMInstance:         The GPSSM Instance
    - spec::AbstractGPSpec:             The GP Specification
    - alghp::AbstractHPAlgorithm:       The Hyper Parameters Algorithm
    - alglt::AbstractLSAlgorithm:       The Latentstates Algoirthm
    - inputs_train::NamedTuple:         The inputs of the training procedure.
    - args::Vector{<:NamedTuple}:       The return arguments from the sample chain.
    - wo_burnin = 0:                    We only use the wo_burnin last samples for the full gp case. By default, we use all of the samples


Predict f(xandu*)

"""


# Predict the FullyBayesian setting
function predict(
    XandUstar::AbstractArray,
    instance::GPSSMInstance,
    spec::AbstractGPSpec,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs_train::NamedTuple,
    args::AbstractVector;
    woburnin = 0,
    kwargs...
)
    alglt = BaseLSAlgorithm(pfalg, alghp)

    trainedtrans =  TrainedTransition(instance,spec, alghp, alglt, inputs_train, args; woburnin = woburnin, kwargs...)
    return predict(XandUstar, trainedtrans)
end
# Predict the FullyBayesian setting
function predict(
    XandUstar::AbstractArray,
    instance::GPSSMInstance,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs_train::NamedTuple,
    args::AbstractVector;
    woburnin = 0,
    kwargs...
)
    spec = get_gpspec_from_instance(instance)
    return predict(XandUstar,instance, spec, alghp, pfalg, inputs_train, args; woburnin = woburnin, kwargs = kwargs)
end




"""
    predict(XandUstar, instacnce::GPSSMInstance, spec, alghp, alglt, inputs_train, args; burnin=burnin, kwargs... )

## Arguments

    - XandUstar::A:                                        The inputs for the predictions
    - instacnce::GPSSMInstance:                            The GPSSM Instance
    - spec::AbstractGPSpec:                                The GP Specification
    - alghp::AbstractHPAlgorithm:                          The Hyper Parameters Algorithm
    - pfalg::AbstractLSAlgorithm:                          The Latentstates Algoirthm
    - inputs_train::NamedTuple:                            The inputs of the training procedure.
    - inputs_predict::NamedTuple:                          The inputs for the mult step ahead predictions
    - args::Vector{<:NamedTuple};                          The return arguments from the sample chain.
    - Xstart::Union{AbstractArray, Nohting} = nothing:     In case we want to fix the starting point
    - argsystart::Union{Vector{<:NamedTuple}, Nothing}:    For the fully bayesian setting, we want might have different arguments for the emission modell as we use a list of hyper paramters.
    - Npredahead = 0:                                      The amount of multi step ahead predictions, by default the length of input_predict
    - Mpred = DEFAULT_PREDICTION_SPEC[2]:                  The amount of MC samples used for multi step ahead prediction
    - woburnin= DEFAULT_PREDICTION_SPEC[3]:                We only use the wo_burnin last samples for the full gp case. By default, we use all of the samples
    - burninpred = DEFAULT_PREDICTION_SPEC[4]:             The amount of burnin samples for the pre training
    - Npredtrain = DEFAULT_PREDICTION_SPEC[5]:             The amount of steps for the pretraining, if Xstart is set, we do not pretrain at all
    - NBoot = DEFAULT_PREDICTION_SPEC[6]:                  The amount of bootstrap samples used to compute the MSE.


Run Npredtrain initial training steps and then do multi step ahead prediction. If Xstar is set, we do not make any pretraining at all.


"""




function predict_ms(
    instance::GPSSMInstance,
    spec::AbstractGPSpec,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs_train::NamedTuple,
    inputs_predict::NamedTuple,
    args::AbstractVector;
    Xstart::Union{AbstractArray, Nothing} = nothing,
    argsystart::Union{Vector{<:NamedTuple}, Nothing} = nothing,
    Npredahead = 0,
    Mpred = DEFAULT_PREDICTION_SPEC[2],
    woburnin= DEFAULT_PREDICTION_SPEC[3],
    burninpred = DEFAULT_PREDICTION_SPEC[4],
    Npredtrain = DEFAULT_PREDICTION_SPEC[5],
    NBoot = DEFAULT_PREDICTION_SPEC[6],
    kwargs...
)
    if Npredahead > 0
        # Get a trained tranistion struct....
        Npred = Npredahead
    else
        Npred = inputs_predict[:sizes][2]+1
    end


    alglt = BaseLSAlgorithm(pfalg, alghp)
    # Extract the trained transisiton
    trainedtrans =  TrainedTransition(instance, spec, alghp, alglt, inputs_train, args; woburnin = woburnin, kwargs...)
    if argsystart === nothing
        argsystart = [args[end][:args_θy]]
    else
        # For every Xstart, we have a specific argsystart...
        @assert typeof(argsystart) <: AbstractVector && typeof(Xstart) <: AbstractVector && length(argsystart) == length(Xstart) "Xstart and argsystart must have the same length"
    end


    if Xstart === nothing
        # We use the last set of hyper parameters.
        predictionhp = ConstantHP(instance, args[end], trainedtrans)

        @views inputs_pretrain = (sizes = (inputs_predict[:sizes][1],Npredtrain, inputs_predict[:sizes][3]), observations = inputs_predict[:observations][:, 1:Npredtrain, :],
            controls=inputs_predict[:controls][:,1:Npredtrain+1,:])

        # Pretrain...
        chn, vals, times_pred_burnin = AbstractMCMC.sample(instance, spec, predictionhp , pfalg, inputs_pretrain, Mpred+burninpred; progress=false, kwargs...)
        # We have a problem with 4D Xvalues...
        Xs = [v for i = 1+burninpred:Mpred+burninpred for v in flatten_to_3D(vals[i][:X]) ]
    else
        # We have a problem with 4D Xvalues...
        Xs = [v  for i = 1:length(Xstart) for v in flatten_to_3D(Xstart[i])]
    end
    # We need some assertions here...
    Xnew = Array{eltype(Xs[1]), 4}(undef, size(Xs[1])[1], Npred+1 ,size(Xs[1])[3], Mpred)
    Mses = Array{Float64}(undef,  size(inputs_predict[:observations])[1], Npred, size(Xs[1])[3], Mpred)
    Logps = Array{Float64}(undef,  Npred, size(Xs[1])[3], Mpred)
    ltrained = size(Xs[1])[2]
    Y = inputs_predict[:observations]
    U = inputs_predict[:controls]
    # Run MC predictions
    @views for j = 1:Mpred

        key = max(1, length(Xs)-(j% length(Xs))) # for some special cases
        Xnew[:,1:ltrained,:, j] = Xs[key]

        for k = 1:size(Xs[1])[3]
            xtn = Xnew[:,ltrained,k,j]
            for i = 1:(Npred+1 -ltrained )
                # Run Mc samples
                mu, sigma = predict_avg(vcat(xtn,U[:, i-1 + ltrained,k]), trainedtrans)
                xtn = rand(MvNormal(mu, Diagonal(sigma)))
                # Set next state
                Xnew[:,ltrained+i,k,j] = xtn
            end
        end
    end

    # Evaluate
    @views for j = 1:Mpred
        key = max(1, length(Xs)-(j% length(Xs))) # For some special cases....
        for k = 1:size(Xs[1])[3]
            for i = 2:Npred # We neglect the first observation....
                # Compute statistics
                ms,lp = get_stats(instance, Y[:,i-1,k], Xnew[:,i,k,j], argsystart[min(length(argsystart), key)]; NBoot = NBoot)
                Mses[:,i-1,k,j] = ms
                Logps[i-1,k,j] = lp
            end
        end
    end


    return Xnew, Mses, Logps
end





function predict_ms(
    instance::GPSSMInstance,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs_train::NamedTuple,
    inputs_predict::NamedTuple,
    args::AbstractVector;
    Xstart::Union{AbstractArray, Nothing} = nothing,
    argsystart::Union{Vector{<:NamedTuple}, Nothing} = nothing,
    Npredahead = 0,
    Mpred = DEFAULT_PREDICTION_SPEC[2],
    woburnin= DEFAULT_PREDICTION_SPEC[3],
    burninpred = DEFAULT_PREDICTION_SPEC[4],
    Npredtrain = DEFAULT_PREDICTION_SPEC[5],
    NBoot = DEFAULT_PREDICTION_SPEC[6],
    kwargs...
)
    predict_ms(instance,get_gpspec_from_instance(instance),alghp,pfalg,inputs_train, inputs_predict,args;
        Xstart = Xstart, argsystart = argsystart,Npredahead = Npredahead,Mpred = Mpred,woburnin = woburnin,burninpred = burninpred,Npredtrain= Npredtrain ,NBoot = NBoot ,kwargs...)
end

"""
    get_stats(instance, Y, X, args_θy;  NBoot = 100)

## Arguments
    - instance::GPSSMInstance:      The instance of the GPmodel
    - Y::AbstractVector:            The observation
    - X::AbstractVector:            The latent states
    - args_θy::NamedTuple:          The args_θy hyper parameters
    - NBoot::Integer:               Amount of Bootstrap samples

Computes the MSE := E[ (y_t - Y)^2] where we average over the dimensions with y_t ~ p(y_t| X, args_θy)
and also computes LL := p(Y|X, args_θy)



"""

function  get_stats(instance::GPSSMInstance, Y::AbstractVector, X::AbstractVector, args_θy::NamedTuple;  NBoot::Integer = 100)
    distxy, yargs = get_distxy_and_args(instance, X, args_θy...)

    distb = distxy(yargs...)
    # if typeof(distb) <: Normal
    #     # We can compute the MSE exaclty....
    #     mse = (mean(distb) .- Y).^2 + var(distb)
    # elseif  distxy <: MvNormal && size(Y)[1] == 1
    #     mse = ((mean(distb) .- Y).^2 .+ var(distb))[1,1]
    # else
    obs = rand(distb, NBoot) # Bootstrap mse
    mse = mean((obs .- Y).^2, dims=2)
    # end
    logpobs = logpdf(distxy(yargs...), Y)::Float64
    return mse, logpobs
end

@inline flatten_to_3D(X::AbstractArray{<:Real,3}) = [X]
@inline flatten_to_3D(X::AbstractArray{<:Real,4}) =  @views [X[:,:,1,:]]



"""

    predict(XandUstar, trainedtrans)

## Arguments
    - XandUstar::AbstractMatrix:            A Matrix of dim+dimc x N containg the positions for the predictions of f(x*)
    - trainedtrans::TrainedTransition:      An trainedtransition instance

Returns a Array containing the predictions based on the trainedtrans instance.


"""

function predict(XandUstar::AbstractMatrix, trainedtrans::TrainedTransition{I}) where {I<: FullGPInference}
    #Now lets predict!
    predictΣ = Array{get_latent_type(), 3}(undef,length(trainedtrans.kernel_ls),trainedtrans.dim[1], size(XandUstar)[2])
    predictμ = Array{get_latent_type(), 3}(undef, length(trainedtrans.kernel_ls),trainedtrans.dim[1], size(XandUstar)[2])

    @views for j = 1:length(trainedtrans.kernel_ls)
        X = trainedtrans.extras.X[j]
        XandU_j = trainedtrans.extras.XandU[j]
        for k = 1:trainedtrans.dim[1]
            Q= trainedtrans.Qs[j][k]
            kernel = trainedtrans.kernel_ls[j][k]
            mean_f = trainedtrans.mean_ls[j][k]
            chol = trainedtrans.extras.Chols[j][k]
            mu = trainedtrans.extras.Mus[j][k]
            for i = 1:size(XandUstar)[2]

                Kxstarx = Kxxt(trainedtrans.instance, kernel, XandUstar[:,i:i], XandU_j)
                Kxstarxstar = Kxx(trainedtrans.instance, kernel, XandUstar[:,i:i])

                predictΣ[j,k,i] = get_prediction_transition_cov(I, Kxstarx, Kxstarxstar .+Q, chol )[1]
                predictμ[j,k,i] = get_prediction_transition_mean(I, vec( mx(trainedtrans.instance, mean_f, XandUstar[:,i:i])), mu,  Kxstarx)[1]
            end
        end
    end
    return (predictμ, predictΣ)
end


function predict(XandUstar::AbstractMatrix, trainedtrans::TrainedTransition{IM}) where IM <:MarkovianInferenceModel   #Now lets predict!
    #Now lets predict!
    predictΣ = Array{get_latent_type(), 3}(undef, 1, trainedtrans.dim[1], size(XandUstar)[2])
    predictμ = Array{get_latent_type(), 3}(undef, 1, trainedtrans.dim[1], size(XandUstar)[2])
    @views for k = 1:trainedtrans.dim[1]
        kernel = trainedtrans.kernel_ls[k]
        mean_f = trainedtrans.mean_ls[k]
        cholu = trainedtrans.extras.CholZ[k]
        sigma = trainedtrans.extras.Sigma[k]
        muu =trainedtrans.extras.Mus[k]
        Z_Ind = trainedtrans.extras.Z_Ind

        Q= trainedtrans.Qs[k]
        for i = 1:size(XandUstar)[2]
            Kxstarx = Kxxt(trainedtrans.instance, kernel, XandUstar[:,i:i], Z_Ind)
            Kxstarxstar = Kxx(trainedtrans.instance, kernel, XandUstar[:,i:i])
            predictΣ[1,k,i] = get_prediction_transition_cov(IM, Kxstarx, Kxstarxstar .+Q, cholu , sigma)[1]
            predictμ[1,k,i] = get_prediction_transition_mean(IM,  mx(trainedtrans.instance, mean_f, XandUstar[:,i:i]), muu,  Kxstarx)[1]

        end
    end
    return (predictμ, predictΣ)
end



"""
    predict_avg(xandustar, trainedtrans)

## Arguments
    - xandustar::AbstractVector:            A Vector of dimension dim+dimc
    - trainedtrans::TrainedTransition:      An trainedtransition instance

Returns the prediction for f(x*). The average indicates that we average in the full gp case


"""

function predict_avg(xandustar::AbstractVector, trainedtrans::TrainedTransition)

    #Now lets predict!
    XandUstar = cat(xandustar, dims=2) # We need to have a matrix...

    predictμ, predictΣ = predict(XandUstar, trainedtrans)

    return vec(mean(predictμ, dims=1)), vec(mean(predictΣ, dims=1))
end





"""
    FullGPTransitionExtras

## Arguments
    - X:        The sampled latent states
    - XandU:    The sampled latent states concatenated with U
    - Chols:    The cholesky decompositions of Kxx +I*Q
    - Mus:      This (Kxx +I*Q)^{-1}(X_2:T-m_0:T-1)


The FullGP prediction extras.

"""

struct FullGPTransitionExtras{XL, XANDUL, U, MS} <: TransitionExtras where {XL <: AbstractVector, XANDUL <: AbstractVector, Chols <: AbstractVector, Mus <: AbstractVector}
    X::XL
    XandU::XANDUL
    Chols::U
    Mus::MS
end


"""
    SparseTransistionExtras

## Arguments
    - CholsZ:        The cholesky decomposition of Kzz
    - CholAZA:       The cholesky decomposition of Kzz^{-1}Σ Kzz^{-1}
    - Mus:           Kzz^{-1}( μ - mz)
    - Z_Inds::MZ     The indcuing points


The Sparse GP transistion
"""

struct SparseTransistionExtras{MZ, U,UA, MS} <: TransitionExtras
    CholZ::U
    Sigma::UA
    Mus::MS
    Z_Ind::MZ
end











"""
    TrainedTransition{I}(XandUstar, instacnce::GPSSMInstance,
        spec::AbstractGPSpec,
        alghp::AbstractHPAlgorithm,
        pfalg::AbstractParticleFilter,
        inputs_train::NamedTuple,
        args::Vector{<:NamedTuple};
        wo_burnin, kwargs... )

## Arguments

    - instacnce::GPSSMInstance:         The GPSSM Instance
    - spec::AbstractGPSpec:             The GP Specification
    - alghp::AbstractHPAlgorithm:       The Hyper Parameters Algorithm
    - alglt::AbstractLSAlgorithm:       The Latentstates Algoirthm
    - inputs_train::NamedTuple:         The inputs of the training procedure.
    - args::Vector{<:NamedTuple}:       The return arguments from the sample chain.
    - wo_burnin = 0:                    We only use the wo_burnin last samples for the full gp case. By default, we use all of the samples


Returns an instance of TrainedTransition{I}.

"""


# Predict the FullyBayesian setting
function TrainedTransition(
    instance::GPSSMInstance,
    spec::AbstractGPSpec,
    alghp::AbstractHPAlgorithm,
    algls::BaseLSAlgorithm{I, A},
    inputs_train::NamedTuple,
    args::AbstractVector;
    woburnin = 0,
    kwargs...
) where {I<:FullGPInference, A<: AbstractLSAlgorithm }

    U = inputs_train[:controls]
    woburnin = woburnin == 0 ?  length(args) : woburnin

    arglist = args[max(1, length(args)-woburnin)+1:end]
    sizes = size(arglist[1][:X])
    kernel_ls = []
    mean_ls = []

    Qs = Vector{Vector{get_latent_type()}}()
    Chols = Vector{Vector{UpperTriangular{get_latent_type(),Matrix{get_latent_type()}}}}()
    Mus = Vector{Vector{Vector{get_latent_type()}}}()
    XL = Vector{Array{get_latent_type(),2}}()
    XandUL =Vector{Array{get_latent_type(),2}}()
    @views for k = 1:length(arglist)

        # Fully bayesian infrence requires to use different hyper paramters....
        if A <: AbstractFBAlgorithm
            kernel_a = arglist[k][:kernel_args]
            mean_a = arglist[k][:mean_args]
            Q = arglist[k][:Q][:]
        else
            # In the frequentist approach, we take the last ones.
            kernel_a = arglist[end][:kernel_args]
            mean_a = arglist[end][:mean_args]
            Q = arglist[end][:Q][:]
        end

        kernels = get_kernels(instance, kernel_a)
        push!(kernel_ls, kernels)
        mean_fs = get_means(instance, mean_a)
        push!(mean_ls, mean_fs)
        # We only take the first one for the SMC case. This is because SMC is abused in a MCMC setting.
        X = arglist[k][:X][:,:,1,:]
        Xout = reshape(X[:,2:end,:],(size(X)[1],:))
        push!(XL, Xout)
        XandU = reshape(cat(X[:,1:end-1,:],U[:,1:end-1,:],dims=1),(size(X)[1]+size(U)[1],:))

        push!(XandUL,XandU )
        Chols_j = Vector{UpperTriangular{get_latent_type(),Matrix{get_latent_type()}}}()
        Mus_j = Vector{Vector{get_latent_type()}}()
        for j = 1:sizes[1]

            # This is cholesky ( K_0:T-1 +I *Q)
            Kx = Kxx(instance, kernels[j],XandU) .+ Q[j]
            chol = cholesky(Kx).U

            # This is ( K_0:T-1 +I *Q)^{-1} (X_1:T - m_0:T-1)
            mu = Kinv_A(chol, Xout[j,:]- mx(instance, mean_fs[j], XandU))

            push!(Chols_j, chol)
            push!(Mus_j, mu)


        end
        push!(Qs, Q)
        push!(Chols, Chols_j)
        push!(Mus, Mus_j)
    end

    extras =FullGPTransitionExtras(XL, XandUL, Chols, Mus)
    return TrainedTransition{I, typeof(instance), typeof(kernel_ls), typeof(mean_ls), typeof(sizes), typeof(Qs), typeof(extras)}(instance, kernel_ls, mean_ls, sizes, Qs, extras)
end



function TrainedTransition(
    instance::GPSSMInstance,
    spec::AbstractGPSpec,
    alghp::AbstractHPAlgorithm,
    algls::BaseLSAlgorithm{I, A},
    inputs_train::NamedTuple,
    args::AbstractVector;
    woburnin = 0,
    kwargs...
) where {I<:MarkovianInferenceModel, A<: AbstractLSAlgorithm }

    U = inputs_train[:controls]
    kernel_ls = []
    mean_ls = []
    sizes = inputs_train[:sizes]
    arglist = args



    # In the frequentist approach, we take the last ones.
    kernel_a = arglist[end][:kernel_args]
    mean_a = arglist[end][:mean_args]
    Q = arglist[end][:Q][:]

    kernels = get_kernels(instance, kernel_a)
    push!(kernel_ls, kernels)
    mean_fs = get_means(instance, mean_a)
    push!(mean_ls, mean_fs)
    Chols = Vector{UpperTriangular{get_latent_type(),Matrix{get_latent_type()}}}()
    Mus = Vector{Vector{get_latent_type()}}()
    CholsAZA = Vector{UpperTriangular{get_latent_type(),Matrix{get_latent_type()}}}()
    Z_Ind = arglist[end][:Z_Ind]

    μ, Σ = extract_mu_sigam(algls, alghp, arglist[end])


    @views for j = 1:sizes[1]
        Cholz = cholesky(Kxx(instance, kernels[j],Z_Ind)).U
        Sigma = Σ[j]
        # This is Kzz^{-1} *μu
        mu = Kinv_A(Cholz,μ[j] .- mx(instance, mean_fs[j], Z_Ind))

        push!(Chols, Cholz)
        push!(Mus, mu)
    end


    extras = SparseTransistionExtras(Chols,Σ , Mus, Z_Ind)
    return TrainedTransition{I, typeof(instance), typeof(kernels), typeof(mean_fs), typeof(sizes), typeof(Q), typeof(extras)}(instance, kernels, mean_fs, sizes, Q, extras)
end


"""

    extract_mu_sigam(algls, alghp, arglist)

"""
function extract_mu_sigam(algls::BaseLSAlgorithm, alghp::AbstractHPAlgorithm, args::NamedTuple)
    μ = args[:mu]
    Σ = args[:Sigma]
    return μ, Σ
end
function extract_mu_sigam(algls::BaseLSAlgorithm{<:HybridVIInference}, alghp::AbstractHPAlgorithm, args::NamedTuple)
    return get_Σ_μ_VI(args[:Mu1], args[:Mu2])
end
