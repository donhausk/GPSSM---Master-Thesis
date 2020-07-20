# GP-SSM
This is a package for high quality implementations for GPSSM learning and inference algorithms including:

1. R. Frigola, F. Lindsten,T. B. Schoen and C. E. Rasmussen. (2013). Bayesian inference and learning in Gaussian process state-space modelswithparticleMCMC.InAdvancesinNeuralInformationProcessing Systems 26 (NIPS), pp. 3156-3164.
2. R. Frigola, Y. Chen and C. E. Rasmussen. (2014). Variational Gaussian Process State-Space Models, in Advances in Neural Information Processing Systems (NIPS).
3. R. Frigola, F. Lindsten, T. B. Schoen and C. E. Rasmussen. (2014). Identiﬁcation of Gaussian process state-space models with particle stochastic approximation EM. In 19th World Congress of the International Federation of Automatic Control (IFAC), pp. 4097-4102.
# Implementation
The package is written in Julia and based on the packages AdvancedPS, Turing (Branch Compiler3.0), AdvancedHMC and Stheno. The reason for not choosing PyCall and GPFlow is because PyCall is not compatible with ForwarDiff from Julia, which is used in AdvancedHMC.

# Useage

```julia

using Distributions
using Random
using GPSSM
import Test_Runner: @run_experiments, @getelementm, ExpData, storagepath,
    datapath, Gaussian1D, kinkf, get_test_specifications, generate_data, Narendra_li, NarendraLiParameters
using Test_Runner


###############################################################
# Generate Dataset
################################################################

T = Test_Runner.InvertedPendulum()
N = 1000 # Length of generated data
m = 2 # Amount of parallel chains
dataset = generate_data(T, datapath*"Somepath", [0.1]; N=N,m=2, normalise=true)


len = 50 # length used for training
inputs = (sizes = (T.latentdim,len,m), observations = dataset[2][1][:, 1:len,:], controls = dataset[3][:,1:len+1,:])

# Specify a GPSSM instance, which should be renamed to a GPSSM model
instance = GPSSM.LinearGaussianInstance(inputs[:sizes][1], size(inputs[:observations])[1], size(inputs[:controls])[1];
    mean= Stheno.ZeroMean(), kernel = Stheno.Matern32(), emission_noise = 0.1, init_var = Exponential(), init_scale = Exponential())

# The actual inference algorithm, returning a AbstractMCMC chain and some additional parameters needed for prediciton tasks
chn, predict_args, predtimes = sample(instance,ahp,alt,inputs,50)

# Predictions for the inputs rand(2,100)
predictions = GPSSM.predict(rand(2,100),instance, ahp, alt, inputs, predict_args)

# Multi step ahead predicitons
mspredictions = GPSSM.predict_ms(instance, ahp, alt, inputs, inputs, predict_args)

```


# Experiments

In order to run experiments, the following Macro is used:

```julia
using Distributed

@everywhere using Distributed: @distributed
println(nworkers())
@everywhere push!(LOAD_PATH, "/home/kongi/GPSSM/")
@everywhere push!(LOAD_PATH, "/home/kongi/GPSSM/Test_Runner/")
@everywhere using Random
@everywhere using Distributions
@everywhere using Distributed
@everywhere using AbstractMCMC
@everywhere using GPSSM
@everywhere using GPSSM: Zygote_MvNormal
@everywhere using Flux
@everywhere using Stheno
@everywhere using JLD2
@everywhere using Optim
@everywhere using LinearAlgebra
@everywhere using Turing
@everywhere using Test_Runner
@everywhere import Test_Runner: @run_experiments, @getelementm, ExpData, storagepath,
    datapath, Gaussian1D, kinkf, get_test_specifications, generate_data, Narendra_li, NarendraLiParameters, UnicycleHP
################################################################
# Model
################################################################

@everywhere  Expname = "Benchmarking"
###############################################################
# Gridsearch
###############################################################
@everywhere N,m = 500, 10
@everywhere Observation_Noise = [1/4^2]
@everywhere Iterations = 500
@everywhere Algorithms = ["HybridVI"]
@everywhere len = 150
@everywhere Rep = [1,2,3]
@everywhere on = Observation_Noise[1]
@everywhere nindp= 50
@everywhere Niter = 1000



task = @run_experiments storagepath*Expname*"/" storagepath*Expname*"/" NarendraLI( Algorithms, Rep) = begin

    #Make sure the data is generated.
    T = Test_Runner.Narendra_li()
    _,Yf,Uf,_,_,_ =  Test_Runner.generate_data(T, storagepath*"Data/"*"NarendraLi/", Observation_Noise; N=N,m=m, normalise = true, key_test = "benchmark")
    @data ~ ExpData(T.latentdim,Yf[1][:,1:len+30,1:1], Uf[:,1:len+1+30,1:1]; predictahead=30)
    j = @i

    @Niter ~4000
    @instance ~  GPSSM.NarendraLiInstance(; trainQ=true,
        mean= Stheno.ZeroMean(), kernel = Stheno.Matern32(), emission_noise = on, Z_Ind = rand(T.latentdim+T.cdim, nindp), trainZ_Ind = true)
    @ahp ~ GPSSM.HybridVIAlgorithm(n_grad = 5,optimiser= Flux.Descent(0.001), lag = 1, lrschedule = 1000, schedulerate = 0.5, burnin = 50, optimisermu1=Flux.Descent(0.05), optimisermu2 = Flux.Descent(0.05))
    @alt ~ GPSSM.SMC(50)
end
```

