

module GPSSM

        using Zygote
        using Zygote: @nograd
        using AbstractMCMC
        using AbstractMCMC: AbstractModel, AbstractSampler
        using Bijectors
        using Turing
        using DynamicPPL
        using DynamicPPL: Selector
        using LinearAlgebra
        using StatsFuns: logsumexp, softmax!
        using Flux
        using NamedTupleTools
        using Distributions
        using Random
        using MCMCChains: Chains
        using Stheno
        using ForwardDiff
        using FillArrays
        import NamedTupleTools: namedtuple
        using MacroTools
        using Optim
        using StatsFuns


        #The GPSSM Model
        abstract type GPSSMInstance end
        # A particle filter instance
        abstract type PFExtras end
        abstract type EmissionType <: PFExtras end
        abstract type PFStorage end
        # This was removed in AbstractMCMC...
        abstract type AbstractTransition end

        # GPSSM Algorithm
        abstract type AbstractGPSSMAlgorithm  end

        # Algorithms
        # Hyper Parameters
        abstract type AbstractHPAlgorithm<: AbstractGPSSMAlgorithm end
        abstract type AbstractFBAlgorithm<: AbstractHPAlgorithm end
        abstract type AbstractVIAlgorithm<: AbstractHPAlgorithm end
        abstract type AbstractEMAlgorithm<: AbstractHPAlgorithm end


        # Hyper Parameter specifications
        abstract type AbstractGPSpec end

        # Models
        # Latent space models
        abstract type AbstractLSAlgorithm<: AbstractGPSSMAlgorithm end
        abstract type AbstractLSSMCAlgorithm<:AbstractLSAlgorithm end
        abstract type AbstractLSPGAlgorithm <: AbstractLSAlgorithm end

        abstract type AbstractLSPGFAAlgoirthm <: AbstractLSPGAlgorithm end

        #Hyper parameters
        abstract type AbstractHPModel<: AbstractModel end
        abstract type AbstractHPFBModel<: AbstractHPModel end
        abstract type AbstractHPEMModel<: AbstractHPModel end
        abstract type AbstractHPVIModel<: AbstractHPModel end
        # Samplers
        abstract type AbstractGPSSMSampler <: AbstractSampler end
        abstract type AbstractHPSampler <: AbstractGPSSMSampler end
        # Fully Bayesian Sampler
        abstract type AbstractHPFBSampler <:AbstractHPSampler end

        # Hyper Parameter Optimiser. In this code, we threat Optimiser as sampler.
        abstract type AbstractHPOptimiser <:AbstractHPSampler end # Well, we have to make it like this
        abstract type AbstractHPEMOptimiser<: AbstractHPOptimiser end
        abstract type AbstractHPVIOptimiser<: AbstractHPOptimiser end

        #Latent State Sampler
        abstract type AbstractLSSampler <: AbstractSampler end
        abstract type AbstractBaseLSModel <: AbstractModel end

        # Transition for the AbstractMCMC Chain

        abstract type InferenceModel end
        abstract type MarkovianInferenceModel <:InferenceModel end

        abstract type AbstractPFInstance end


        abstract type GPSSMEmission end




        # Some Turing Things
        const TypedVarInfo = VarInfo{<:NamedTuple}
        const ConstSelector = Selector(:constant)
        const TrainableSelector = Selector(:trainable)



        # We need to fix a bug in namedtuple!
        NamedTupleTools.namedtuple(t::typeof(Tuple([])),v::typeof(Tuple([]))) = NamedTuple()

        include("Utilities/turing_interface.jl")
        include("Utilities/generalutilities.jl")
        include("Utilities/gaussianprocess.jl")


        include("Models/HyperParams/hyperparamspec.jl")
        include("Models/gpinstance.jl")
        include("Algorithms/LatentStates/algorithms.jl")
        include("Models/LatentStates/inferencemodel.jl")
        include("Utilities/predict.jl")
        include("Models/HyperParams/optimisationmodel.jl")
        include("Algorithms/HyperParams/basealg.jl")
        include("Algorithms/LatentStates/implementations.jl")


        include("Models/LatentStates/fixedgp.jl")
        include("Models/LatentStates/fullgp.jl")
        include("Models/LatentStates/hybridvigp.jl")
        include("Models/LatentStates/sparsegp.jl")


        include("Algorithms/HyperParams/em.jl")
        include("Algorithms/HyperParams/vi.jl")
        include("Algorithms/HyperParams/pgas.jl")
        include("Algorithms/HyperParams/constanthp.jl")

        include("Models/HyperParams/fullybayesianmodel.jl")
        include("Algorithms/LatentStates/step.jl")
        include("AbstractMCMC_interface.jl")

        include("Utilities/transitions.jl")



        DEFAULT_PREDICTION_SPEC = (NTest = 30, MTest = 200, woburnin = 20, burninpred = 200, Npretrain = 10, NBoot=100)
        DEFAULT_VAR_SCALE_VALUES = (var = 1.0, scale = 1.0)
        DEFAULT_GRADIENT_CLIPPING = 100.0

        export  sample,
                predict
end
