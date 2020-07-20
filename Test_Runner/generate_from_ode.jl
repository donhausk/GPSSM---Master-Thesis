## This is a file used for linearisation of differential equtions.




"""
    ControlSystemDataSet

## Fields
    - x::Vector{<:Sym}:        A list of symbols of the states
    - u::Vector{<:Sym}:        A list of symbols of the controls
    - A::Matrix{<Sym}:         specified below
    - intevall_size::Float64:  The interval size for the discretisation
    - B::Matrix{<:Sym}         specified below
    - arguemnts::NamedTuple:   Additional arguements.
    - distraction::Function:  This is used to add some noise to the control.
We have dx/dt = A(x)*x + B(x) *u


    ControlSystemDataSet(x, u, F,  FB, inteval_size, arguments)


## Arguments
    - x::Vector{<:Sym}:                    A list of symbols of the states
    - u::Vector{<:Sym}:                    A list of symbols of the controls
    - F::Matrix{<Sym}:                     specified below
    - FB::Matrix{<:Sym}                    specified below
    - intevall_size::Float64:              The interval size for the discretisation
    - arguemnts::NamedTuple = NamedTuple() Additional arguements.
    - C::Union{Nothing,Matrix{<:Sym}}:     For greater generality. Default it is the identity matrix.
    - hyp::Vector{<:Sym}:                  A list of hyper paramters.
    - hyp_vals::Vector{<:Sym}:             And their values.
If tranform is false, we have C dx/dt = F(x)*x + FB(x)*x. Else, for greater generality, we assume that C*dx/dt = F(x)+FB(x)*u and,
in a first step, transfer the model to the form dx/dt = A(x)*x +B(x)*u.

"""

struct ControlSystemDataSet{ARGS<:NamedTuple} <: ModelParameters
    x::AbstractVector
    u::AbstractVector
    A::AbstractMatrix
    B::AbstractMatrix
    control::Bool
    interval_size::Float64
    arguments::ARGS
    distraction_signal::AbstractMatrix
    generate_distraction::Function
    omit_samples::Integer
    function ControlSystemDataSet( x::AbstractVector,
        u::AbstractVector,
        F::AbstractArray,
        FB::AbstractArray,
        interval_size::Float64 = 0.01,
        arguments::NamedTuple = NamedTuple();
        C::Union{Nothing,AbstractMatrix} = nothing,
        hyp::Vector{<:Sym} = Vector{<:Sym}(),
        hyp_vals::Vector = Vector(),
        generate_distraction::Function = () -> [0.0],
        control::Bool = true,
        linearise::Bool = false,
        linearise_around = nothing

    )
        eval = generate_eval((hyp, hyp_vals))
        if C === nothing
            C = Diagonal(ones(length(x)))
        else
            C = evaluatemat(C, eval)
        end
        FA = evaluatemat(F, eval)
        FB = evaluatemat(FB, eval)
        if typeof(FA) <: AbstractVector
            FA = transformA(FA, x)
        end
        invc = inv(C)
        Fadapted = invc*FA
        FadaptedB = invc*FB
        if length(size(FadaptedB)) ==1
            FadaptedB = reshape(FadaptedB, (:,1))
        end
        # if linearise
        #     linearise_around = linearise_around == nothing ? zeros(length(x)) : linearise_around
        #
        #     Fadapted = linearise_mat(Fadapted, x , linearise_around)
        #     FadaptedB = linearise_mat(FadaptedB, x ,linearise_around)
        # end

        distraction_signal = generate_distraction()
        new{typeof(arguments)}(x, u, Fadapted ,FadaptedB, control, interval_size, arguments, distraction_signal, generate_distraction, 50)
    end
end

function Gaussian(hp::ControlSystemDataSet, name::String = "ControlSystemDataSet")
    Nlt = length(hp.x)
    C = haskey(hp.arguments, :C) ? hp.arguments[:C] : Diagonal(ones(Nlt))
    b = haskey(hp.arguments, :b) ? hp.arguments[:b] : zeros(size(C)[1])
    f_obs(x, on, hp) = C*x +b + rand(MultivariateNormal(zeros(size(C)[1]),on))

    TestSetGen(name, cs_dataset_t, f_obs, cs_dataset_control, cs_dataset_init, Nlt, size(C)[1], length(hp.u), hp)
end

# linearise_mat(mat::AbstractMatrix,  x::Vector{<:Sym}, vals::Vector{<:Sym}) = mat
# function linearise_mat(mat::Matrix{<:Sym}, x::Vector{<:Sym}, vals::Vector{<:Sym})
#     ret = Matrix{Sym}( undef, size(mat)[1], size(mat)[2])
#     for i = 1:size(mat)[1]
#         for j = 1:size(mat)[2]
#             ret[i,j] = linearise_eq(mat[i,j], x, vals)
#         end
#     end
# end





# Get a new distraciton signal
function reset_inputs!(hp::ControlSystemDataSet)
    hp.distraction_signal[:] = hp.generate_distraction()
end


"""
    transformA!(F,syms)

## Arguments
    - F::Matrix{<:Smy}:     A Matrix of expressions
    - syms::Tuple:          A tuple of symbols

Transforms F(x) to A(x)*x, with x a vector of symbols. Attention, we destroy f!!

"""
# First execute the outter loop then the inner...

function transformA(F::Vector{<:Sym},syms::Vector{<:Sym})
    ret = Matrix{Sym}( undef, length(F), length(F))
    for i = 1:length(syms)
        for j = 1:length(syms)
            if i == j
                ret[i,i] = F[i]/syms[i]
            else
                ret[i,j] = 0.0
            end
        end
    end
    return ret
end

"""
    evaluatemat(F::Matrix{<:Symbols}, kwargs...)

## Arguments
    - F::Matrix{<:Symbols}:     A symbolic matrix
    - kwargs:                   A list of tuples, with (symbols, values)

Evaluates the matrix at the list of tuples provided.

"""
evaluatemat(F::Union{AbstractMatrix,AbstractVector}, eval::Vector) = F
function evaluatemat(F::Matrix{<:Sym}, eval::Vector)

    ret = Matrix{Sym}( undef, size(F))
    for i = 1:size(F)[1]
        for j = 1:size(F)[2]
            ret[i,j] = N(F[i,j](eval...))
        end
    end
    return ret
end
function evaluatemat(F::Vector{<:Sym}, eval::Vector)

    ret = Vector{Sym}( undef, size(F))
    for i = 1:size(F)[1]
        ret[i] = N(F[i](eval...))
    end
    return ret
end
evaluatemat(F::Union{AbstractVector,AbstractMatrix}, kwargs...) = F
function evaluatemat(F::Union{Vector{<:Sym},Matrix{<:Sym}}, kwargs...)
    eval = generate_eval(kwargs...)
    return evaluatemat(F, eval)
end

generate_eval(kwargs...) = [el  for (sym,val) in kwargs for el in helperevalmat(sym, val)]
helperevalmat(sym::Sym, val) = [sym => val]
helperevalmat(sym::Vector{<:Sym}, val) = [sym[i] => val[i] for i=1:length(sym) ]





# function Unicycle(hp::UnicycleHP)
#     TestSetGen(unicycle_t, unicycle_e, unicycle_control, unicycle_init, 7, size(hp.C)[2], 2, hp)
# end


# We use the notation from Julia Differential Equations, therefore u corresponds to x
# And p are the hyper parameters including the controls.


function cs_dataset_t(xin::AbstractVector, hp::ControlSystemDataSet, n::Integer)
    refstate  = haskey(hp.arguments, :refstate) ?  hp.arguments[:refstate] : ones(length(hp.x))

    xinp = xin[1:length(hp.x)] .+ refstate
    controls = xin[length(hp.x)+1:end]

    df_eq(u_val,p_val,t_val) = begin
        # In case we want to have a non autonomous system
        evalx = generate_eval((hp.x, u_val), (symbols("t"), t_val))
        N(evaluatemat(hp.A, evalx))*u_val + N(evaluatemat(hp.B, evalx))*controls
    end
    tinit =  haskey(hp.arguments, :tinit) ?  hp.arguments[:tinit] : 0.0
    # We assume that there is at least one control term involved!!
    tspan = (tinit+ (n-1)*hp.interval_size, tinit+ n * hp.interval_size) # We have a autonomous system...
    prob = ODEProblem(df_eq,xinp,tspan)
    reltol=  haskey(hp.arguments, :reltol) ?  hp.arguments[:reltol] : 1e-8
    abstol=  haskey(hp.arguments, :abstol) ?  hp.arguments[:abstol] : 1e-8
    sol = DifferentialEquations.solve(prob,AutoVern7(Rodas5()),reltol=reltol,abstol=abstol) # Want good solutions!



    return sol[end] .- refstate
end



function cs_dataset_control(xin::Vector, t::Integer, hp::ControlSystemDataSet)
    # Extract A and B
    if hp.control
        refstate  = haskey(hp.arguments, :refstate) ?  hp.arguments[:refstate] : ones(length(hp.x))
        xin = xin .+ refstate

        tinit =  haskey(hp.arguments, :tinit) ?  hp.arguments[:tinit] : 0.0
        # We assume that there is at least one control term involved!!
        tval = tinit+ (t-1)*hp.interval_size

        eval = generate_eval((hp.x, xin), (symbols("t"), tval))
        A = N(evaluatemat(hp.A, eval))
        B = N(evaluatemat(hp.B, eval))
        R =  haskey(hp.arguments, :R) ?  hp.arguments[:R] : Diagonal(ones(length(hp.u)))
        Q =  haskey(hp.arguments, :Q) ?  hp.arguments[:Q] : Diagonal(ones(length(hp.x))) ./ hp.interval_size^2

        # Add jitter to guarantee the sainty of the equation.
        A = A + Diagonal(ones(size(A)[1])) * 1.0e-8

        #1) compute solution to the continous riccati equation
        P = ControlSystems.care(A, B, Q, R)
        # Compute F
        F = inv(R)* transpose(B)* P
        # u = -F*x
        u = -F*xin

        # cycle trough..
        d = hp.distraction_signal[:, (t%length(hp.distraction_signal)[1])+1]

        clip_ctrl = haskey(hp.arguments,:clip_ctrl) ? hp.arguments[:clip_ctrl] : Inf
        return _clip(u, clip_ctrl) .+ d
    else
        return hp.distraction_signal[:, (t%length(hp.distraction_signal)[1])+1]
    end

end


function cs_dataset_init(hp::ControlSystemDataSet)
    init = haskey(hp.arguments,:init) ? hp.arguments[:init] : zeros(length(hp.x))
    init_noise_v = haskey(hp.arguments,:init_noise_v) ? hp.arguments[:init_noise_v] : ones(length(hp.x))
    init_noise = haskey(hp.arguments,:init_noise) ? hp.arguments[:init_noise] : 1.0

    init_clip = haskey(hp.arguments,:init_clip) ? hp.arguments[:init_clip] : Inf
    refstate  = haskey(hp.arguments, :refstate) ?  hp.arguments[:refstate] : ones(length(hp.x))

    return init + _clip(rand(Normal(0.0, init_noise) ,length(hp.x)), 1.0) .* init_noise_v .- refstate
end


function generate_distraction_from_gp(stepsize::Float64 = 0.01, kernel::Stheno.Kernel =Stheno.stretch(Stheno.Matern32(),2.0); scale = 0.1, n_steps::Integer=1000, dims::Integer = 1)
    timeline = [(k-1)*stepsize for k = 1:n_steps]
    Kx = Stheno.pairwise(kernel, timeline)
    Cx = cholesky(Kx).L

    Ctrl = Cx * rand(Normal(), n_steps, dims)
    Ctrl = Ctrl .* scale
    return transpose(Ctrl)
end
