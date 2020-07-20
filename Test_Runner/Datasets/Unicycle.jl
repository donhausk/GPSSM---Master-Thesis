
struct UnicycleHP <: ModelParameters
    a0dot::Float64
    g::Float64
    F::Float64
    interval_size::Float64
    init::AbstractVector
    C::AbstractMatrix
    init_noise::Float64
end
UnicycleHP(C::AbstractMatrix) = UnicycleHP(0.6, 9.81, 0.01 , 0.01, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], C,  0.001)
UnicycleHP() = UnicycleHP(0.6, 9.81, 0.01 , 0.01, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Matrix{Float64}(I, 7,7),  0.001)


Unicycle() = Unicycle(UnicycleHP())

function Unicycle(hp::UnicycleHP)
    TestSetGen("Unicycle", unicycle_t, unicycle_e, unicycle_control, unicycle_init, 7, size(hp.C)[2], 2, hp)
end


# Paper https://link.springer.com/article/10.1023/A:1026481216262

function unicycle(u,p,t)
    get_A_unicyle(u,p)*u + get_B_unicyle(u,p)*p[:u]
end

# We use the notation from Julia Differential Equations, therefore u corresponds to x
# And p are the hyper parameters including the controls.

function get_A_unicyle(u,p)
    [0 14.6*u[2]*u[3] -194.6 0 0 (3.54*u[3]*u[5]*u[1]-2.7*u[4] + 30.1*u[3]*u[2]*u[5]) 0;
       0 0 66 0 0 (0.75*u[6]*u[3]-1.77*u[4]-1.78*u[3]*u[1]*u[5]) 0;
       0 1 0 0 0 0 0;
       0 0 0 0 1.1*p[:g] (1.53*u[2]+0.22*(u[1]+p[:a0dot])) 0;
       0 0 0 1 0 0 0 ;
       -0.1*u[4] -1.4*u[4] 1.7*p[:g]*u[5] -0.1*p[:a0dot] 0 (0.43*u[3]*(u[1]+p[:a0dot]) -2.6*p[:F]-0.81*u[2]*u[3]+0.53*u[5]*u[4]+0.37*u[3]*u[7]*u[5]) 0.37*u[2]*u[3];
       0.1*u[4] 0.21*u[4] -2.9*p[:g]*u[5] 0.1*p[:a0dot] 0 (-0.65*u[3]*(u[1]+p[:a0dot]) +2.6*p[:F] -0.3*u[3]*u[5]*u[7]-0.94*u[3]*u[6]*u[5]) -0.35*u[2]*u[3]]
end
function get_B_unicyle(u,p)
     [-3.1+12.8*u[3]^2 -2.6*u[5];
      0.89-3.9*u[3]^2 -2.6*u[5];
      0 0;
      0 2*u[3];
      0 0;
      0 2.6;
      0 -4.3+1.5*u[3]^2]
end




# Paper https://link.springer.com/article/10.1023/A:1026481216262
function unicycle_t(xin::AbstractVector, hp::UnicycleHP, n::Integer)
    x = xin[1:7]
    u = xin[8:end]
    p = (a0dot = hp.a0dot, g = hp.g, F = hp.F, u = u)

    tspan = (0.0,hp.interval_size) # We have a autonomous system...
    prob = ODEProblem(unicycle,x,tspan, p)
    sol = solve(prob,AutoVern7(Rodas5()),reltol=1e-8,abstol=1e-8) # Want good solutions!
    return sol[end]
end

function unicycle_control(xin::Vector, t, hp::UnicycleHP)
    # Extract A and B
    p = (a0dot = hp.a0dot, g = hp.g, F = hp.F)
    A = get_A_unicyle(xin, p)
    B = get_B_unicyle(xin, p)
    R = Diagonal([0.7, 0.08])
    Q = Diagonal([1.0, 1, 10, 1, 1, 1, 1])
    #1) compute solution to the continous riccati equation

    P = ControlSystems.care(A, B, Q, R)
    # Compute F
    F = inv(R)* transpose(B)* P
    # u = -F*x
    return _clip(-F*xin, 5.0) + [ sin(2*pi*t/50), sin(2*pi*t/200)] # We want to have a weak controller
end

function unicycle_init(hp::UnicycleHP)
    return hp.init + _clip(rand(Normal(0.0, hp.init_noise)), 1.0) .* [ 0, 0.0, 1.0, 0, 0.0, 0.0, 0.0] + _clip(rand(Normal(0.0, hp.init_noise)), 1.0) .* [ 0, 0.0, 0.0, 0, 1.0, 0.0, 0.0]
end
function unicycle_e(xin::Vector, v, hp::UnicycleHP)
    yt = hp.C*xin
    yt +=  rand(Normal(0.0,v), length(yt))
    return yt
end
