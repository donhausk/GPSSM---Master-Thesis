


struct LinearGaussianHP<: ModelParameters
    v::AbstractVector
    tint::Float64
    dim::Int
    get_control::Function
    omit_samples::Integer
end
LinearGaussianHP() = LinearGaussianHP(ones(1),0.2, 1,  (x,i, hp) -> zeros(0), 50)
LinearGaussianHP(v::AbstractVector, dim::Int) = LinearGaussianHP(v,0.1,  dim,  (x,i, hp) -> zeros(0), 50)
function LinearGaussianHP( dim::Int)
    v = [(-1)^k *1/dim for k = 1:dim]
    v[end] = 0.0
    LinearGaussianHP(v,0.1, dim,  (x,t, hp) -> [sin(2*pi*t/50)+sin(2*pi*t/200)], 50)
end


function LinearGaussian(dim::Integer)
    return LinearGaussian(LinearGaussianHP(dim))
end
# https://arxiv.org/pdf/1610.08962.pdf
function LinearGaussian(hp::LinearGaussianHP)
    TestSetGen("LinearGaussian", lineargaussian_t, lineargaussian_e, lineargaussian_control , lineargaussian_init, hp.dim, hp.dim, size(hp.get_control(zeros(hp.dim), 0, hp))[1], hp)
end




function get_A( hp::LinearGaussianHP)
    if hp.dim ==1
        return hp.v*ones(1,1)
    end
    A0 =zeros(hp.dim, hp.dim)

    for k = 1:hp.dim-1
        A0[k,k+1] = 1.0
        #A0[k+1,k] = hp.a1
    end
    A0[end,:] = hp.v
    return A0
end


function lineargaussian_t(xin::AbstractVector, hp::LinearGaussianHP, n::Integer)
    u = zeros(hp.dim)
    if length(xin) > hp.dim
        @assert length(xin) == hp.dim+1 "We do allow only a one dimensional control!"
        u[hp.dim] = xin[hp.dim+1]
    end
    return get_A(hp)*xin[1:hp.dim] .+ u
end


function lineargaussian_control(xin::AbstractVector,t, hp::LinearGaussianHP)
    return hp.get_control(xin, t, hp)
end

function lineargaussian_e(xin::Vector, v, hp::LinearGaussianHP)
    return xin .+ rand(Normal(0.0,v), length(xin))
end
function lineargaussian_init(hp::LinearGaussianHP)
    return rand(MvNormal(zeros(hp.dim), 1.0)) # This is used in Finkes paper
end
