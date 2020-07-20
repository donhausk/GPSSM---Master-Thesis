
"""

The dataset parameters

"""


struct NarendraLiParameters <: ModelParameters
    p1::Float64
    p2::Float64
    p3::Float64
    p4::Float64
    p5::Float64
    p6::Float64
end


NarendraLiParameters() = NarendraLiParameters(1.05, 7, 0.52, 0.52, 0.48, 1.0)

"""

Get the dataset

"""

function Narendra_li(params::NarendraLiParameters)
    return TestSetGen(narendra_li_t, narendra_li_e,narendra_li_control, narendra_li_init, 2, 1, 1, params)
end

function Narendra_li()
    return TestSetGen("NarendraLi", narendra_li_t, narendra_li_e,narendra_li_control, narendra_li_init, 2, 1, 1, NarendraLiParameters())
end


"""

The model functions

"""


#form https://ch.mathworks.com/help/ident/examples/narendra-li-benchmark-system-nonlinear-grey-box-modeling-of-a-discrete-time-system.html

function narendra_li_t(xin::Vector, hp::NarendraLiParameters, n::Integer)
    x1 = xin[1]
    x2 = xin[2]
    u = xin[3]
    x1t = (x1/(1+x1^2)+ hp.p1 )*sin(x2)
    x2t = x2*cos(x2) + x1*exp(-(x1^2+x2^2)/hp.p2) + u^3/(1+u^2+hp.p3*cos(x1+x2))
    return [x1t, x2t]
end

function narendra_li_e(xin::Vector, v, hp::NarendraLiParameters)
    x1 =xin[1]
    x2 = xin[2]
    yt = x1/(1+hp.p4*sin(x2)+hp.p5*sin(x1)) + rand(Normal(0.0,v))
    return [yt]
end

function narendra_li_control(xin::Vector, t, hp::NarendraLiParameters)
    u = sin(2*pi*t/10) + sin(2*pi*t/25)
    return [u]
end
function narendra_li_init(hp::NarendraLiParameters)
    return rand(Normal(0.0,hp.p6),2)
end
