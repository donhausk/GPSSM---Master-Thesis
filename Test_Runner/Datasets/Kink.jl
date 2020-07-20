function kinkf(xin::Vector, hp::ModelParameters, t::Integer)
    x = xin[1]
    xnew = 0.8 +(x+0.2)*(1-5/(1+exp(-2*x)))
    return [xnew]
end


function Kink()
    return Gaussian1D(kinkf)
end
