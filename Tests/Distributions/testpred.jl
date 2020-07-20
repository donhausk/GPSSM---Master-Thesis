

using GPSSM
using Bijectors
using Distributions
using Zygote
using Turing
using StatsFuns
@model mld(A,b,S, mu,Sigma,y) = begin
    xt ~ GPSSM.Zygote_MvNormal(mu,Diagonal(Sigma))
    y ~ GPSSM.Zygote_MvNormal(A*xt+b, S)
    return @logpdf
end


@model mld2(A,b,S, mu,Sigma) = begin
    xt ~ GPSSM.MvNormal(mu,Diagonal(Sigma))
    y ~ GPSSM.MvNormal(A*xt+b, S)
    return @logpdf
end
@testset "get_pxytx1" begin

    for i = 1:100
        l = 1
        l2 = 1
        A = rand(Normal(), l2,l)
        b = rand(Normal(), l2)
        S = rand(Wishart(dimension,Matrix{Float64}(I,l2,l2)))
        Sigma = rand(Wishart(dimension,Matrix{Float64}(I,l,l)))
        mu = rand(Normal(), l, l)
        em = GPSSM.GaussianEM(A, b, S, Symmetric(inv(S)))
        y = rand(Normal(),l2)
        lpf = GPSSM.get_pyxt1(em, mu,Sigma,y)



        @test sum( Bijectors.invlink(d,v2t) .< 0.0) == 0 # Need to check that not the identity bijector is used
    end



end

nparticles = 20
l = 2
l2 = l
A = Diagonal(ones(l))
b = rand(Normal(), l2)
S = Diagonal(0.2*ones(l))
Sigma = rand(Exponential(),l,nparticles )
mu = rand(Normal(), l, nparticles)
em = GPSSM.GaussianEM(A, b, S, Symmetric(inv(S)))
y = rand(Normal(),l2)
lpf = GPSSM.get_pyxt1(em, mu,Sigma,y)
i = 1
mup, sigmap = GPSSM.get_fa_pred(em, mu,Sigma,y)
p = 1


for p = 1:nparticles
    bt, sigma = em.A * mu[:,p] +em.b, em.S + em.A * Diagonal(Sigma[:,p])* (em.A')
    println(bt - y)
    println(sigma)
    println(logpdf(MvNormal(bt, sigma),y))
end


em.S
em.A * Diagonal(Sigma[:,19])* (em.A')


i = 2

chn = sample(mld(A,b,S,mu[:,i],Diagonal(Sigma[:,i]),y),ESS(),3000)

chn = sample(mld2(A,b,S,mu[:,i],Diagonal(Sigma[:,i])),MH(),100000)

mean(chn["y[1]"].value)
mean(chn["y[2]"].value)
mean(chn["y[3]"].value)
mean(chn["y[4]"].value)
mean(chn["y[5]"].value)
std(chn["y[1]"].value)
std(chn["y[2]"].value)
std(chn["y[3]"].value)
std(chn["y[4]"].value)
std(chn["y[5]"].value)
em.b + em.A * mu[:,i]
em.A*Diagonal(Sigma[:,i]) *em.A' + sqrt(em.S)


mean(chn["xt[1]"].value)
mean(chn["xt[2]"].value)
mean(chn["xt[3]"].value)
mean(chn["xt[4]"].value)
mean(chn["xt[5]"].value)
std(chn["xt[1]"].value)
std(chn["xt[2]"].value)
std(chn["xt[3]"].value)
std(chn["xt[4]"].value)
std(chn["xt[5]"].value)
mu[:,1]
Sigma[:,i]




Diagonal(Sigma[:,i])

for i = 1:nparticles
    chn = sample(mld(A,b,S,mu[:,i],Diagonal(Sigma[:,i]),y),ESS(),5000)
    mui = mup[i]
    sigmai = sigmap[i]
    for k = 1:l
        println("_________________-")
        println(mui[k])
        println(mean(chn["xt[$k]"].value))
        #@test abs(mui[k] - mean(vec(chn["xt[$k]"].value)[1000:end]))/abs(mui[k]) < 0.1
    end
end




for i = 1:nparticles
    chn = sample(mld(A,b,S,mu[:,i],Diagonal(Sigma[:,i]),y),ESS(),5000)
    xvals = transpose(chn.value[1000:end,2:end,1])

    lpfi = lpf[i]
    println("_______________")
    println(lpfi)
    println(StatsFuns.logsumexp([logpdf(MvNormal(em.A*xvals[:,i] + em.b, em.S),y) for i = 1:size(xvals)[2]]) - log(size(xvals)[2]))
end
