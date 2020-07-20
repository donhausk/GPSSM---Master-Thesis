

using GPSSM
using Bijectors
using Distributions
using Zygote


@testset "bijector extension" begin

    for i = 1:100
        d = Distributions.product_distribution([Exponential() for i = 1:10])
        v = rand(d)
        v2 = GPSSM.Bijectors.link(d, v)
        v2t = v2 .- 10000
        @test v ≈ Bijectors.invlink(d,v2)
        @test sum( Bijectors.invlink(d,v2t) .< 0.0) == 0 # Need to check that not the identity bijector is used
    end



    dist = product_distribution([Normal(),Gamma(), Exponential(2.9), Gamma(3.0), Normal(), Uniform(), Gamma()])
    f = (x) -> sum(Bijectors.link(dist, x))
    for i = 1:10
        d = rand(dist)

        val, pb = Zygote.pullback(f,d )
        @test pb(1)[1] == [Zygote.pullback((x)-> Bijectors.link(dist.v[i], x), d[i])[2](1)[1] for i = 1:length(d)]
    end

end
