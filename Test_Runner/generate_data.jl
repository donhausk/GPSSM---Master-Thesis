using Distributions
using Plots
# We use the same configurations as in http://proceedings.mlr.press/v97/ialongo19a/ialongo19a.pdf

function generate_data(T::TestSetGen,  path::String, ON::Vector = [1.0]; key_test::String = "_Default",  N::Int ,m::Int=1, Ntest::Int=0,mtest::Int =0, normalise::Bool=false , pn::Real = 0.0, offset = nothing)
    Xtest, Ytest, Utest = [],[],[]
    if !isdir(path)
        mkpath(path)
    end
    if offset == nothing
        if hasfield(typeof(T.hp), :omit_samples)
            if T.hp.omit_samples > 0
                N = N+ T.hp.omit_samples
                Ntest = Ntest + T.hp.omit_samples
            end
        end
    else
        N = N+ offset
        Ntest = Ntest + offset
    end



    key_test = key_test*"_$pn"
    dataname = T.name*key_test

    generate_new_data = true
    testonly = false
    if isfile(path*dataname*".jld")
        f = JLD2.jldopen(path*dataname*".jld", "r")
        try
            X = f["X"]
            U = f["U"]
            testonly = true
            if mtest> 0
                Xtest = f["Xtest"]
                Utest = f["Utest"]
            end
            println("Successfully loaded the dataset with key_test"*dataname)
            generate_new_data = false
        catch e
            nothing
        end
        close(f)
    end
    if generate_new_data
        println("We generate a new dataset with key_test"*dataname)
        if !testonly
            X = Array{Float64,3}(undef,T.latentdim, N+1, m)
            U = Array{Float64,3}(undef, T.cdim, N+1, m) #no controls

            Xtest = Array{Float64,3}(undef,T.latentdim, Ntest+1, mtest)
            Utest = Array{Float64,3}(undef, T.cdim, Ntest+1, mtest) #no controls

            for i = 1:m
                reset_inputs!(T)
                X[:,1,i] = T.f_x0(T.hp) #standard normal initialize
                for j = 1:N
                    if pn > 0.0
                        pnoise = rand(Normal(0.0, sqrt.(pn)), T.latentdim)
                    else
                        pnoise = zeros(T.latentdim)
                    end
                    ut = T.f_u(X[:,j,i], j, T.hp)
                    U[:,j,i] = ut
                    X[:,j+1,i] = T.f(vcat(X[:,j,i],ut), T.hp, j) .+ pnoise
                end
                ut = T.f_u(X[:,N+1,i],N+1, T.hp)
                U[:,N+1,i] = ut
            end
        end
        if mtest>0
            for i = 1:mtest
                reset_inputs!(T)
                Xtest[:,1,i] = T.f_x0(T.hp) #standard normal initialize
                for j = 1:Ntest
                    ut = T.f_u(Xtest[:,j,i], j, T.hp)
                    Utest[:,j,i] = ut
                    Xtest[:,j+1,i] = T.f(vcat(Xtest[:,j,i],ut), T.hp, j)
                end
                ut = T.f_u(Xtest[:,Ntest+1,i], Ntest+1, T.hp)
                Utest[:,Ntest+1,i] = ut
            end
        end

        JLD2.jldopen(path*dataname*".jld", "a+") do file_jld
            if !testonly
                write( file_jld ,"X" ,X)
                write(file_jld ,"U" ,U)
            end
            if mtest >0
                write( file_jld ,"Xtest" ,Xtest)
                write( file_jld ,"Utest" ,Utest)
            end
        end
    end


    if normalise
        existing = false
        f = JLD2.jldopen(path*dataname*".jld", "a+")
        try
            nmconst = f["Norm_"*dataname]
            uconst = f["UNorm_"*dataname]
            close(f)
            X = X ./ nmconst
            U = U ./ uconst
            if mtest > 0
                Xtest = Xtest ./ nmconst
                Utest = Utest ./ uconst
            end
            existing = true
        catch e
            #noting
        end
        close(f)
        if ! existing
            nmconst = vec(mean(abs.(X),dims=(2,3)))
            X = X ./ nmconst
            uconst = vec(mean(abs.(U),dims=(2,3)))
            U = U ./ uconst

            if mtest > 0
                Xtest = Xtest ./ nmconst
                Utest = Utest ./ uconst

            end

            JLD2.jldopen(path*dataname*".jld", "a+") do file_jld
                write( file_jld ,"Norm_"*dataname ,nmconst)
                write( file_jld ,"UNorm_"*dataname ,uconst)
            end
        end
    end


    Y = Vector{Array{Float64,3}}(undef, length(ON))
    if mtest>0
        Ytest = Vector{Array{Float64,3}}(undef, length(ON))
    end


    for (i,on) in enumerate(ON)
        f = JLD2.jldopen(path*dataname*".jld", "r")
        generate_new_data = true
        try
            Y[i] = f["Y_$on"]
            if mtest >0
                Ytest[i] = f["Ytest_$on"]
            end

            println("Successfully loaded the Observations $on")
            generate_new_data = false
        catch e
            nothing
        end
        close(f)

        if generate_new_data
            Yo = Array{Float64,3}(undef,T.obsdim, N, m)
            if mtest >0
                Yotest =  Array{Float64,3}(undef,T.obsdim, Ntest, mtest)
            end


            for i = 1:m
                for j = 1:N
                    Yo[:,j,i] = T.f_obs(X[:,j+1,i], sqrt.(on), T.hp)
                end
            end
            Y[i] = Yo
            if mtest > 0
                for i = 1:mtest
                    for j = 1:Ntest
                        Yotest[:,j,i] = T.f_obs(Xtest[:,j+1,i], sqrt.(on), T.hp)
                    end
                end
                Ytest[i] = Yotest
            end

            JLD2.jldopen(path*dataname*".jld", "a+") do file_jld
                write( file_jld ,"Y_$on" ,Yo)
                if mtest >0
                    write( file_jld ,"Ytest_$on" ,Yotest)
                end
            end

            println("We generated the observations for on $on")

        end
    end
    if offset == nothing
        if hasfield(typeof(T.hp), :omit_samples)
            if T.hp.omit_samples > 0
                return @views X[:,T.hp.omit_samples+1:end,:],[Yi[:,T.hp.omit_samples+1:end,:] for Yi in Y],U[:,T.hp.omit_samples+1:end,:],Xtest[:,T.hp.omit_samples+1:end,:],[Yi[:,T.hp.omit_samples+1:end,:] for Yi in Ytest], Utest[:,T.hp.omit_samples+1:end,:]
            end
        end
    else
        return @views X[:,offset+1:end,:],[Yi[:,offset+1:end,:] for Yi in Y],U[:,offset+1:end,:],Xtest[:,offset+1:end,:],[Yi[:,offset+1:end,:] for Yi in Ytest], Utest[:,offset+1:end,:]
    end
    return X,Y,U,Xtest,Ytest,Utest
end
