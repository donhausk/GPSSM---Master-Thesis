# struct ExpData{M<:Union{Nothing,AbstractArray{<:Real,3}},MP<:Union{Nothing,AbstractArray{N,2}}, MPr<:Union{Nothing,AbstractArray{N,3}}}
#     inputdim::Int
#     Y::M
#     U::M
#     X::Union{AbstractArray, Nothing}
#     Xtest::M
#     Ytest::M
#     Utest::M
#     Xplot::MP
#     Yplot::MP
#     Uplot::MP
#     YpredSeq::MPr
#     UpredSeq::MPr
#     predictahead::Integer
# end

struct ExpData
    inputdim::Integer
    Y
    U
    X
    Xtest
    Ytest
    Utest
    Xplot
    Uplot
    YpredSeq
    UpredSeq
    predictahead::Integer
end


ExpData(d,Y, U; X=nothing, Xtest=nothing, Ytest=nothing, Utest=nothing, Xplot=nothing, Uplot=nothing, YpredSeq=nothing, UpredSeq=nothing, predictahead = 0) = ExpData(d,Y, U,X, Xtest, Ytest, Utest, Xplot, Uplot, YpredSeq, UpredSeq, predictahead)


macro run_experiments(path, pathstorage, input_expr)
    build_model_info(path, pathstorage,  input_expr) |>  replace_counters! |> replace_mdl! |> build_output!
end
macro run_experiments_gpt(path, pathstorage, input_expr)
    build_model_info(path, pathstorage,  input_expr) |>  replace_counters! |> replace_mdl_gpt! |> build_output_gpt!
end
macro getelementm(arr, key)
    return  esc(
    quote
        eval($arr)[$key]
    end)
end

function build_model_info(path, pathstorage,  input_expr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(input_expr)
    # Function body of the model is empty
    args = map(modeldef[:args]) do arg
        if (arg isa Symbol)
            arg
        else
            error("Only Symbols as parameters allowed")
        end
    end

    argnames = map(modeldef[:args]) do arg
        if (arg isa Symbol)
            string(arg)
        else
            error("Only Symbols as parameters allowed")
        end
    end
    @assert length(args) != 0 "No experiments are specified!"
    model_info = Dict(
        :path => path,
        :pathstorage => pathstorage,
        :name => string(modeldef[:name]),
        :main_body => modeldef[:body],
        :args => args,
        :argnames => argnames,
        :keys_macro =>  [:i_macro_1, :i_macro_2, :i_macro_3, :i_macro_4,:i_macro_5,:i_macro_6,:i_macro_7],
        :setspec => false)

    return model_info
end

function replace_counters!(model_info)
    ex = model_info[:main_body]
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @i)
            :i_macro_1
        elseif @capture(x, @j)
            :i_macro_2
        elseif @capture(x, @k)
            :i_macro_3
        elseif @capture(x, @l)
            :i_macro_4
        elseif @capture(x, @m)
            :i_macro_5
        elseif @capture(x, @n)
            :i_macro_6
        elseif @capture(x, @o)
            :i_macro_7
        else
             x
        end
    end
    model_info[:main_body] = ex
    return model_info
end

function replace_mdl!(model_info)
    ex = model_info[:main_body]
    vars = falses(5)
    spectrue = false
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @alt ~ R_)
            vars[1] = true
            quote
                alt =$R
            end
        elseif @capture(x, @ahp ~ R_)
            vars[2] = true
            quote
                ahp =$R
            end
        elseif @capture(x, @spec ~ R_)
            spectrue = true
            quote
                spec =$R
            end
        elseif @capture(x, @Niter ~ R_)
            vars[4] = true
            quote
                Niter =$R
            end
        elseif @capture(x, @instance ~ R_)
            vars[5] = true
            quote
                instance =$R
            end
        elseif @capture(x, @data ~ R_)
            vars[3] = true
            quote
                expdata =$R
                Y = expdata.Y[:,1:end-expdata.predictahead,:]
                U = expdata.U[:,1:end-expdata.predictahead,:]
                inputs = (sizes = (expdata.inputdim, size(Y)[2], size(Y)[3]), controls = U,
                    observations = Y)
                if expdata.predictahead>0
                    Ypredseqahead = expdata.Y[:,end-expdata.predictahead+1:end,:]
                     # Note that this is indeed what we want
                    Upredseqahead = expdata.U[:,end-expdata.predictahead:end,:]
                    inputs_pred_temporary = (sizes = (expdata.inputdim, size(Ypredseqahead)[2], size(Ypredseqahead)[3]), controls = Upredseqahead,
                        observations = Ypredseqahead)
                end
            end
        else
            x
        end
    end
    model_info[:main_body] = ex
    @assert sum(vars) == 5 "All variables @alt, @ahp, @Niter, @instance and @data must be specified"

    ex = model_info[:main_body]
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @alt)
            :alt
        elseif @capture(x, @spec)
            :spec
        elseif @capture(x, @ahp)
            :ahp
        elseif @capture(x, @data)
            :expdata
        elseif @capture(x, @instance)
            :instance
        elseif @capture(x, @Niter)
            :Niter
        else
            x
        end
    end
    model_info[:setspec] =spectrue
    model_info[:main_body] = ex
    return model_info
end



function replace_mdl_gpt!(model_info)
    ex = model_info[:main_body]
    vars = falses(2)
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @args ~ R_)
            vars[1] = true
            quote
                arguments_macro =$R
            end
        elseif @capture(x, @data ~ R_)
            vars[2] = true
            quote
                expdata =$R
                Y = expdata.Y[:,1:end-expdata.predictahead,:]
                U = expdata.U[:,1:end-expdata.predictahead-1,:]
                inputs = (sizes = (expdata.inputdim, size(Y)[2], size(Y)[3]), controls = U,
                    observations = Y)
                if expdata.predictahead>0
                    Ypredseqahead = expdata.Y[:,end-expdata.predictahead+1:end,:]
                     # Note that this is indeed what we want
                    Upredseqahead = expdata.U[:,end-expdata.predictahead:end-1,:]
                    inputs_pred_temporary = (sizes = (expdata.inputdim, size(Ypredseqahead)[2], size(Ypredseqahead)[3]), controls = Upredseqahead,
                        observations = Ypredseqahead)
                else
                    Ypredseqahead = nothing
                    Upredseqahead = nothing
                end
            end
        else
            x
        end
    end
    model_info[:main_body] = ex
    @assert sum(vars) == 2 "All variables @args, @data must be specified"
    return model_info
end


function setup(model_info)
    argnames = model_info[:argnames]

    return quote

        pth = eval($model_info[:path])*"/"*eval($model_info[:name])*"/"
        pthstorage = eval($model_info[:pathstorage])*"/"*eval($model_info[:name])*"/"
        # We need to have a valid
        if !isdir( eval($model_info[:path]))
            mkpath( eval($model_info[:path]))
        end
        if !isdir(eval($model_info[:pathstorage]))
            mkpath(eval($model_info[:pathstorage]))
        end
        if !isdir(pth)
            mkpath(pth)
        end
        if !isdir(pthstorage)
            mkpath(pthstorage)
        end
        if !isdir(pth*"Results/")
            mkpath(pth*"Results/")
        end
        if !isdir(pth*"Results/Chains/")
            mkpath(pth*"Results/Chains/")
        end
        if !isdir(pth*"Results/Predictions/")
            mkpath(pth*"Results/Predictions/")
        end
        if !isdir(pth*"Logger/")
            mkpath(pth*"Logger/")
        end
        path = pth
        name = $model_info[:name]


        try
            spec_data = load(path*name*"_spec.jld")
            has_changed = false
            for (h,arg) in enumerate($model_info[:args])
                if !(string.(arg) == spec_data[$argnames[h]])
                    spec_data[$argnames[h]] = unique(spec_data[$argnames[h]]..., string.(arg)...)
                    $model_info[:args][h] = spec_data[$argnames[h]]
                    has_changed = true
                end
            end
            if has_changed
                println("The Experiment specificiations are not consistent! We extend them!")
                JLD2.jldopen(path*name*"_spec.jld", "w") do file_jld
                    for (h,arg) in enumerate($model_info[:args])
                        write(file_jld, $argnames[h], SAVESPEC[$argnames[h]])
                    end
                end
            end
        catch e
            JLD2.jldopen(path*name*"_spec.jld", "w") do file_jld
                for (h,arg) in enumerate($model_info[:args])
                    write(file_jld, $argnames[h], arg)
                end
            end
        end
    end
end

function main(model_info)
    mainbody = model_info[:main_body]
    len = length(model_info[:args])
    setspec = model_info[:setspec]
    quote
        keys_macro = [i_macro_1 i_macro_2 i_macro_3 i_macro_4 i_macro_5 i_macro_6 i_macro_7]

        sp = []
        for m = 1:length($model_info[:args])

            str = "&"*$model_info[:argnames][m]*"="
            v = string(@getelementm $model_info[:args][m] keys_macro[m] )
            str *= v
            push!(sp, str)
        end
        Spec_name = join(sp)
        path = eval($model_info[:path])*"/"*eval($model_info[:name])*"/"
        pathstorage = eval($model_info[:pathstorage])*"/"*eval($model_info[:name])*"/"

        name = $model_info[:name]

        already_exitsing = isfile(path*"Results/Predictions/Predictions"*Spec_name*".jld")

        if !already_exitsing
            println("Experiment "*Spec_name*" not yet existing. We run the Experiment")
            $mainbody

            #This is that the time measurements are reliable!
            # We precompile everything here.

            if $setspec
                AbstractMCMC.sample(instance, spec,ahp,alt,inputs, 2)
            else
                AbstractMCMC.sample(instance, ahp,alt,inputs, 2)
            end
            string_to_file = ""
            string_to_file = string_to_file* "#########################################################"

            string_to_file = string_to_file * " \n " * string("We run the specification "*Spec_name)
            if $setspec
                timestr = @timed chn, args_save, samplingtimes = AbstractMCMC.sample(instance, spec,ahp,alt,inputs, Niter)
            else
                timestr = @timed chn, args_save, samplingtimes = AbstractMCMC.sample(instance, ahp,alt,inputs, Niter)
            end

            string_to_file = string_to_file * " \n " * string("Used time: "*string(timestr[2:end])*" allocated bytes:"*string(timestr[3])*" %GC time "*string(timestr[4]))


            chn_dir = path*"Results/Chains/"*Spec_name*".jls"
            MCMCChains.write(chn_dir, chn )


            if expdata.Xtest !== nothing
                mu = Array{GPSSM.get_latent_type(), 4}(undef,length(args_save),  inputs[:sizes][1], size(expdata.Xtest)[2]+1, size(expdata.Xtest)[3])
                sigma = Array{GPSSM.get_latent_type(), 4}(undef, length(args_save),  inputs[:sizes][1], size(expdata.Xtest)[2]+1, size(expdata.Xtest)[3])

                for h = 1:size(expdata.Xtest)[3]
                    if $setspec
                        mu[:,:,:,h], sigma[:,:,:,h] = GPSSM.predict(vcat(expdata.Xtest[:,:,h], expdata.Utest[:,:,h]), instance, spec, ahp, alt, inputs, args_save)
                    else
                        mu[:,:,:,h], sigma[:,:,:,h] = GPSSM.predict(vcat(expdata.Xtest[:,:,h], expdata.Utest[:,:,h]), instance, ahp, alt, inputs, args_save)
                    end
                end
            end

            if expdata.YpredSeq !== nothing
                predseq_size = (inputs[:sizes][1], size(expdata.YpredSeq)[2] ,size(expdata.YpredSeq)[3])
                predseq_inputs = (sizes= predseq_size, controls = expdata.UpredSeq, observations =  expdata.YpredSeq)
                if $setspec
                    results = GPSSM.predict_ms(instance, spec, ahp, alt, inputs,predseq_inputs,args_save )

                else
                    results = GPSSM.predict_ms(instance, ahp, alt, inputs,predseq_inputs,args_save )
                end

            end

            if expdata.Xplot !== nothing

                if $setspec
                    muplot, sigmaplot = GPSSM.predict(vcat(expdata.Xplot, expdata.Uplot), instance, spec, ahp, alt, inputs, args_save;)
                else
                    muplot, sigmaplot = GPSSM.predict(vcat(expdata.Xplot, expdata.Uplot),instance, ahp, alt, inputs, args_save;)
                end
            end

            if expdata.predictahead >0
                Xvals = [args_save[k][:X][:,end:end,:,:] for k = 1:length(args_save)]
                if $setspec
                    results2 = GPSSM.predict_ms(instance, spec, ahp, alt, inputs,inputs_pred_temporary,args_save; Xstart = Xvals, Npredahead =  expdata.predictahead  )

                else
                    results2 = GPSSM.predict_ms(instance, ahp, alt, inputs,inputs_pred_temporary ,args_save;  Xstart = Xvals, Npredahead =  expdata.predictahead )
                end
            end

            JLD2.jldopen(path*"Results/Predictions/Predictions"*Spec_name*".jld", "w") do file
                if expdata.Xtest !== nothing
                    write(file ,"mu",mu)
                    write(file ,"sigma",sigma)
                end
                if expdata.Xplot !== nothing
                    write(file ,"mu_plot",muplot)
                    write(file ,"sigma_plot",sigmaplot)
                end
                if expdata.YpredSeq !== nothing
                    write(file, "SamplingTimes", samplingtimes)
                    write(file, "MahPX", results[1])
                    write(file, "MahPMSE", results[2])
                    write(file, "MahPNLPP", results[3])
                end
                if expdata.predictahead >0
                    write(file, "MahPX2", results2[1])
                    write(file, "MahPMSE2", results2[2])
                    write(file, "MahPNLPP2", results2[3])
                end
                write(file ,"TimesTrain",timestr[2:end])
            end

            JLD2.jldopen(pathstorage*"Data"*Spec_name*".jld", "w") do file
                for h_save = 1:length(args_save)
                    write(file ,"T$h_save", args_save[h_save])
                end
            end
            open(path*"Logger/"*Spec_name*"_info.txt", "w") do file
                write(file, string_to_file)
            end


        else
            println("Experiment "*Spec_name*" already exists in the dataset and is therefore omitted.")
        end
    end
end



function main_gpt(model_info)
    mainbody = model_info[:main_body]
    len = length(model_info[:args])

    quote
        keys_macro = [i_macro_1 i_macro_2 i_macro_3 i_macro_4 i_macro_5 i_macro_6 i_macro_7]

        sp = []
        for m = 1:length($model_info[:args])

            str = "&"*$model_info[:argnames][m]*"="
            v = string(@getelementm $model_info[:args][m] keys_macro[m] )
            str *= v
            push!(sp, str)
        end
        Spec_name = join(sp)
        path = eval($model_info[:path])*"/"*eval($model_info[:name])*"/"
        pathstorage = eval($model_info[:pathstorage])*"/"*eval($model_info[:name])*"/"

        name = $model_info[:name]


        assert_trigger = false
        already_existing = isfile(path*"Logger/"*Spec_name*".txt") ? true : false
        if already_existing
            println(path*"Logger/"*Spec_name*".txt")
            println("Experiment "*Spec_name*" already exists in the dataset and is therefore omitted.")
        end


        if !already_existing
            println("We start with the Experiment!")
            $mainbody
            # Prepare script to run.
            if !haskey(arguments_macro, :taskset)
                ncpus = haskey(arguments_macro, :ncpus) ? arguments_macro.ncpus : 2
                cmd = " MPLBACKEND=Agg "
            else
                cmd = "MPLBACKEND=Agg  "
            end

            if !haskey(arguments_macro, :python)
                cmd = cmd *"/usr/bin/python3"
            else
                cmd = cmd *arguments_macro.python
            end

            dir = splitdir(splitdir(pathof(Test_Runner))[1])[1]
            cmd = cmd * " "*dir*"/Test_Runner/gpt_base_file.py  "

            cmd = cmd * " --dir '"*path * "' --largestorage '"*pathstorage*"'"
            keys_macro = ["n_ind_pts", "model_class", "emission_model", "filter_length", "seed","test_length", "n_seq", "n_iter", "process_noise_var", "emission_noise_var", "train_emission"]
            for h in 1:length(arguments_macro)
                k = keys(arguments_macro)[h]
                v = arguments_macro[h]
                if string(k) in keys_macro
                    cmd = cmd * " --"*string(k)*" "*string(v)
                end
            end




            cmd = cmd * " --name '"* Spec_name *"'"
            cmd = cmd * " --data '"* path*"'"
            cmd = cmd * " --T "*string(inputs[:sizes][2])
            cmd = cmd * "  --D "*string(inputs[:sizes][1])
            pathout =path*"Logger/"*Spec_name*".txt"
            cmd = cmd * " > "*"'"*pathout*"'"

            expdata.X == nothing ? npzwrite(path*"X_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"X_"*Spec_name*".npz",expdata.X[:,1:size(inputs[:observations])[2],:])
            expdata.Y == nothing ? npzwrite(path*"Y_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Y_"*Spec_name*".npz",inputs[:observations])
            expdata.U == nothing ? npzwrite(path*"U_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"U_"*Spec_name*".npz",inputs[:controls])
            expdata.Xtest == nothing ? npzwrite(path*"Xtest_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Xtest_"*Spec_name*".npz",expdata.Xtest)
            expdata.Ytest == nothing ? npzwrite(path*"Ytest_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Ytest_"*Spec_name*".npz",expdata.Ytest)
            expdata.Utest == nothing ? npzwrite(path*"Utest_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Utest_"*Spec_name*".npz",expdata.Utest)
            expdata.Xplot == nothing ? npzwrite(path*"Xplot_"*Spec_name*".npz",zeros(0,0)) : npzwrite(path*"Xplot_"*Spec_name*".npz",expdata.Xplot)
            expdata.Uplot == nothing ? npzwrite(path*"Uplot_"*Spec_name*".npz",zeros(0,0)) : npzwrite(path*"Uplot_"*Spec_name*".npz",expdata.Uplot)
            expdata.YpredSeq == nothing ? npzwrite(path*"Ypred_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Ypred_"*Spec_name*".npz",expdata.YpredSeq)
            expdata.UpredSeq == nothing ? npzwrite(path*"Upred_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Upred_"*Spec_name*".npz",expdata.UpredSeq)
            Ypredseqahead == nothing ? npzwrite(path*"Ypredseqahead_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Ypredseqahead_"*Spec_name*".npz",Ypredseqahead)
            Upredseqahead == nothing ? npzwrite(path*"Upredseqahead_"*Spec_name*".npz",zeros(0,0,0)) : npzwrite(path*"Upredseqahead_"*Spec_name*".npz",Upredseqahead)

            println(`sh -c $cmd`)
            res = @timed run(`sh -c $cmd`)
        end
    end
end


function build_output!(model_info)
    name = model_info[:name]
    # First the saving structure
    Setup = setup(model_info)
    Main = main(model_info)

    n_args = length(model_info[:args])
    excheck = quote
        @everywhere niter = 1
        @everywhere iterlengths = []
    end

    for check_i = 1:n_args
        sym = model_info[:args][check_i]
        excheck = quote
            $excheck
            @assert typeof($sym) <: AbstractVector "$(sym) must be a Vector"
            @everywhere niter *= length($sym)
            @everywhere push!(iterlengths, length($sym))
        end
    end
    base_expression = quote
        @everywhere base_iterator_length = niter
        @sync @distributed for base_iter = 1:base_iterator_length
            i_macro_1,i_macro_2, i_macro_3, i_macro_4, i_macro_5, i_macro_6, i_macro_7 =  0,0,0,0,0,0,0
            i_macro_1 = base_iter % iterlengths[1] +1
            if $n_args > 1
                rbase_iter = Integer(floor(base_iter / iterlengths[1]))
                i_macro_2 = rbase_iter% iterlengths[2] +1
                if $n_args >2
                    rbase_iter = Integer(floor(rbase_iter/iterlengths[2]))
                    i_macro_3 = rbase_iter% iterlengths[3] +1
                    if $n_args >3
                        rbase_iter = Integer(floor(rbase_iter/iterlengths[3]))
                        i_macro_4 = rbase_iter% iterlengths[4] +1
                        if $n_args >4
                            rbase_iter = Integer(floor(rbase_iter/iterlengths[4]))
                            i_macro_5 = rbase_iter% iterlengths[5] +1
                            if $n_args >5
                                rbase_iter = Integer(floor(rbase_iter/iterlengths[5]))
                                i_macro_6 = rbase_iter% iterlengths[6] +1
                                if $n_args >6
                                    rbase_iter = Integer(floor(rbase_iter/iterlengths[6]))
                                    i_macro_17 = rbase_iter% iterlengths[7] +1
                                end
                            end
                        end
                    end
                end
            end
            println("We starte with iterator: $base_iter out of $niter")
            $Main
            println("We have completed with iterator: $base_iter out of $niter")
        end
    end

    println("okey")

    if length(model_info[:args]) >7
        error("Too many arguments!")
    end

    r = quote
        $Setup
        $excheck
        $base_expression

    end
    esc(r)
end


function build_output_gpt!(model_info)
    name = model_info[:name]
    # First the saving structure
    Setup = setup(model_info)
    Main = main_gpt(model_info)

    n_args = length(model_info[:args])
    excheck = quote
        @everywhere niter = 1
        @everywhere iterlengths = []
    end

    for check_i = 1:n_args
        sym = model_info[:args][check_i]
        excheck = quote
            $excheck
            @assert typeof($sym) <: AbstractVector "$(sym) must be a Vector"
            @everywhere niter *= length($sym)
            @everywhere push!(iterlengths, length($sym))
        end
    end
    base_expression = quote
        @everywhere base_iterator_length = niter
        @sync  @distributed for base_iter = 1:base_iterator_length
            i_macro_1,i_macro_2, i_macro_3, i_macro_4, i_macro_5, i_macro_6, i_macro_7 =  0,0,0,0,0,0,0
            i_macro_1 = base_iter % iterlengths[1] +1
            if $n_args > 1
                rbase_iter = Integer(floor(base_iter / iterlengths[1]))
                i_macro_2 = rbase_iter% iterlengths[2] +1
                if $n_args >2
                    rbase_iter = Integer(floor(rbase_iter/iterlengths[2]))
                    i_macro_3 = rbase_iter% iterlengths[3] +1
                    if $n_args >3
                        rbase_iter = Integer(floor(rbase_iter/iterlengths[3]))
                        i_macro_4 = rbase_iter% iterlengths[4] +1
                        if $n_args >4
                            rbase_iter = Integer(floor(rbase_iter/iterlengths[4]))
                            i_macro_5 = rbase_iter% iterlengths[5] +1
                            if $n_args >5
                                rbase_iter = Integer(floor(rbase_iter/iterlengths[5]))
                                i_macro_6 = rbase_iter% iterlengths[6] +1
                                if $n_args >6
                                    rbase_iter = Integer(floor(rbase_iter/iterlengths[6]))
                                    i_macro_17 = rbase_iter% iterlengths[7] +1
                                end
                            end
                        end
                    end
                end
            end
            println("We starte with iterator: $base_iter out of $niter")
            $Main
            println("We have completed with iterator: $base_iter out of $niter")
        end
    end

    println("okey")

    if length(model_info[:args]) >7
        error("Too many arguments!")
    end

    r = quote
        $Setup
        $excheck
        $base_expression

    end
    esc(r)
end
