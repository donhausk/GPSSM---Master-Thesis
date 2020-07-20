
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    mhp::MHP,
    shp::SHP,
    slt::SLT,
    N::I,
    tshp::Vector{THP},
    tslt::Vector{TLT};
    kwargs...
) where {
    I<:Integer,
    MHP<:AbstractHPModel,
    SLT<:AbstractLSSampler,
    SHP<:AbstractGPSSMSampler,
    THP<:AbstractTransition,
    TLT<:AbstractTransition}

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nmshp, parrayhp = bundle_transitions(tshp)
    nmslt, parraylt = bundle_transitions(tslt)

    if !isempty(nmslt)
        if !isempty(nmshp)
            nms = vcat(nmshp..., nmslt...)
            parray = hcat(parrayhp, parraylt)
        else
            nms = nmslt
            parray = parraylt
        end
    else
        if !isempty(nmshp)
            nms = nmshp
            parray = parrayhp
        else
            nms = ["Nothing"]
            parray = zeros(Union{Missing, Real}, N,1)
        end
    end

    le = missing
    # Set up the info tuple.
    info = NamedTuple()
    # Chain construction.
    return Chains(
        parray,
        string.(nms);
        evidence=le,
        info=info
    )
end



"""
    ConstTransition{T}

This is a constant transition, which will be used to construct the MCMC chain.

"""

struct ConstTransition{T} <: AbstractTransition
    θ::T
end
function additional_parameters(::Type{<:ConstTransition})
    return []
end

"""
     PriorTransition{Tθx, Tθy}

This transition is used for the fully bayesian sampling where we separate between emission
and transition hyper parameters.

"""

# We have to leave this untyped...
struct PriorTransition{Tθx, Tθy} <: AbstractTransition
    Transθx::Tθx
    Transθy::Tθy
end

function additional_parameters(::Type{<:Turing.VarInfo})
    return []
end


# AbstractMCMC.transition_type(S::T) where{T <: LSMHGibbsSampler} = MHGibbsTransition
# AbstractMCMC.transition_type(S::T) where{T <: Union{LSSampler, CustomLSSampler}} = ConstTransition

transition_type(S::TuringHPSampler)      = PriorTransition
transition_type(S::AbstractGPSSMSampler) = ConstTransition
transition_type(S::AbstractLSSampler)    = ConstTransition






function bundle_transitions(tshp::Vector{PriorTransition})
    nmsθx, valsθx = _params_to_array([tshp[i].Transθx for i in 1:length(tshp)])
    if !(typeof(tshp[1].Transθx) <: ConstTransition)
        ep, ev = Turing.Inference.get_transition_extras([tshp[i].Transθx for i in 1:length(tshp)])
        ep = ["θx_"*string(v) for v in ep]
        valsθx = hcat(valsθx, ev)
        nmsθx = string.(vcat(nmsθx..., string.(ep)...)) # very bad...
    end


    nmsθy, valsθy = _params_to_array([tshp[i].Transθy for i in 1:length(tshp)])
    if !(typeof(tshp[1].Transθy) <: ConstTransition)
        ep, ev = Turing.Inference.get_transition_extras([tshp[i].Transθy for i in 1:length(tshp)])
        ep = ["θy_"*string(v) for v in ep]
        valsθy = hcat(valsθy, ev)
        nmsθy = string.(vcat(nmsθy..., ep...)) # very bad...
    end


    # Get the values of the extra parameters in each Transition struct.
    #extra_paramsθx, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθx for i in 1:length(tshp)])
    #extra_paramsθy, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθy for i in 1:length(tshp)])
    # Extract names & construct param array.
    nms = string.(vcat(nmsθx...,nmsθy...))
    parray = hcat(valsθx, valsθy )
    return nms, parray
end

function bundle_transitions(tshp::Vector{ConstTransition})
    if tshp[1].θ !== NamedTuple()
        nmsθx, valsθx = _params_to_array(tshp)
        nms = string.(nmsθx)
        parray = valsθx
        return nms, parray
    end
    return [],[]
end

#function bundle_transitions(tshp::Vector{<:Union{Turing.Inference.Transition, AdvancedPS.PFTransition}})
function bundle_transitions(tshp::Vector{<:Turing.Inference.Transition})

    nmsθx, valsθx = _params_to_array(tshp)

    ep, ev = Turing.Inference.get_transition_extras(tshp)
    ep = [string(v) for v in ep]
    valsθx = hcat(valsθx, ev)
    nmsθx = string.(vcat(nmsθx..., string.(ep)...)) # very bad...
    nms = string.(nmsθx)
    parray = valsθx
    return nms, parray
end


function extend_str(str::String, i::Integer)
    ileft = 0
    iright =0
    ileft = findlast("[",str)[1]
    iright =  findlast("]",str)[1]
    @assert ileft >1 && ileft+3 < iright "[Bundle Samples] misspecification of states"
    strsub = SubString(str, ileft+1, iright-1)
    ls = eval(Meta.parse(strsub))
    name = SubString(str,1, ileft-1)
    r =  name*"[$(ls[1]),$(ls[2]),$i]"
    return r
end
function renameX(str::String, j::Integer, i::Integer)
    ileft = 0
    iright =0
    ileftr = findlast("[",str)
    irightr =  findlast("]",str)
    if ileftr != nothing
        @assert irightr != nothing && ileftr[1] >1 && ileftr[1]+1 < irightr[1] "[Bundle Samples] States are named in a wrong way."
        ileft = ileftr[1]
        iright = irightr[1]
        strsub = SubString(str, ileft+1, iright-1)
        idx = eval(Meta.parse(strsub))[1]
        name = SubString(str,1, ileft[1]-1)
        r =  name*"[$j,$idx,$i]"
        return r
    else
        return str*"[$j,$i]"
    end
    return r
end

function renamex0(str::String, i::Integer)
    ileft = 0
    iright =0
    ileft = findlast("[",str)
    iright =  findlast("]",str)
    if ileft != nothing
        @assert iright != nothing && ileft[1] >1 && ileft[1]+1 < iright[1] "[Bundle Samples] States are named in a wrong way."
        strsub = SubString(str, ileft[1]+1, iright[1]-1)
        idx = eval(Meta.parse(strsub))[1]
        name = SubString(str,1, ileft[1]-1)
        r =  name*"[$idx,1,$i]"
        return r
    else
        return str*"[$i]"
    end
end





function get_transition_extras(ts::Vector{T}) where T<:AbstractTransition
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)
    # Get the values of the extra parameters.
    local extra_names
    all_vals = []
    # Iterate through each transition.
    for t in ts
        extra_names = String[]
        vals = []
        # Iterate through each of the additional field names
        # in the struct.
        for p in extra_params
            # Check whether the field contains a NamedTuple,
            # in which case we need to iterate through each
            # key/value pair.
            prop = getproperty(t, p)
            if prop isa NamedTuple
                for (k, v) in pairs(prop)
                    push!(extra_names, string(k))
                    push!(vals, v)
                end
            else
                push!(extra_names, string(p))
                push!(vals, prop)
            end
        end
        push!(all_vals, vals)
    end
    # Convert the vector-of-vectors to a matrix.
    valmat = [all_vals[i][j] for i in 1:length(ts), j in 1:length(all_vals[1])]
    return extra_names, valmat
end





function flatten(names, value::AbstractArray, k::String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                name = k * name
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, v[i])
                push!(names, name)
            elseif isa(v[i], AbstractArray)
                name = k * string(ind2sub(size(v), i))
                flatten(names, value, name, v[i])
            else
                error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
            end
        end
    else
        error("Unknown var type: typeof($v)=$(typeof(v))")
    end
    return
end

function flatten_namedtuple(nt::NamedTuple{pnames}) where {pnames}
    vals = Vector{Real}()
    names = Vector{AbstractString}()
    for k in pnames
        v = nt[k]
        if length(v) == 1
            flatten(names, vals, string(k), v)
        else
            for (vnval, vn) in zip(v[1], v[2])
                flatten(names, vals, vn, vnval)
            end
        end
    end
    return names, vals
end




function _params_to_array(ts::Vector)
    names = Vector{String}()
    dicts = Vector{Dict{String, Any}}()
    # Extract the parameter names and values from each transition.
    init = false
    for t in ts
        nms, vs = flatten_namedtuple(t.θ)
        if length(nms) >0
            if !init
                names =Vector{String}(nms) # We want to keep the ordering!
                init = true
            end
            # Convert the names and values to a single dictionary.
            d = Dict{String, Any}()
            for (k, v) in zip(nms, vs)
                d[k] = v
            end
            push!(dicts, d)
        end
    end
    # Convert the set to an ordered vector so the parameter ordering
    # is deterministic.
    ordered_names = names


    vals = Matrix{Union{Real, Missing}}(undef, length(ts), length(ordered_names))
    # Place each element of all dicts into the returned value matrix.
    for i in eachindex(dicts)
        for (j, key) in enumerate(ordered_names)
            vals[i,j] = get(dicts[i], key, missing)
        end
    end
    return ordered_names, vals
end

ind2sub(v, i) = Tuple(CartesianIndices(v)[i])


@inline function tonamedtuple(X::Array{T,2}) where T<:Real
    lnames = []
    lvals = []
    for i in 1:size(X)[2]
        tmp = [Symbol("x[$k, $i]") for k = 1:size(X)[1]]
        lnames = append!(lnames, tmp )
        tmp = [X[k, i] for k = 1:size(X)[1]]
        lvals = append!(lvals, tmp)
    end
    tnames = Tuple(lnames)
    tvalues = Tuple(lvals)
    return namedtuple(tnames, tvalues)
end
function tonamedtuple(x::AbstractMatrix, varname)
    lnames = []
    lvals = []
    for i in 1:size(x)[2]
        tmp = [Symbol(varname*"[$k, $i]") for k = 1:size(x)[1]]
        lnames = append!(lnames, tmp )
        tmp = [x[k, i] for k = 1:size(x)[1]]
        lvals = append!(lvals, tmp)
    end
    tnames = Tuple(lnames)
    tvalues = Tuple(lvals)
    return namedtuple(tnames, tvalues)
end
function tonamedtuple(x::AbstractArray{T, 3}, varname) where T
    lnames = []
    lvals = []
    for l = 1:size(x)[3]
        for i in 1:size(x)[2]
            tmp = [Symbol(varname*"[$k,$i,$l]") for k = 1:size(x)[1]]
            lnames = append!(lnames, tmp )
            tmp = [x[k, i, l] for k = 1:size(x)[1]]
            lvals = append!(lvals, tmp)
        end
    end
    tnames = Tuple(lnames)
    tvalues = Tuple(lvals)
    return namedtuple(tnames, tvalues)
end

function tonamedtuple_smc(x::AbstractArray{T, 3}, varname) where T
    lnames = []
    lvals = []
    for l = 1:size(x)[3]
        for i in 1:size(x)[2]
            tmp = [Symbol(varname*"[$l][$k,$i]") for k = 1:size(x)[1]]
            lnames = append!(lnames, tmp )
            tmp = [x[k, i, l] for k = 1:size(x)[1]]
            lvals = append!(lvals, tmp)
        end
    end
    tnames = Tuple(lnames)
    tvalues = Tuple(lvals)
    return namedtuple(tnames, tvalues)
end



#
# struct LatentTransition{T} <: AbstractTransition
#     θ::Vector{T}
# end
#
#
# # We have to leave this untyped...
# struct MHGibbsOldTransition{TX, TF} <: AbstractTransition
#     TransX::TX
#     TransF::TF
# end
# struct MHGibbsTransition{TX,Tx0, TF} <: AbstractTransition
#     Transx0::Tx0
#     TransX::TX
#     TransF::TF
# end
#
# struct VITransition{TMu1,TMu2, THP} <: AbstractTransition
#     TransHP::THP
#     TransMu1::TMu1
#     TransMu2::TMu2
# end
#
#
#
# function bundle_transitions(tslt::Vector{LatentTransition})
#     nms =[]
#     pval = []
#
#     for j in 1:length(tslt[1].θ)
#         nmsx, valsx = _params_to_array([tslt[i].θ[j] for i in 1:length(tslt)])
#         for k in 1:length(nmsx)
#             nmsx[k] = extend_str(string(nmsx[k]),j)
#         end
#         if !(typeof(tslt[1].θ[j]) <:ConstTransition)
#             extra_paramsx, extra_valuesx = get_transition_extras([tslt[i].θ[j] for i in 1:length(tslt)])
#             for k in 1:length(extra_paramsx)
#                 extra_paramsx[k]  = extend_str(string.(extra_paramsx[k]),j)
#             end
#             nms = vcat(nms..., nmsx..., extra_paramsx...)
#             push!(pval,hcat(valsx, extra_valuesx))
#         else
#             nms = vcat(nms..., nmsx... )
#             push!(pval,valsx)
#         end
#     end
#     parray = hcat(pval...)
#     return nms, parray
# end

#
# function bundle_transitions(tshp::Vector{VITransition})
#     nmsθhp, valsθhp = _params_to_array([tshp[i].TransHP for i in 1:length(tshp)])
#     nmsθmu1, valsθmu1 = _params_to_array([tshp[i].TransMu1 for i in 1:length(tshp)])
#     nmsθmu2, valsθmu2 = _params_to_array([tshp[i].TransMu2 for i in 1:length(tshp)])
#
#     nms = string.(vcat(nmsθhp...,nmsθmu1...,nmsθmu2...))
#     parray = hcat(valsθhp, valsθmu1, valsθmu2)
#     return nms, parray
# end
#
# function bundle_transitions(tshp::Vector{MHGibbsOldTransition})
#     nmsθx, valsθx = _params_to_array([tshp[i].TransX for i in 1:length(tshp)])
#     ep, ev = Turing.Inference.get_transition_extras([tshp[i].TransX for i in 1:length(tshp)])
#     ep = [string(v) for v in ep]
#     valsθx = hcat(ev, valsθx)
#     nmsθx = string.(vcat(string.(ep)...,nmsθx...)) # very bad...
#     nmsθy, valsθy = _params_to_array([tshp[i].TransF for i in 1:length(tshp)])
#
#     # Get the values of the extra parameters in each Transition struct.
#     #extra_paramsθx, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθx for i in 1:length(tshp)])
#     #extra_paramsθy, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθy for i in 1:length(tshp)])
#     # Extract names & construct param array.
#     nms = string.(vcat(nmsθx...,nmsθy...))
#     parray = hcat(valsθx, valsθy )
#     return nms, parray
# end
#
#
# function bundle_transitions(tshp::Vector{MHGibbsTransition})
#     nmsX = []
#     valsX = []
#     for i = 1:get_size()[3]
#         nmsθx, valsθx0 = _params_to_array([tshp[k].Transx0[i] for k in 1:length(tshp)])
#         ep, ev = Turing.Inference.get_transition_extras([tshp[k].Transx0[i] for k in 1:length(tshp)])
#         ep = [string(v) for v in ep]
#         valsθx0 = hcat(ev, valsθx0)
#         nmsθx = string.(vcat(string.(ep)...,nmsθx...)) # very bad...
#         nmsθx = string.(vcat([renamex0(v, i) for v in nmsθx]...))
#         push!(nmsX, nmsθx)
#         push!(valsX, valsθx0)
#         for j = 1:get_size()[1]
#             nmsθx, valsθx = _params_to_array([tshp[k].TransX[j,i]  for k in 1:length(tshp)])
#             ep, ev = Turing.Inference.get_transition_extras([tshp[k].TransX[j,i] for k in 1:length(tshp)])
#             ep = [string(v) for v in ep]
#             valsθx = hcat(ev, valsθx)
#             nmsθx = string.(vcat(string.(ep)...,nmsθx...)) # very bad...
#             nmsθx = string.(vcat([renameX(v, j, i) for v in nmsθx]...))
#             push!(nmsX, nmsθx)
#             push!(valsX, valsθx)
#         end
#     end
#
#     nmsθF, valsθF = _params_to_array([tshp[i].TransF for i in 1:length(tshp)])
#
#     # Get the values of the extra parameters in each Transition struct.
#     #extra_paramsθx, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθx for i in 1:length(tshp)])
#     #extra_paramsθy, extra_valuesθx = AdvancedPS.get_transition_extras([tshp[i].Transθy for i in 1:length(tshp)])
#     # Extract names & construct param array.
#     nms = string.(vcat(vcat(nmsX...)...,nmsθF...))
#     parray = hcat(valsX..., valsθF )
#     return nms, parray
# end
#
