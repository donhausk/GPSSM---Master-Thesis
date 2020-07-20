"""

We need to make some extensions for DynamicPPL. Especially, we remove the dependency on the sampler for all methods
manipulating VarInfo.

"""


using DynamicPPL
using DynamicPPL: Selector, TypedVarInfo


tonamedtuple(vi::Turing.VarInfo) = Turing.tonamedtuple(vi)
@nograd function varname(name::String)
    sym_str, inds = DynamicPPL.split_var_str(name, String)
    sym = Symbol(sym_str)
    return DynamicPPL.VarName{sym}(inds)
end

r = Base.RefValue{Float64}(3.0)


function NewVarInfo(old_vi::TypedVarInfo, ::Val{space}, x::AbstractVector) where space
    md = DynamicPPL.newmetadata(old_vi.metadata, Val(space), x)
    VarInfo(md, Base.RefValue{eltype(x)}(DynamicPPL.getlogp(old_vi)), Ref(DynamicPPL.get_num_produce(old_vi)))
end

function DynamicPPL.updategid!(vi::AbstractVarInfo, sel::Selector, vn::VarName)
    DynamciPPL.setgid!(vi, sel, vn)
end

# Get all indices of variables belonging to a given sampler

@inline function  DynamicPPL._getidcs(vi::TypedVarInfo, s::Selector)
    return DynamicPPL._getidcs(vi.metadata, s)
end
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@generated function DynamicPPL._getidcs(metadata::NamedTuple{names}, s::Selector) where {names}
    exprs = []
    # Iterate through each varname in metadata.
    for f in names
        # If the varname is in the sampler space
        # or the sample space is empty (all variables)
        # then return the indices for that variable.
        push!(exprs, :($f = DynamicPPL.findinds(metadata.$f, s)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

@inline function DynamicPPL._getranges(vi::AbstractVarInfo, s::Selector, space)
    return DynamicPPL._getranges(vi, DynamicPPL._getidcs(vi, s))
end
@inline function DynamicPPL._getranges(vi::AbstractVarInfo, s::Selector)
    return DynamicPPL._getranges(vi, DynamicPPL._getidcs(vi, s))
end


@inline function DynamicPPL.findinds(f_meta, s::Selector)
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter((i) ->
        (s in f_meta.gids[i] || isempty(f_meta.gids[i])), 1:length(f_meta.gids))
end


function DynamicPPL.link!(vi::TypedVarInfo, s::Selector)
    vns = DynamicPPL._getvns(vi, s)
    return DynamicPPL._link!(vi.metadata, vi, vns, Val(()))
end

function invlink!(vi::TypedVarInfo, s::Selector)
    vns = DynamicPPL._getvns(vi, s)
    return DynamicPPL._invlink!(vi.metadata, vi, vns, Val(()))
end

function DynamicPPL._getvns(vi::TypedVarInfo, s::Selector)
    return DynamicPPL._getvns(vi.metadata, DynamicPPL._getidcs(vi, s))
end


function DynamicPPL.setindex!(vi::TypedVarInfo, val, s::Selector)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = DynamicPPL._getranges(vi, s)
    DynamicPPL._setindex!(vi.metadata, val, ranges)
    return val
end

function DynamicPPL.getindex(vi::TypedVarInfo, s::Selector)
    # Gets the ranges as a NamedTuple
    ranges = DynamicPPL._getranges(vi, s)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return vcat(DynamicPPL._getindex(vi.metadata, ranges)...)
end

###

### Some function which have been overloaded from Turing.

###

# We only need the typed variant of the function!
@inline function _getidcs_nonempty(vi::UntypedVarInfo, s::Selector)
    error("This function should not be called, please report this error!s")
end

@inline function _getidcs_nonempty(vi::TypedVarInfo, s::Selector)
    return _getidcs_nonempty(vi.metadata, s)
end




# Get a NamedTuple for all the indices belonging to a given selector for each symbol

@generated function _getidcs_nonempty(metadata::NamedTuple{names}, s::Selector) where {names}
    exprs = []
    # Iterate through each varname in metadata.
    for f in names
        # If the varname is in the sampler space
        # or the sample space is empty (all variables)
        # then return the indices for that variable.
        push!(exprs, :($f = findinds_nonempty(metadata.$f, s)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end


@inline function findinds_nonempty(f_meta, s) where {space}
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter((i) ->
        (s in f_meta.gids[i])
        ,1:length(f_meta.gids))
end

# A bit sloppy...
@inline function _getspace(vi::AbstractVarInfo, s::Selector)
    idcs = DynamicPPL._getidcs(vi, s)
    space = []
    for (i,idc) in enumerate(idcs)
        if length(idc) >0
            push!(space, keys(idcs)[i])
        end
    end

    return Val(Tuple(space))
end




"""

These are Placeholder distributions to make the GPSpec compatible with Truing VI. Important, we need to have positive definite Placeholder Distributions as well.


"""


# Empty Distributions to push vi!
struct MVPlaceHolderDist<:ContinuousMultivariateDistribution
    size::Tuple
end
struct MatPlaceHolderDist <:ContinuousMatrixDistribution
    size::Tuple
end


MatPlaceHolderDist(T::AbstractMatrix{<:Real}) = MatPlaceHolderDist(size(T))
MatPlaceHolderDist(T::AbstractArray{<:Real,3}) = MatPlaceHolderDist(size(T))
MVPlaceHolderDist(T::AbstractVector{<:Real}) = MVPlaceHolderDist(size(T))

Bijectors._invlink(dist::MatPlaceHolderDist, val::AbstractMatrix{<:Real}) = val
Bijectors._invlink(dist::MatPlaceHolderDist, val::AbstractArray{<:Real,3}) = val
Bijectors._invlink(dist::MVPlaceHolderDist, val::AbstractVector{<:Real}) = val
Bijectors._link(dist::MatPlaceHolderDist, val::AbstractArray{<:Real,3}) = val
Bijectors._link(dist::MatPlaceHolderDist, val::AbstractMatrix{<:Real}) = val
Bijectors._link(dist::MVPlaceHolderDist, val::AbstractVector{<:Real}) = val

Base.size(T::Union{MatPlaceHolderDist, MVPlaceHolderDist}) = T.size

get_placeholder_dist(T::Real) = Normal() # Placeholder distribution.
get_placeholder_dist(T::AbstractVector{<:Real}) = MVPlaceHolderDist(T)
get_placeholder_dist(T::AbstractMatrix{<:Real}) = MatPlaceHolderDist(T)
get_placeholder_dist(T::AbstractArray{<:Real,3}) = MatPlaceHolderDist(T) # Ugly...

# Some placeholder distributinos for Bijectors!

# We are instantiating arbitrary positive definite distributions to be compatible with Bijectors. Note that they will only be used for
# to determine which invlink function should be called.
PDMatPlaceHolderDist(T) =  Wishart(length(T),Array{eltype(T),2}(I, size(T))) # This is important for reconstructino!
PositivePlaceHolderDist(T) =  Exponential()
MvLogNormalPlaceHolderDist(T) = MvLogNormal(ones(length(T)),0.0)

struct PDMatPlaceHolderDist_NH <:ContinuousMatrixDistribution
    size::Tuple
end
Base.size(T::PDMatPlaceHolderDist_NH) = T.size

PDMatPlaceHolderDist_NH(T::AbstractMatrix{<:Real}) = PDMatPlaceHolderDist_NH(size(T))

get_replace_diag_mat(size) = UpperTriangular(ones(size,size) - Diagonal(ones(size)))

function replace_diag(f, X::UpperTriangular)
    # We assume they are squared matrices
    #Xn = X .* get_replace_diag_mat(size(X)[1])
    #return UpperTriangular(Xn + Diagonal(f.(diag(X))))
    g(i, j) = ifelse(i == j, f(X[i, i]), X[i, j])
    return g.(1:size(X, 1), (1:size(X, 2))')
end

u = UpperTriangular(ones(10,10))

# We need to ensure that the matrix is symmetric, which may not be the case due to rounding errors
Bijectors.invlink(dist::PDMatPlaceHolderDist_NH, val::AbstractMatrix{<:Real}) =UpperTriangular(replace_diag(exp ,UpperTriangular(val)))
Bijectors.link(dist::PDMatPlaceHolderDist_NH, val::AbstractMatrix{<:Real}) = UpperTriangular(replace_diag(log ,UpperTriangular(val)))

Bijectors.logpdf_with_trans(d::PDMatPlaceHolderDist_NH, x::AbstractMatrix{<:Real}, transform::Bool = true) = error("The logpdf_with_trans must be defined for the UpperTriangular case!")

#Bijectors.logpdf_with_trans(d::PDMatPlaceHolderDist_NH, x::AbstractMatrix{<:Real}, transform::Bool = true) = logpdf_with_trans(PDMatPlaceHolderDist(dist.size), Symmetric(x), transform)

get_placeholder_dist_positive(T::Real) = PositivePlaceHolderDist(T)
get_placeholder_dist_positive(T::AbstractVector{<:Real}) = MvLogNormalPlaceHolderDist(T)
get_placeholder_dist_positive(T::AbstractMatrix{<:Real}) = PDMatPlaceHolderDist(T)
get_placeholder_dist_positive(T::UpperTriangular{<:Real}) = PDMatPlaceHolderDist_NH(T)
get_placeholder_dist_positive(T::AbstractArray{<:Real, 3}) = error("This is not yet supported. The inducing points can not be forced to be positive definite.")




Zygote.@nograd function increment_pos!(v::Vector)
   v[1] += 1
   v[1]-1
end

Zygote.@nograd get_bijector(dist::Distributions.Product, pos::Vector{<:Integer}) = bijector(dist.v[increment_pos!(pos)])
apply_bij(bij,v::Real) = bij(v)

# We need to have a binectors extenstion for this
function Bijectors.link(dist::Distributions.Product, x::AbstractVector{<:Real})
   pos = ones(Integer,1)
   [ apply_bij(get_bijector(dist,pos),v) for v in x ]
end
# We need to have a binectors extenstion for this
function Bijectors.invlink(dist::Distributions.Product, y::AbstractVector{<:Real})
   pos = ones(Integer,1)
   @views [ apply_bij(inv(get_bijector(dist,pos)),v) for v in y ]
end
# We need to have a binectors extenstion for this
#Bijectors.logpdf_with_trans(d::Distributions.Product, x::AbstractVector{<:Real}, transform::Bool = true) = transform ? logpdf_forward(transformed(d), x) : logpdf(d, x)
Bijectors.logpdf_with_trans(d::Distributions.Product, x::AbstractVector{<:Real}, transform::Bool = true) = transform ? sum(map(i -> logpdf_forward(transformed(d.v[i]),x[i]), 1:length(x)))  : logpdf(d, x)




# Quite dirty Code for Zygote - get item is very expensive in Zygote even if X does not depend on any hyper parameters.
# Should be rewritten in a compact and nicer way.
##########################################################################

@nograd nograd_get_index(X, idcs) = @view X[idcs...]
@nograd get_identity_mat(i) = Matrix{get_latent_type()}(I, i, i)
get_metadata_from_vi(vi::DynamicPPL.VarInfo{<:NamedTuple}, s::String, i::Integer) = vi.metadata[Symbol(s*"$i")]



##########################################################################

"""
Zygote compatible Invlink.

Attention, Invlink is not Zygote Compatible, therefore, provide a small workaround !!!
Thus only positive definite parameters are allowed.
"""
@nograd get_range_el(ranges,i) = ranges[i]
@nograd get_dist_el(dists,i) = dists[i]
@nograd check_if_trans_and_single(vn) = ( vn.flags["trans"].== 1, length(vn.ranges) ==1)


# Quite dirty Code for Zygote - get item is very expensive in Zygote even if X does not depend on any hyper parameters.
# Should be rewritten in a compact and nicer way.
##########################################################################

@inline function reconstruct(v::AbstractVector, d::Distribution)
    return isempty(size(d)) ? v[1] : reshape(v,size(d))
end

@inline function _invlink_nt(vn::DynamicPPL.Metadata{Names}) where Names
    flag, len = check_if_trans_and_single(vn)
    if len >1
        return vcat(map(i ->  flag[i] == true ?
            (isempty(size( get_dist_el(vn.dists,i))) ? Bijectors._invlink(get_dist_el(vn.dists,1), reconstruct(vn.vals[get_range_el(vn.ranges,i)], get_dist_el(vn.dists,i))) :
             Bijectors.invlink(get_dist_el(vn.dists,i), reconstruct(vn.vals[get_range_el(vn.ranges,i)], get_dist_el(vn.dists,i)))) :
             reconstruct(vn.vals[get_range_el(vn.ranges,i)], get_dist_el(vn.dists,i)), 1:length(vn.ranges))...)
    else
        if flag[1]
            #Bijectors is broken at the moment for zygote...
            # Therefore, we need to make the following hack...

            val = reconstruct(vn.vals,get_dist_el(vn.dists,1))
            if typeof(val) <: Real
                return Bijectors._invlink(get_dist_el(vn.dists,1), val)
            end
            return Bijectors.invlink(get_dist_el(vn.dists,1), val)
        end
    end
    return reconstruct(vn.vals, get_dist_el(vn.dists,1))
end


@inline function _invlink_nt(vn::DynamicPPL.Metadata{Names}, size::Tuple) where Names
    flag, len = check_if_trans_and_single(vn)
    @assert len == 1 "[GPSSM] In Utilities _invlink_nt, the length should be one!"
    val = isempty(size) ? vn.vals[1] : reshape(vn.vals, size)
    if flag[1]
        # This is a hack around Bijectors issues.
        return Bijectors.invlink(get_dist_el(vn.dists,1), val)
    end
    return val
end

@inline function fix_scalar(size::Tuple)
    return size
end


"""

Functions to extract the hyer parameter from the Turing VarInfor struct.

"""

@inline function get_distθx0_args(vi::UntypedVarInfo,  hp::AbstractGPSpec)
    tnames = Symbol[]
    tvals = []
    for key in keys(hp.θx0keys)
        #push!(tnames,key)
        vn = varname(string(key))
        val = isempty(getfield(hp.θx0keys, key)) ? vi[vn] : reshape(vi[vn], getfield(hp.θx0keys, key)...)
        push!(tvals, val )
    end
    return NamedTupleTools.namedtuple(keys(hp.θx0),Tuple(tvals))
end


@inline function get_distθx0_args(vi::TypedVarInfo,  hp::AbstractGPSpec)
    return _get_gp_args_s(vi.metadata, hp.θx0, hp.θx0keys)
end




@inline function get_kernel_args(vi::UntypedVarInfo,  hp::AbstractGPSpec)
    nt = []
    for key in keys(hp.θxkernel)
        tnames = Symbol[]
        tvals = []
        vkeys = keys(hp.θxkernel[key])
        nvals = hp.θxkernelkeys[key]
        nkeys = keys(hp.θxkernelkeys[key])
        nlist = hp.θxkernelkeys[key]
        for i = 1:length(nkeys)
            key_tuple = vkeys[i]
            push!(tnames,key_tuple)
            vn = varname(string(nkeys[i]))
            push!(tvals, isempty(nlist[i]) ? vi[vn] : reshape(vi[vn], nlist[i]...))
        end
        push!(nt, NamedTupleTools.namedtuple(Tuple(tnames),Tuple(tvals)))
    end
    return NamedTuple{keys(hp.θxkernel)}(Tuple(nt))
end

@inline function get_kernel_args(vi::TypedVarInfo,  hp::AbstractGPSpec)
    return _get_gp_args(vi.metadata, hp.θxkernel, hp.θxkernelkeys)
end

@inline function get_mean_args(vi::UntypedVarInfo,  hp::AbstractGPSpec)
    nt = []
    for key in keys(hp.θxmean)
        tnames = Symbol[]
        tvals = []
        vkeys = keys(hp.θxmean[key])
        nvals = hp.θxmeankeys[key]
        nkeys = keys(hp.θxmeankeys[key])
        nlist =hp.θxmeankeys[key]
        for i = 1:length(nkeys)
            key_tuple = vkeys[i]
            push!(tnames, key_tuple)
            vn = varname(string(nkeys[i]))
            val = isempty(nlist[i]) ? vi[vn] : reshape(vi[vn], nlist[i]...)
            push!(tvals, val )
        end
        push!(nt, NamedTupleTools.namedtuple(Tuple(tnames),Tuple(tvals)))
    end
    return  NamedTuple{keys(hp.θxmean)}(Tuple(nt))
end

@inline function get_mean_args(vi::TypedVarInfo,  hp::AbstractGPSpec)
    return  _get_gp_args(vi.metadata, hp.θxmean, hp.θxmeankeys)
end


@generated function _get_gp_args(metadata::NamedTuple{VI}, vlist::NamedTuple{vnames}, nlist::NamedTuple{lnames}) where {VI, lnames, vnames}
    exprs = []
    #lnames == vnames == ("d1", "d2"...)
    for key in lnames
        push!(exprs, :($key = _get_gp_args_s(metadata, vlist.$key, nlist.$key)))
    end
    length(exprs) == 0 && return :(Tuple())
    return :($(exprs...),)
end

@generated function _get_gp_args_s(metadata::NamedTuple{VI}, vlist::NamedTuple{vnames}, nlist::NamedTuple{lnames}) where {VI, lnames, vnames}
    exprs = []
    for i = 1:length(lnames)
        vn = lnames[i]
        key = vnames[i]
        #push!(exprs, :($key = isempty(nlist.$vn) ? _invlink_nt(metadata.$vn) : reshape(_invlink_nt(metadata.$vn), nlist.$vn)))

        push!(exprs, :($key = _invlink_nt(metadata.$vn, nlist.$vn) ))

    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end




@inline  function get_distθy_args(vi::UntypedVarInfo,  hp::AbstractGPSpec)
    tnames = Symbol[]
    tvals = []
    for key in keys(hp.θykeys)
        push!(tnames, key)
        vn = varname(string(key))
        val = isempty(getfield(hp.θykeys, key)) ? vi[vn] : reshape(vi[vn], getfield(hp.θykeys, key)...)
        push!(tvals, val)
    end
    return NamedTupleTools.namedtuple(keys(hp.θy),Tuple(tvals))
end


@inline function get_distθy_args(vi::TypedVarInfo,  hp::AbstractGPSpec)
    return _get_gp_args_s(vi.metadata, hp.θy, hp.θykeys)
end


@inline function get_Q(vi::TypedVarInfo)
    return _invlink_nt(vi.metadata.Q)
end
