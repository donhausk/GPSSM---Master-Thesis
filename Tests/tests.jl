using Revise
push!(LOAD_PATH, "/home/kongi/GPSSM")
using GPSSM
using Random
using Test
using Distributions
using Turing


dir = splitdir(splitdir(pathof(GPSSM))[1])[1]

include(dir*"/Tests/test_utils/AllUtils.jl")
