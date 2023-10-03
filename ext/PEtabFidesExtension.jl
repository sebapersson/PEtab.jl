module PEtabFidesExtension

using CSV
using PyCall
using SciMLBase
using QuasiMonteCarlo
using Random
using Printf
using YAML
using PEtab 

include(joinpath(@__DIR__, "PEtabFidesExtension/", "Setup_fides.jl"))

end