module CatalystExtension

using SciMLBase
using DataFrames
using CSV
using RuntimeGeneratedFunctions
using PEtab
using Printf
using DiffEqCallbacks
using Catalyst

import PEtab.getObsOrSdParam

RuntimeGeneratedFunctions.init(@__MODULE__)

# For Optimization and model selection
include(joinpath(@__DIR__, "CatalystExtension", "Read_PEtabModel.jl"))

end