using PEtab

path_yaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml") # @__DIR__ = file directory
petab_problem = PEtabODEProblem(PEtabModel(path_yaml, verbose=true))
dir_save = nothing

using Optim
res1 = calibrate_model_multistart(petab_problem, IPNewton(), 10, dir_save,
                                  options=Optim.Options(iterations = 200,
                                  show_trace=true))

using PyCall
using QuasiMonteCarlo
res2 = calibrate_model_multistart(petab_problem, Fides(nothing; verbose=true), 10, dir_save,
                                  sampling_method=QuasiMonteCarlo.LatinHypercubeSample())

using Ipopt
res3 = calibrate_model_multistart(petab_problem, IpoptOptimiser(false), 10, dir_save,
                               save_trace=true,
                               seed=123)
println("x-trace = ", res3.runs[1].xtrace)
println("f-trace = ", res3.runs[1].ftrace)
res3.runs[1].xtrace
res3.runs[1].ftrace

p0 = petab_problem.θ_nominalT .* 0.5
res = calibrate_model(petab_problem, p0, IpoptOptimiser(false),
                     options=IpoptOptions(max_iter = 1000, print_level=5))