using PEtab

path_yaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml") # @__DIR__ = file directory
petab_problem = createPEtabODEProblem(readPEtabModel(path_yaml, verbose=true))
dir_save = nothing

using Optim
res1 = calibrateModelMultistart(petab_problem, IPNewton(), 10, dir_save,
                                options=Optim.Options(iterations = 200,
                                                      show_trace=true))

using PyCall
using QuasiMonteCarlo
res2 = calibrateModelMultistart(petab_problem, Fides(nothing; verbose=true), 10, dir_save,
                                samplingMethod=QuasiMonteCarlo.LatinHypercubeSample())

using Ipopt
res3 = calibrateModelMultistart(petab_problem, IpoptOptimiser(false), 10, dir_save,
                               saveTrace=true,
                               seed=123)
println("x-trace = ", res3.runs[1].xTrace)
println("f-trace = ", res3.runs[1].fTrace)
res3.runs[1].xTrace
res3.runs[1].fTrace

p0 = petab_problem.θ_nominalT .* 0.5
res = calibrateModel(petab_problem, p0, IpoptOptimiser(false),
                     options=IpoptOptions(max_iter = 1000, print_level=5))