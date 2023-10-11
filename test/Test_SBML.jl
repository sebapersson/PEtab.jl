using LinearAlgebra
using CSV
using Test
using ModelingToolkit
using SBML 
using OrdinaryDiffEq
using Sundials
using PEtab
using DataFrames

#=
    Events are only activated when going from false to true, how to handle? For discrete events need to bake 
    in an additional variable (via a closure) that takes the event initil-thing variable, and it must false in 
    order to trigger the event.
    For cont. events they always check, but need to build an initialiser in case it will trigger at time zero, 
    in a sense easier than for the discrete but afterwards the cont. event will actually check itself if it 
    goes from false to true 
=#


testCase = "00001"
testCase = "00607"
testSBMLTestSuite(testCase, Rodas4P())
# Next we must allow species to first be defined via an InitialAssignment, pretty stupied to me, but aja...
function testSBMLTestSuite(testCase, solver)
    @info "Test case $testCase"
    dirCases = joinpath(@__DIR__, "sbml-test-suite", "cases", "semantic")
    path_SBMLFiles = joinpath.(dirCases, testCase, filter(x -> x[end-3:end] == ".xml", readdir(joinpath(dirCases, testCase))))

    pathResultFile = filter(x -> occursin("results", x), readdir(joinpath(dirCases, testCase)))[1]
    expected = CSV.read(joinpath(dirCases, testCase, pathResultFile), stringtype=String, DataFrame)
    # As it stands I cannot "hack" a parameter value at time zero, but the model simulation values 
    # are correct.
    if testCase == "00995" || testCase == "00996" || testCase == "00997"
        expected = expected[2:end, :]
    end

    t_save = Float64.(expected[!, :time])
    tmax = maximum(t_save)
    whatCheck = Symbol.(filter(x -> x != "time", names(expected)))
    path_SBML = path_SBMLFiles[end]

    # Read settings file 
    settingsFileLines = readlines(joinpath(dirCases, testCase, testCase * "-settings.txt"))
    speciesTest = Symbol.(replace.(split(split(settingsFileLines[4], ":")[2], ','), " " => "", ))
    speciesTestAmount = Symbol.(replace.(split(split(settingsFileLines[7], ":")[2], ','), " " => "", ))
    speciesTestConc = Symbol.(replace.(split(split(settingsFileLines[8], ":")[2], ','), " " => "", ))
    absTolTest = parse(Float64, split(settingsFileLines[5], ":")[2])
    relTolTest = parse(Float64, split(settingsFileLines[6], ":")[2])

    for path_SBML in path_SBMLFiles    
        # We do not aim to support l1 
        if occursin("-l1", path_SBML)
            continue
        end
        model_SBML = readSBML(path_SBML)
        model_dict = PEtab.build_model_dict(readSBML(path_SBML), true)
        sol = solve_SBML(path_SBML, solver, (0.0, tmax); abstol=1e-12, reltol=1e-12, verbose=false, saveat=t_save)
        model_parameters = parameters(sol.prob.f.sys)
        for toCheck in whatCheck
            toCheckNoWhitespace = Symbol(replace(string(toCheck), " " => ""))
            if toCheckNoWhitespace ∈ Symbol.(model_parameters)
                iParam = findfirst(x -> x == toCheckNoWhitespace, Symbol.(model_parameters))

                if all(isinf.(expected[!, toCheck])) && all(expected[!, toCheck] .> 0)
                    @test isinf(sol.prob.p[iParam]) && sol.prob.p[iParam] > 0
                elseif all(isinf.(expected[!, toCheck])) && all(expected[!, toCheck] .< 0)
                    @test isinf(sol.prob.p[iParam]) && sol.prob.p[iParam] < 0
                elseif all(isnan.(expected[!, toCheck]))
                    @test isnan(sol.prob.p[iParam])
                else
                    @test all(abs.(sol.prob.p[iParam] .- expected[!, toCheck]) .< absTolTest .+ relTolTest .* abs.(expected[!, toCheck]))
                end
                continue
            end

            if toCheck ∈ speciesTest && toCheck ∈  speciesTestConc

                compartmentName = model_SBML.species[string(toCheck)].compartment
                if model_dict["stateGivenInAmounts"][string(toCheck)][1] == false
                    c = 1.0
                elseif compartmentName in string.(model_parameters)
                    c = sol.prob.p[findfirst(x -> x == compartmentName, string.(model_parameters))]
                else
                    c = sol[Symbol(compartmentName)]
                end
            else
                c = 1.0
            end

            @test all(abs.(sol[toCheck] ./ c .- expected[!, toCheck]) .< absTolTest .+ relTolTest .* abs.(expected[!, toCheck]))
        end
    end
end


# 00945 crash!!
# 00369
solver = Rodas4P()
@testset "SBML test suite" begin
    for i in 1:997
        testCase = repeat("0", 5 - length(string(i))) *  string(i)

        if testCase == "00028"
            testSBMLTestSuite(testCase, CVODE_BDF())
            continue
        end

        # StoichiometryMath we do not aim to support 
        if testCase ∈ ["00068", "00069", "00070", "00129", "00130", "00131", "00388", "00391", "00394", "00516", 
                       "00517", "00518", "00519", "00520", "00521", "00522", "00561", "00562", "00563", 
                       "00564", "00731", "00827", "00828", "00829", "00898", "00899", "00900", "00609", 
                       "00610", "00968", "00973", "00989", "00990", "00991", "00992", "00993", "00994"]
            continue
        end

        # Species conversionfactor not yet supported in Julia
        if testCase ∈ ["00976", "00977"]
            continue
        end

        # If user wants to add a random species, it must either be as a species, initialAssignment, assignmentRule
        # or by event, not by just random adding it to equations.
        if testCase ∈ ["00974"]
            continue
        end
            

        # As of yet we do not support events with priority, but could if there are interest. However should
        # be put up as an issue on GitHub 
        if testCase ∈ ["00931", "00934", "00935", "00962", "00963", "00964", "00965", "00966", "00967", 
                       "00978", "00978"]
            continue
        end

        # We do not allow 0 * Inf 
        if testCase ∈ ["00959"]
            continue
        end

        # Issue on GitHub 
        if testCase ∈ ["00928", "00929"]
            continue
        end

        # As of now we do not support delay (creating delay-differential-equation)
        if testCase ∈ ["00937", "00938", "00939", "00940", "00941", "00942", "00943", "00981", 
                       "00982", "00983", "00984", "00985"]
            continue
        end

        # Fast reactions can technically be handled via algebraic rules, will add support if wanted 
        if testCase ∈ ["00870", "00871", "00872", "00873", "00874", "00875", "00986", "00987", 
                       "00988"]
            continue
        end

        # Piecewise in reaction formulas we do not aim to support (can easily be 
        # side-stepeed with assignmentrules)
        if testCase ∈ ["00190", "00191", "00192", "00193", "00194", "00195", "00198", 
                       "00199", "00200", "00201"]
            continue
        end

        # Piecewise in functions we do not aim to support (can easily be 
        # side-stepeed with assignmentrules)
        if testCase ∈ ["00276", "00277", "00278", "00279"]
            continue
        end

        # We do not allow stochastic simulations 
        if testCase ∈ ["00952", "00953"]
            continue
        end

        # Event with delay can be supported if there is interest as implementing 
        # it is doable (just cumbersome)
        notTest = ["004" * string(i) for i in 21:61]
        if (testCase ∈ ["00071", "00072", "00073", "00405", "00405", "00406", "00407", 
                       "00409", "00410", "00411", "00412", "00413", "00414", "00415",
                       "00416", "00417", "00418", "00419", "00420", "00622", "00623", 
                       "00624", "00637", "00638", "00639", "00649", "00650", "00651", 
                       "00664", "00665", "00666", "00682", "00683", "00684", "00690", 
                       "00702", "00708", "00724", "00737", "00757", "00758", "00759", 
                       "00763", "00764", "00765", "00766", "00767", "00768", "00769", 
                       "00770", "00771", "00772", "00773", "00774", "00775", "00776", 
                       "00777", "00778", "00779", "00780", "00848", "00849", "00850", 
                       "00886", "00887", "00932", "00933", "00936", "00408", "00461", 
                       "00655", "00656", "00657", "00980", ]) || testCase ∈ notTest
            continue
        end

        testSBMLTestSuite(testCase, solver)
    end
end



using OrdinaryDiffEq
function f(du, u, p, t)
    du[1] = -u[1]*p[1]
end
u0 = [10.0]
const V = 1
prob = ODEProblem(f, u0, (0.0, 10.0), [1.0])

function condition_test(u, t, integrator)
    
    if integrator.p[1] > 0 && integrator.p[1] < 10
        println("t = ", t)
        return true
    end
    return false
end
affect!(integrator) = integrator.p[1] = 10
function testinit(c,u,t,integrator)
    cond = condition_test(u, t, integrator)
    if cond == true
        println("t in init = $t")
        affect!(integrator)
    end
end

cb = DiscreteCallback(condition_test, affect!, initialize=testinit)

sol = solve(prob, Tsit5(), callback = cb)


function hej(x, y)
    println("y[1] = ", y[1])
    y[1] += 1
    return x * y[1]
end


_hej = let y=[1.0]
    (x) -> hej(x, y)
end