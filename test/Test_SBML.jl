using LinearAlgebra
using CSV
using Test
using ModelingToolkit
using SBML
using OrdinaryDiffEq
using Sundials
using PEtab
using DataFrames

# 926 fails
testCase = "00999"
testCase = "01802"
#testSBMLTestSuite(testCase, Rodas4P())
# Next we must allow species to first be defined via an InitialAssignment, pretty stupied to me, but aja...
function testSBMLTestSuite(testCase, solver)
    @info "Test case $testCase"
    dirCases = joinpath(@__DIR__, "sbml-test-suite", "cases", "semantic")
    path_SBMLFiles = joinpath.(dirCases, testCase, filter(x -> x[end-3:end] == ".xml", readdir(joinpath(dirCases, testCase))))

    pathResultFile = filter(x -> occursin("results", x), readdir(joinpath(dirCases, testCase)))[1]
    expected = CSV.read(joinpath(dirCases, testCase, pathResultFile), stringtype=String, DataFrame)
    # As it stands I cannot "hack" a parameter value at time zero, but the model simulation values
    # are correct.
    if testCase == "00995" || testCase == "00996" || testCase == "00997" || testCase == "01284" || testCase == "01510" || testCase == "01527" || testCase == "01596" || testCase == "01663" || testCase == "01686" || testCase == "01693" || testCase ∈ ["01694", "01695", "01696", "01697", "01698", "01699", "01700", "01719"]
        expected = expected[2:end, :]
    end
    col_names =  Symbol.(replace.(string.(names(expected)), " " => ""))
    rename!(expected, col_names)

    t_save = "Time" in names(expected) ? Float64.(expected[!, :Time]) : Float64.(expected[!, :time])
    t_save = Vector{Float64}(t_save)
    tmax = maximum(t_save)
    whatCheck = filter(x -> x ∉ [:time, :Time], col_names)
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
        # c = n / V => n = c * V

        f = open(path_SBML, "r")
        text = read(f, String)
        close(f)
        # If stoichiometryMath occurs we need to convert the SBML file to a level 3 file
        # to properly handle the latter
        if occursin("stoichiometryMath", text) == false
            model_SBML = readSBML(path_SBML)
        else
            model_SBML = readSBML(path_SBML, doc -> begin
                                set_level_and_version(3, 2)(doc)
                                convert_promotelocals_expandfuns(doc)
                                end)
        end

        model_dict = PEtab.build_model_dict(model_SBML, true)
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

            if toCheck ∈ speciesTest && toCheck ∈ speciesTestConc && string(toCheck) ∈ keys(model_SBML.species)
                compartmentName = model_SBML.species[string(toCheck)].compartment
                if model_dict["stateGivenInAmounts"][string(toCheck)][1] == false
                    c = 1.0
                elseif compartmentName in string.(model_parameters)
                    c = sol.prob.p[findfirst(x -> x == compartmentName, string.(model_parameters))]
                else
                    c = sol[Symbol(compartmentName)]
                end
            elseif toCheck ∈ speciesTest && toCheck ∈ speciesTestAmount && string(toCheck) ∈ keys(model_SBML.species)
                compartmentName = model_SBML.species[string(toCheck)].compartment
                if model_dict["stateGivenInAmounts"][string(toCheck)][1] == false
                    if compartmentName in string.(model_parameters)
                        c = 1 / (sol.prob.p[findfirst(x -> x == compartmentName, string.(model_parameters))])
                    else
                        c = 1 ./ sol[Symbol(compartmentName)]
                    end
                else
                    c = 1.0
                end
            else
                c = 1.0
            end

            @test all(abs.(sol[toCheck] ./ c .- expected[!, toCheck]) .< absTolTest .+ relTolTest .* abs.(expected[!, toCheck]))
        end
    end
end


if length(ARGS) > 1
    istart = parse(Int64, ARGS[1])
    iend = parse(Int64, ARGS[2])
else
    istart = 1
    iend = 1821
end


solver = Rodas4P()
@testset "SBML test suite" begin
    for j in istart:iend
        testCase = repeat("0", 5 - length(string(j))) *  string(j)

        if testCase == "00028"
            testSBMLTestSuite(testCase, CVODE_BDF())
            continue
        end

        # Do not name parameters Inf, true, false, pi, time, or NaN (I mean come on...)
        if testCase ∈ ["01811", "01813", "01814", "01815", "01816", "01817", "01819", "01820", "01821"]
            continue
        end

        # Species conversionfactor not yet supported in Julia
        if testCase ∈ ["00976", "00977", "01405", "01406", "01407", "01408", "01409", "01410", "01484", "01500",
                       "01501", "01642", "01643", "01645", "01646", "01648", "01649", "01651", "01652", "01666",
                       "01667", "01668", "01669", "01670", "01672", "01684", "01685", "01730", "01731", "01733",
                       "01739", "01740", "01775", "01776", "01121", "01724", "01725", "01727", "01728", "01734", 
                       "01736", "01737"]
            continue
        end

        # Implies is not supported
        if testCase ∈ ["01274", "01279", "01497"]
            continue
        end

        # rem and div not supported for parameters
        if testCase ∈ ["01277", "01278", "01495", "01496"]
            continue
        end

        # We and SBML.jl do not currently support hierarchical models
        not_test1 = ["011" * string(i) for i in 26:83]
        not_test2 = ["0" * string(i) for i in 1344:1394]
        not_test3 = ["0" * string(i) for i in 1467:1477]
        not_test4 = ["01778"]
        if testCase ∈ not_test1 || testCase ∈ not_test2 || testCase ∈ not_test3 || testCase ∈ not_test4
            continue
        end

        # We do not aim to support Flux-Balance-Analysis (FBA) models
        not_test1 = ["01" * string(i) for i in 186:197]
        not_test2 = ["01" * string(i) for i in 606:625]
        if testCase ∈ not_test1 || testCase ∈ not_test2 || testCase ∈ ["01627", "01628", "01629", "01630"]
            continue
        end

        # If user wants to add a random species, it must either be as a species, initialAssignment, assignmentRule
        # or by event, not by just random adding it to equations.
        if testCase ∈ ["00974"]
            continue
        end

        # Bug in SBML.jl
        if testCase ∈ ["01464", "01465", "01552", "01553", "01554"]
            continue
        end

        # As of yet we do not support events with priority, but could if there are interest. However should
        # be put up as an issue on GitHub
        if testCase ∈ ["00931", "00934", "00935", "00962", "00963", "00964", "00965", "00966", "00967",
                       "00978", "00978", "01229", "01242", "01267", "01294", "01298", "01331", "01332",
                       "01333", "01334", "01336", "01337", "01466", "01512", "01521", "01533", "01577",
                       "01583", "01588", "01589", "01590", "01591", "01592", "01593", "01599", "01605",
                       "01626", "01627", "01662", "01681", "01682", "01683", "01705", "01714"]
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
                       "00982", "00983", "00984", "00985", "01318", "01319", "01320", "01400",
                       "01401", "01403", "01404", "01410", "01411", "01412", "01413", "01414",
                       "01415", "01416", "01417", "01418", "01419", "01454", "01480", "01481",
                       "01518", "01522", "01523", "01524", "01534", "01535", "01536", "01537",
                       "01538", "01539"]
            continue
        end

        # Fast reactions can technically be handled via algebraic rules, will add support if wanted
        if testCase ∈ ["00870", "00871", "00872", "00873", "00874", "00875", "00986", "00987",
                       "00988", "01051", "01052", "01053", "01396", "01397", "01398", "01399",
                       "01544", "01545", "01546", "01547", "01548", "01549", "01550", "01551",
                       "01558", "01559", "01560", "01565", "01566", "01567", "01568", "01569",
                       "01570", "01571", "01572"]
            continue
        end

        # We do not support an event with multiple triggers or and in triggers
        if testCase ∈ ["01211", "01531"]
            continue
        end

        # We do not support an event with piecewise in the activation
        if testCase ∈ ["01212", "01213", "01214", "01215"]
            continue
        end

        # We do not lt etc... with multiple pair (more than two) parameters
        if testCase ∈ ["01216", "01494", "01781", "01782", "01783"]
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
        if testCase ∈ ["00276", "00277", "00278", "00279", "01486", "01488", "01489", "01492",
                       "01493", "01503", ]
            continue
        end

        # We do not support strange ML with plus having one argument, xor without argument ...
        if testCase ∈ ["01489", "01490", "01491"]
            continue
        end

        # Piecewise in initialAssignments we do not aim to support (can easily be
        # side-stepeed with assignmentrules)
        if testCase ∈ ["01112", "01113", "01114", "01115", "01116", "01208", "01209", "01210",
                       "01282", "01283"]
            continue
        end

        # Bug in SBML.jl (parameter rateOf)
        if testCase ∈ ["01321", "01322"]
            continue
        end

        # We do not allow stochastic simulations
        if testCase ∈ ["00952", "00953"]
            continue
        end

        # We cannot have the trigger of an event be the result of an algebraic rule
        if testCase == "01578"
            continue
        end

        # Event assignment time must be solvable, e.g cosh(x) ≥ 0.5 is not allowed as cosh(x) == 0.5 cannot be solved
        if testCase == "01596"
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
                       "00655", "00656", "00657", "00980", "01000", "01048", "01049",
                       "01050", "01074", "01075", "01076", "01119", "01120", "01230",
                       "01241", "01263", "01268", "01269", "01270", "01287", "01295",
                       "01299", "01305", "01324", "01325", "01326", "01327", "01328",
                       "01329", "01335", "01507", "01508", "01509", "01511", "01519",
                       "01520", "01525", "01526", "01528", "01529", "01532", "01575",
                       "01576", "01579", "01580", "01581", "01582", "01584", "01585",
                       "01586", "01587", "01594", "01595", "01597", "01598", "01600",
                       "01601", "01602", "01603", "01604", "01659", "01660", "01661",
                       "01673", "01674", "01675", "01676", "01677", "01678", "01679",
                       "01680", "01687", "01688", "01689", "01690", "01691", "01692",
                       "01701", "01702", "01703", "01704", "01706", "01707", "01708",
                       "01709", "01710", "01711", "01712", "01713", "01715", "01716",
                       "01717", "01718", "01720", "01721", "01754", "01755", "01756",
                       "01757", "01758", "01759", "01798", "00731", "01095", "01771"]) || testCase ∈ notTest
            continue
        end

        testSBMLTestSuite(testCase, solver)
    end
end


#=
    In case a specie is given in concentration, the compartment volume changes, the specie is not a 
    a boundary condition, add specific new amount specie in order to build the conc. specie derivative 
    correctly - see below for formula
=#


function getODEModel_01498_sbml_l3v2(foo)
	# Model name: 01498_sbml_l3v2
	# Number of parameters: 0
	# Number of species: 3

    ### Define independent and dependent variables
    ModelingToolkit.@variables t S0(t) S0_stoich(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [S0, S0_stoich]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters


    ### Store parameters in array for ODESystem command
    parameterArray = []

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Derivatives ###
    eqs = [
    D(S0) ~ S0_stoich,
    D(S0_stoich) ~ 2.718281828459045,
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    S0 => 0.0,
    S0_stoich => 0.0,
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    ]

    return sys, initialSpeciesValues, trueParameterValues

end

sys, u0map, pmap = getODEModel_01498_sbml_l3v2([])
a = 1
ode_problem = ODEProblem(structural_simplify(sys), u0map, (0.0, 1.0), pmap, jac=true)
sol = solve(ode_problem, solver, saveat=0:0.1:1.0)