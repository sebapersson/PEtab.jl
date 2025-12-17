#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#

@testset "Define Julia input format" begin
    # Define reaction network model
    rn = @reaction_network begin
        @parameters a0
        @species A(t)=a0
        (k1, k2), A <--> B
    end
    speciemap = [:B => 1.0] # Constant initial value for B

    # Measurement data
    measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                             obs_id=["obs_a", "obs_a", "obs_a", "obs_a"],
                             time=[0, 10.0, 0, 10.0],
                             measurement=[0.7, 0.1, 0.8, 0.2])
    # Single experimental condition
    simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                 "c1" => Dict(:a0 => 0.9))
    # Observable equation
    @unpack A = rn
    observables = Dict(["obs_a" => PEtabObservable(A, :lin, 1.0)])

    @testset "Non-specified parameters" begin
        simulation_conditions = Dict("c0" => Dict(), "c1" => Dict())
        parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                      PEtabParameter(:k2, value=0.6, scale=:lin)]
        str_warn = "The parameter a0 has not been assigned a value among PEtabParameters, \
                    simulation conditions, or in the parameter map. It default to 0."
        @test_warn str_warn begin
            model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                               simulation_conditions = simulation_conditions)
        end

        simulation_conditions = Dict("c0" => Dict(:a0 => 1.0), "c1" => Dict(:a0 => 2.0))
        parameters = [PEtabParameter(:k1, value=0.8, scale=:lin)]
        str_warn = "The parameter k2 has not been assigned a value among PEtabParameters, \
                    simulation conditions, or in the parameter map. It default to 0."
        @test_warn str_warn begin
            model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                               simulation_conditions = simulation_conditions)
        end
    end

    @testset "Experimental conditions input" begin
        # PEtab-parameter to "estimate"
        parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                      PEtabParameter(:k2, value=0.6, scale=:lin)]

        # Case 1 everything should work
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                     "c1" => Dict(:a0 => 0.9))
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        @test typeof(model) <: PEtabModel

        simulation_conditions = Dict("c0" => Dict(:b0 => 0.8),
                                     "c1" => Dict(:b0 => 0.9))
        @test_throws PEtab.PEtabFileError begin
            model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                               simulation_conditions = simulation_conditions)
        end

        # PEtab-parameter to "estimate"
        parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                      PEtabParameter(:k2, value=0.6, scale=:lin),
                      PEtabParameter(:noise, value=0.6, scale=:lin)]
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                     "c1" => Dict(:a0 => :noise))
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        @test model isa PEtabModel

        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                     "c1" => Dict(:k1 => :noise))
    end

    @testset "Measurement data format" begin
        # Single experimental condition
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                     "c1" => Dict(:a0 => 0.9))
        # PEtab-parameter to "estimate"
        parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin),
                            PEtabParameter(:noise, value=0.6, scale=:lin),
                            PEtabParameter(:scale_P, value=1.0, scale=:lin),
                            PEtabParameter(:offset_P, value=1.0, scale=:lin)]

        # Observable equation
        @unpack A, B = rn
        @parameters noiseParameter1_obs_A observableParameter1_obs_B observableParameter2_obs_B
        observables = Dict(["obs_a" => PEtabObservable(A, :lin, noiseParameter1_obs_A),
                            "obs_b" => PEtabObservable(observableParameter1_obs_B + observableParameter2_obs_B*B, :lin, 1.0)])
        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                 obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                 time=[0, 10.0, 0, 10.0],
                                 measurement=[0.7, 0.1, 0.8, 0.2],
                                 observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                 noise_parameters=[1.0, "noise", missing, missing])
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        @test model isa PEtabModel

        path = joinpath(@__DIR__, "Tmp.csv")
        CSV.write(path, measurements)
        dataread = CSV.read(path, DataFrame)
        rm(path)
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        @test model isa PEtabModel

        # Start messing up the data
        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, "tada"],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFileError begin
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        end

        measurements = DataFrame(simulation_id=[1.0, "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFileError begin
            model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                               simulation_conditions = simulation_conditions)
        end

        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise1", missing, missing])
        @test_throws PEtab.PEtabFileError begin
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        end

        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P1", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFileError begin
        model = PEtabModel(rn, observables, measurements, parameters; speciemap=speciemap,
                           simulation_conditions = simulation_conditions)
        end
    end
end
