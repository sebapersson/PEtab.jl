#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#



    # Define reaction network model
    rn = @reaction_network begin
        @parameters a0
        @species A(t)=a0
        (k1, k2), A <--> B
    end
    state_map = [:B => 1.0] # Constant initial value for B

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


    @testset "Experimental conditions input" begin

        # PEtab-parameter to "estimate"
        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin)]

        # Case 1 everything should work
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                    "c1" => Dict(:a0 => 0.9))
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel


        simulation_conditions = Dict("c0" => Dict(:b0 => 0.8),
                                    "c1" => Dict(:b0 => 0.9))
        @test_throws PEtab.PEtabFormatError begin
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end


        # PEtab-parameter to "estimate"
        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin), 
                            PEtabParameter(:noise, value=0.6, scale=:lin)]
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                    "c1" => Dict(:a0 => :noise))
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel


        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                    "c1" => Dict(:a0 => :noise1))
        @test_throws PEtab.PEtabFormatError begin
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end


        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                    "c1" => Dict(:k1 => :noise))
        @test_throws PEtab.PEtabFormatError begin
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end
    end


    @testset "PEtabParameters conditions input" begin
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                    "c1" => Dict(:a0 => :noise))

        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin),
                            PEtabParameter(:noise, value=0.6, scale=:lin)]
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                        petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel


        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin),
                            PEtabParameter(:noise, value=0.6, scale=:lin),
                            PEtabParameter(:k3, value=0.6, scale=:lin)]
        @test_throws PEtab.PEtabFormatError begin
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                        petab_parameters, verbose=false, state_map=state_map)
        end


        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                            PEtabParameter(:k2, value=0.6, scale=:lin),
                            PEtabParameter(:noise, value=0.6, scale=:lin),
                            PEtabParameter(:k3, value=0.6, scale=:lin, estimate=false)]
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                        petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel
    end


    @testset "Measurement data format" begin

        # Single experimental condition
        simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                                     "c1" => Dict(:a0 => 0.9))

        # PEtab-parameter to "estimate"
        petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
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
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel

        path = joinpath(@__DIR__, "Tmp.csv")
        CSV.write(path, measurements)
        dataFromDisk = CSV.read(path, DataFrame)
        rm(path)
        petab_model = PEtabModel(rn, simulation_conditions, observables, dataFromDisk,
                                    petab_parameters, verbose=false, state_map=state_map)
        @test typeof(petab_model) <: PEtabModel

        # Start messing up the data
        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, "tada"],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFormatError begin                         
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end                             

        measurements = DataFrame(simulation_id=[1.0, "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFormatError begin                         
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end                             

        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_c"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFormatError begin                         
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)
        end                             

        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P", "1.0;1.0"],
                                noise_parameters=[1.0, "noise1", missing, missing])
        @test_throws PEtab.PEtabFormatError begin                         
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map)                             
        end                             

        measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                                obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                                time=[0, 10.0, 0, 10.0],
                                measurement=[0.7, 0.1, 0.8, 0.3],
                                observable_parameters=[missing, "", "scale_P;offset_P1", "1.0;1.0"],
                                noise_parameters=[1.0, "noise", missing, missing])
        @test_throws PEtab.PEtabFormatError begin                         
        petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                                    petab_parameters, verbose=false, state_map=state_map) 
        end                                   
    end
