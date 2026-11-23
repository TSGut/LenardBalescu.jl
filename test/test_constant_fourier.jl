@testset "Constant Fourier interaction naming" begin
    @test LenardBalescu.is_constant_fourier(:constant_fourier)
    @test_throws ArgumentError LenardBalescu.is_constant_fourier(:maxwell)
    @test !LenardBalescu.is_constant_fourier(:coulomb)
    @test_throws ArgumentError LenardBalescu.is_constant_fourier(:unknown)

    for precompute in (LenardBalescu.precompute_k_space, LenardBalescu.precompute_k_space_3d)
        current = precompute(4, 5.0, 0.02, 3; potential=:constant_fourier)
        coulomb = precompute(4, 5.0, 0.02, 3; potential=:coulomb)
        @test current != coulomb
        @test_throws ArgumentError precompute(4, 5.0, 0.02, 3; potential=:maxwell)
    end

    for dim in (2, 3), batchsize in (0, 5), static_dielectric in (false, true)
        par = SetParam()
        par.dim = dim
        par.Nv = 4
        par.N = 16
        par.ntot = 1
        par.T = par.dt
        par.batchsize = batchsize
        par.k_perp_order = 4
        par.k_perp_max = 5.0
        par.nang = 8
        update_kernel_params!(par)

        mktempdir() do outdir
            function run(potential)
                par.potential = potential
                run_simulation(; par, use_lenard_balescu=true,
                    static_dielectric, radial_symmetry=true,
                    rng=MersenneTwister(17), verbose=0, savecsv=false, outdir,
                    init_particles=(N, dV, Vgrid, p, rng) -> initialize_particles_maxwellian(N, p, rng))
            end
            current = run(:constant_fourier)
            @test all(isfinite, current.vx)
            @test all(isfinite, current.vy)
            @test all(isfinite, current.L2)
            dim == 3 && (@test all(isfinite, current.vz))
        end
    end
end
