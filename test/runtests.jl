using Test

using LenardBalescu
using DelimitedFiles
using Random

@testset "C++ Landau code output sanity checks" begin
    data_dir = abspath(joinpath(@__DIR__, "..", "data", "testdata"))
    if !isdir(data_dir)
        @info "Test data directory not found; skipping CSV comparison tests." data_dir
        return
    end

    errors_path = joinpath(data_dir, "errors.csv")
    @testset "errors.csv exists" begin
        @test isfile(errors_path)
    end

    errors = readdlm(errors_path, ',', Float64; skipstart=1)
    tcol = errors[:, 1]
    L1col = errors[:, 2]
    L2col = errors[:, 3]

    sample_path = joinpath(data_dir, "f_recon_t0.csv")
    @testset "grid size from f_recon_t0.csv" begin
        @test isfile(sample_path)
        f_sample = read_vector_csv(sample_path)
        m = length(f_sample)
        Nv = Int(round(sqrt(m)))
        @test Nv * Nv == m
        global Nv = Nv
    end

    par = SetParam()
    # Archived dataset parameters, independent of current simulation defaults.
    par.dim = 2
    par.Lv = 4.0
    par.temp = 1.0
    par.c = 0.5
    par.beta = 1.0 / 16.0
    par.Nv = Nv
    par.N = par.Nv * par.Nv * par.Npc
    par.dt = length(tcol) > 1 ? (tcol[2] - tcol[1]) : par.dt
    par.T = tcol[end]
    par.ntot = length(tcol) - 1
    update_kernel_params!(par)

    _, dV, _, Vgrid = build_velocity_grid(par)

    # The archived reconstruction snapshots were saved every three steps.
    steps = 0:3:par.ntot
    for step in steps
        @testset "step $(step)" begin
            t = step * par.dt
            fpath = joinpath(data_dir, "f_recon_t" * string(step) * ".csv")
            @testset "file exists" begin
                @test isfile(fpath)
            end

            f_recon_vec = read_vector_csv(fpath)
            f_analytic = BKW_vector(t, par.temp, par.c, par.beta, Vgrid)

            L1 = L1_error(f_recon_vec, f_analytic, dV)
            L2 = L2_error(f_recon_vec, f_analytic, dV)

            idx = step + 1
            @testset "time column" begin
                @test isapprox(tcol[idx], t; atol=1e-12)
            end
            @testset "L1 error" begin
                @test isapprox(L1, L1col[idx]; rtol=5e-6, atol=5e-8)
            end
            @testset "L2 error" begin
                @test isapprox(L2, L2col[idx]; rtol=5e-6, atol=5e-8)
            end
        end
    end
end;

@testset "BKW-Landau simulation accuracy" begin
    par = SetParam()
    par.Nv = 20  # Coarse grid for test speed
    par.Npc = 2
    par.N = par.Nv * par.Nv * par.Npc
    par.dt = 0.01
    par.ntot = 50  # Short test
    par.T = par.ntot * par.dt
    update_kernel_params!(par)
    
    rng = MersenneTwister(42)
    
    result = run_simulation(; par=par, rng=rng, use_lenard_balescu=false,
                            outdir=mktempdir(), snapshot_interval=10, verbose=0)
    
    _, dV, _, Vgrid = build_velocity_grid(par)
    
    vstack = hcat(result.vx, result.vy)
    f_reconstructed = f_recon(vstack, result.wg, Vgrid, par)
    
    f_analytic = BKW_vector(par.T, par.temp, par.c, par.beta, Vgrid)
    
    L2_final = L2_error(f_reconstructed, f_analytic, dV)
    
    @test L2_final < 0.05
end;

@testset "Stratified Maxwellian initialization" begin
    for dim in (2, 3), Lv in (1.5, 6.0)
        par = SetParam()
        par.dim = dim
        par.Lv = Lv
        par.Nv = 24
        par.temp = 1.7
        update_kernel_params!(par)
        _, dV, _, Vgrid = build_velocity_grid(par)
        N = 100_000
        v, w = initialize_particles_stratified_maxwellian(N, dV, Vgrid, par, MersenneTwister(42))
        grids = LenardBalescu.component_grid_views(Vgrid, par)
        r2 = sum(g .^ 2 for g in grids)
        masses = exp.(-r2 ./ (2 * par.temp)) .* dV^dim ./ (2 * pi * par.temp)^(dim / 2)
        @test size(v) == (N, dim)
        @test sum(w) ≈ sum(masses)
        counts = zeros(Int, ntuple(_ -> par.Nv, dim))
        for i in 1:N
            cell = ntuple(d -> floor(Int, (v[i, d] + Lv) / dV) + 1, dim)
            counts[cell...] += 1
        end
        @test maximum(abs.(counts .- N .* masses ./ sum(masses))) <= 1
        if Lv == 6.0
            @test isapprox(sum(w), 1.0; atol=2e-5)
            @test isapprox(sum(abs2, v) / (N * dim), par.temp; atol=0.08)
        end
    end
end

include("test_lb_radial_vs_generic.jl")
include("test_radial_grid.jl")
include("test_rbm_batchsize.jl")
include("test_constant_fourier.jl")
