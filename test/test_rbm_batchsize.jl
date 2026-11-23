using Test

using Random
using LenardBalescu

# Verifies partition_random_batches creates a balanced partition that
# covers all particles exactly once with no overlap.
@testset "RBM batch partitioning" begin
    rng = MersenneTwister(7)
    N = 23
    p = 5
    batches, rem = LenardBalescu.partition_random_batches(rng, N, p)

    # Balanced means batch sizes differ by at most one.
    sizes = length.(batches)
    @test maximum(sizes) - minimum(sizes) <= 1

    # Full coverage and disjointness: every index appears exactly once.
    all_ids = vcat(batches...)
    @test length(all_ids) == N
    @test length(unique(all_ids)) == N
    @test sort(all_ids) == collect(1:N)

    # The returned remainder should match the balanced redistribution formula.
    nb = cld(N, p)
    expected_rem = N - fld(N, nb) * nb
    @test rem == expected_rem
end

# Smoke test for batched RBM integration on the 2D Landau path.
# We mainly check that the simulation completes and returns finite diagnostics.
@testset "RBM batchsize smoke: 2D Landau" begin
    par = SetParam()
    par.dim = 2
    par.Nv = 5
    par.N = 25
    par.batchsize = 6
    par.dt = 0.01
    par.ntot = 2
    par.T = par.ntot * par.dt
    update_kernel_params!(par)

    result = run_simulation(
        par=par,
        rng=MersenneTwister(11),
        use_lenard_balescu=false,
        init_particles=(N, dV, Vgrid, par, rng) -> initialize_particles_maxwellian(N, par, rng),
        verbose=0,
        savecsv=false,
        outdir=mktempdir()
    )

    # Output sizes should remain consistent with particle count.
    @test length(result.vx) == par.N
    @test length(result.vy) == par.N
    # Error histories should be numerically well-defined.
    @test all(isfinite, result.L1)
    @test all(isfinite, result.L2)
end

# Smoke test for batched RBM integration on the 2D Lenard-Balescu path.
# Uses small k-space settings to keep runtime short while exercising the LB branch.
@testset "RBM batchsize smoke: 2D Lenard-Balescu" begin
    par = SetParam()
    par.dim = 2
    par.Nv = 4
    par.N = 16
    par.batchsize = 5
    par.dt = 0.01
    par.ntot = 1
    par.T = par.ntot * par.dt
    par.k_perp_order = 8
    par.k_perp_max = 10.0
    par.k_parallel_width = 0.05
    par.k_parallel_order = 3
    update_kernel_params!(par)

    result = run_simulation(
        par=par,
        rng=MersenneTwister(13),
        use_lenard_balescu=true,
        radial_symmetry=true,
        init_particles=(N, dV, Vgrid, par, rng) -> initialize_particles_maxwellian(N, par, rng),
        verbose=0,
        savecsv=false,
        outdir=mktempdir()
    )

    # Output sizes should remain consistent with particle count.
    @test length(result.vx) == par.N
    @test length(result.vy) == par.N
    # Error histories should be numerically well-defined.
    @test all(isfinite, result.L1)
    @test all(isfinite, result.L2)
end
