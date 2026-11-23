using Test

using LenardBalescu
using Random

@testset "Radial dielectric half-line integral" begin
    radius = collect(range(-6.0, 6.0; length=1201))
    f_prime = @. -2.0 * radius * exp(-radius^2)

    _, _, _, I_2d = LenardBalescu.precompute_dielectric_data(
        radius, f_prime, 2, 16; radial_symmetry=true)
    _, _, _, I_3d = LenardBalescu.precompute_dielectric_data(
        radius, f_prime, 3, 16; radial_symmetry=true)

    # For f(r)=exp(-r²):
    # ∫₀∞ f'(r)dr = -1 and ∫₀∞ r f'(r)dr = -√π/2.
    @test I_2d ≈ -1.0 atol=2e-5
    @test I_3d ≈ -sqrt(π) / 2.0 atol=2e-7
end

function run_lb_compare_case(par::Params, init_particles::Function;
                             radial_seed::Int=42, generic_seed::Int=42)
    result_radial = run_simulation(; par=par,
        rng=MersenneTwister(radial_seed),
        use_lenard_balescu=true,
        outdir="",
        snapshot_interval=par.ntot,
        verbose=0,
        savecsv=false,
        trackconserved=false,
        radial_symmetry=true,
        init_particles=init_particles)

    result_generic = run_simulation(; par=par,
        rng=MersenneTwister(generic_seed),
        use_lenard_balescu=true,
        outdir="",
        snapshot_interval=par.ntot,
        verbose=0,
        savecsv=false,
        trackconserved=false,
        radial_symmetry=false,
        init_particles=init_particles)

    _, dV, _, Vgrid = build_velocity_grid(par)
    vstack_radial = par.dim == 2 ? hcat(result_radial.vx, result_radial.vy) : hcat(result_radial.vx, result_radial.vy, result_radial.vz)
    vstack_generic = par.dim == 2 ? hcat(result_generic.vx, result_generic.vy) : hcat(result_generic.vx, result_generic.vy, result_generic.vz)
    f_radial = f_recon(vstack_radial, result_radial.wg, Vgrid, par)
    f_generic = f_recon(vstack_generic, result_generic.wg, Vgrid, par)
    cq_radial = compute_conserved_quantities(vstack_radial, result_radial.wg, par)
    cq_generic = compute_conserved_quantities(vstack_generic, result_generic.wg, par)
    entropy_radial = LenardBalescu.compute_entropy(f_radial, dV; dim=par.dim)
    entropy_generic = LenardBalescu.compute_entropy(f_generic, dV; dim=par.dim)

    diff_l2 = sqrt(sum((f_radial .- f_generic) .^ 2) * dV)
    ref_l2 = max(sqrt(sum(f_radial .^ 2) * dV), 1e-14)
    rel_l2 = diff_l2 / ref_l2

    return result_radial, result_generic, f_radial, f_generic, rel_l2,
           cq_radial, cq_generic, entropy_radial, entropy_generic
end

@testset "Radial vs generic Lenard-Balescu consistency" begin
    @testset "2D isotropic BKW" begin
        par = SetParam()
        par.dim = 2
        par.Nv = 16
        par.Npc = 3
        par.N = par.Nv^par.dim * par.Npc
        par.dt = 0.01
        par.ntot = 12
        par.T = par.ntot * par.dt
        par.k_perp_order = 20
        par.k_perp_max = 20.0
        par.k_parallel_width = 0.05
        par.k_parallel_order = 3
        par.nang = 24
        update_kernel_params!(par)

        _, dV, _, Vgrid = build_velocity_grid(par)
        rng0 = MersenneTwister(7)
        vstack0, wg0 = initialize_particles_stratified_BKW(par.N, dV, Vgrid, par, rng0)
        init_particles = (N, dV, Vgrid, par, rng) -> (copy(vstack0), copy(wg0))

        result_radial, result_generic, f_radial, f_generic, rel_l2,
            cq_radial, cq_generic, entropy_radial, entropy_generic = run_lb_compare_case(par, init_particles)

        @test all(isfinite, f_radial)
        @test all(isfinite, f_generic)
        @test rel_l2 < 0.30

        @test isapprox(cq_radial.mass, cq_generic.mass; rtol=5e-3, atol=5e-6)
        @test isapprox(cq_radial.energy, cq_generic.energy; rtol=6e-2, atol=2e-4)
        @test isapprox(entropy_radial, entropy_generic; rtol=8e-2, atol=2e-4)

        pr = cq_radial.momentum
        pg = cq_generic.momentum
        @test isapprox(pr[1], pg[1]; rtol=8e-2, atol=2e-4)
        @test isapprox(pr[2], pg[2]; rtol=8e-2, atol=2e-4)
    end

    @testset "3D isotropic Maxwellian" begin
        par = SetParam()
        par.dim = 3
        par.Nv = 7
        par.Npc = 1
        par.N = par.Nv^par.dim * par.Npc
        par.dt = 0.01
        par.ntot = 3
        par.T = par.ntot * par.dt
        par.k_perp_order = 10
        par.k_perp_max = 12.0
        par.k_parallel_width = 0.05
        par.k_parallel_order = 3
        par.nang = 8
        par.dielectric_rho_points_3d = 8
        par.dielectric_psi_points_3d = 8
        update_kernel_params!(par)

        rng0 = MersenneTwister(11)
        vstack0, wg0 = initialize_particles_maxwellian(par.N, par, rng0)
        init_particles = (N, dV, Vgrid, par, rng) -> (copy(vstack0), copy(wg0))

        result_radial, result_generic, f_radial, f_generic, rel_l2,
            cq_radial, cq_generic, entropy_radial, entropy_generic = run_lb_compare_case(par, init_particles)

        @test all(isfinite, f_radial)
        @test all(isfinite, f_generic)
        @test rel_l2 < 0.30

        @test isapprox(cq_radial.mass, cq_generic.mass; rtol=1e-2, atol=5e-6)
        @test isapprox(cq_radial.energy, cq_generic.energy; rtol=8e-2, atol=2e-4)
        @test isapprox(entropy_radial, entropy_generic; rtol=1.2e-1, atol=2e-4)

        pr = cq_radial.momentum
        pg = cq_generic.momentum
        @test isapprox(pr[1], pg[1]; rtol=2e-1, atol=3e-4)
        @test isapprox(pr[2], pg[2]; rtol=2e-1, atol=3e-4)
        @test isapprox(pr[3], pg[3]; rtol=2e-1, atol=3e-4)
    end

    @testset "3D spherical shell" begin
        par = SetParam()
        par.dim = 3
        par.Nv = 7
        par.Npc = 1
        par.N = par.Nv^par.dim * par.Npc
        par.dt = 0.01
        par.ntot = 3
        par.T = par.ntot * par.dt
        par.k_perp_order = 10
        par.k_perp_max = 12.0
        par.k_parallel_width = 0.05
        par.k_parallel_order = 3
        par.nang = 8
        par.dielectric_rho_points_3d = 8
        par.dielectric_psi_points_3d = 8
        update_kernel_params!(par)

        rng0 = MersenneTwister(19)
        vstack0, wg0 = initialize_particles_spherical_shell_3d(par.N, par, rng0;
            shell_radius=2.0, thickness=0.3)
        init_particles = (N, dV, Vgrid, par, rng) -> (copy(vstack0), copy(wg0))

        result_radial, result_generic, f_radial, f_generic, rel_l2,
            cq_radial, cq_generic, entropy_radial, entropy_generic = run_lb_compare_case(par, init_particles)

        @test all(isfinite, f_radial)
        @test all(isfinite, f_generic)
        @test rel_l2 < 0.35

        @test isapprox(cq_radial.mass, cq_generic.mass; rtol=1e-2, atol=5e-6)
        @test isapprox(cq_radial.energy, cq_generic.energy; rtol=1e-1, atol=3e-4)
        @test isapprox(entropy_radial, entropy_generic; rtol=1.5e-1, atol=3e-4)

        pr = cq_radial.momentum
        pg = cq_generic.momentum
        @test isapprox(pr[1], pg[1]; rtol=2.5e-1, atol=4e-4)
        @test isapprox(pr[2], pg[2]; rtol=2.5e-1, atol=4e-4)
        @test isapprox(pr[3], pg[3]; rtol=2.5e-1, atol=4e-4)
    end
end
