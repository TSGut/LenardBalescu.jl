# 3D computational complexity benchmark for Lenard-Balescu vs k_perp_order.

using LenardBalescu
using Random, Statistics
using Plots

function run_kperp_complexity_3d(; k_perp_orders=24:8:72,
                                 N=500,
                                 seed=1,
                                 trials=3,
                                 radial_symmetry=false,
                                 out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_3d_kperp_complexity.png"))
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 3
    par.N = N
    par.Nv = 10
    par.Npc = 1
    par.k_perp_max = 20.0
    par.k_parallel_width = 0.05
    par.k_parallel_order = 3
    update_kernel_params!(par)

    _, _, _, Vgrid = build_velocity_grid(par)
    dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=radial_symmetry)

    println("\n" * "="^60)
    println("Computing 3D Lenard-Balescu complexity vs k_perp_order...")
    println("="^60)

    times_lb = Float64[]
    for kpo in k_perp_orders
        trial_times = Float64[]
        par.k_perp_order = kpo
        update_kernel_params!(par)

        for _ in 1:trials
            vstack = rand(rng, N, 3) .* (2 * par.Lv) .- par.Lv
            wg = fill(1.0 / N, N)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                radial_symmetry=radial_symmetry,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len)
            push!(trial_times, elapsed[])
        end

        tmed = median(trial_times)
        push!(times_lb, tmed)
        println("  k_perp_order=$(kpo)  median time=$(tmed) s")
    end

    logK = log.(float.(collect(k_perp_orders)[2:end]))
    logT = log.(times_lb[2:end])
    slope = cov(logK, logT) / var(logK)

    ref_k = collect(k_perp_orders)
    ref_k1 = times_lb[1] * (ref_k ./ ref_k[1])
    ref_k2 = times_lb[1] * (ref_k ./ ref_k[1]).^2

    p = plot(ref_k, times_lb;
        xscale=:log10, yscale=:log10,
        marker=:square, markersize=6,
        label="3D Lenard-Balescu (slope ≈ $(round(slope, digits=2)))",
        xlabel="k_perp_order", ylabel="Collision time (s)",
        title="3D LB Complexity vs k_perp_order (N=$N)",
        linewidth=2, grid=false, legend=:topleft, widen=true)

    plot!(p, ref_k, ref_k1; linestyle=:dot, linewidth=1.5, label="O(k)", color=:black, alpha=0.7)
    plot!(p, ref_k, ref_k2; linestyle=:dash, linewidth=1.5, label="O(k²)", color=:black, alpha=0.7)

    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved 3D k_perp complexity plot to $out_png")

    return (k_perp_orders=ref_k, times_lb=times_lb, slope=slope)
end

run_kperp_complexity_3d();
