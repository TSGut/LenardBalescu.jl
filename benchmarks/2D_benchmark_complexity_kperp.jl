# Computational complexity test for Lenard-Balescu collision operator for varying k_perp_order which are the quadrature nodes
# Fixes N and varies k_perp_order, plotting timing results.

using LenardBalescu
using Random, Statistics
using Plots

function run_kperp_complexity(; k_perp_orders=140:20:300, N=1600, seed=1, trials=5, out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_kperp_complexity.png"))
    """
    Run computational complexity analysis for Lenard-Balescu operator as a function of k_perp_order.
    """
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 2
    par.N = N
    par.Nv = Int(round(sqrt(N)))
    par.Npc = Int(N / par.Nv^2)
    par.k_perp_max = 100.0
    par.k_parallel_width = 0.02
    par.k_parallel_order = 3
    update_kernel_params!(par)
    _, _, _, Vgrid = build_velocity_grid(par)
    dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=false)

    println("\n" * "="^60)
    println("Computing Lenard-Balescu collision complexity vs quadrature nodes...")
    println("="^60)
    times_lb = Float64[]
    for kpo in k_perp_orders
        trial_times = Float64[]
        par.k_perp_order = kpo
        update_kernel_params!(par)
        for trial in 1:trials
            vx = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vy = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vstack = hcat(vx, vy)
            wg = fill(1.0 / N, N)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                radial_symmetry=false,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len)
            push!(trial_times, elapsed[])
        end
        median_time = median(trial_times)
        push!(times_lb, median_time)
        println("  k_perp_order=$(kpo)  median time=$(median_time) s")
    end

    # Estimate scaling log-log slope (exclude smallest k_perp_order)
    logK = log.(float.(k_perp_orders[2:end]))
    logT = log.(times_lb[2:end])
    slope = cov(logK, logT) / var(logK)

    println("\n" * "="^60)
    println("Complexity Analysis Summary")
    println("="^60)
    for (kpo, t) in zip(k_perp_orders, times_lb)
        println("  k_perp_order=$(kpo)  time=$(t) s")
    end
    println("  Estimated slope ~ $(round(slope, digits=3)) (1 ~ linear, 2 ~ quadratic)")

    # Reference lines
    ref_k = k_perp_orders
    ref_k1 = times_lb[1] * (ref_k ./ ref_k[1])
    ref_k2 = times_lb[1] * (ref_k ./ ref_k[1]).^2

    p = plot(k_perp_orders, times_lb;
        xscale=:log10,
        yscale=:log10,
        marker=:square,
        markersize=6,
        label="Lenard-Balescu (slope ≈ $(round(slope, digits=2)))",
        xlabel="k_perp_order",
        ylabel="Collision time (s)",
        title="LB Complexity vs k_perp_order (N=$(N))",
        grid=false,
        titlefontsize=14,
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=9,
        legend = :topleft,
        linewidth=2,
        widen=true)
    plot!(p, ref_k, ref_k1; linestyle=:dot, linewidth=1.5, label="O(k)", color=:black, alpha=0.7)
    plot!(p, ref_k, ref_k2; linestyle=:dash, linewidth=1.5, label="O(k²)", color=:black, alpha=0.7)

    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved k_perp_order complexity plot to $out_png")

    return (k_perp_orders=k_perp_orders, times_lb=times_lb, slope=slope)
end

# Run the complexity comparison for k_perp_order
run_kperp_complexity();
