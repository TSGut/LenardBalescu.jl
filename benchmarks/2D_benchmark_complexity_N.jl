# Computational complexity test for Landau and Lenard-Balescu collision operators for varying N
# Compares both operators using independent uniform velocity samples on the same domain.

using LenardBalescu
using Random, Statistics
using Plots

function run_complexity_comparison(; N_list=[50, 100, 150, 200, 300, 400, 600, 800, 1000, 1400, 1600, 2000, 2400], seed=1,
                                    trials=5, out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_complexity.png"))
    """
    Run computational complexity analysis for both Landau and Lenard-Balescu operators,
    plotting both on the same log-log graph for direct comparison.
    """
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 2
    par.Nv = 25
    par.Npc = 2
    par.N = par.Nv * par.Nv * par.Npc
    update_kernel_params!(par)
    _, _, _, Vgrid = build_velocity_grid(par)
    dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=true)

    # Landau collision operator timings
    println("\n" * "="^40)
    println("Computing Landau collision complexity...")
    println("="^40)
    times_landau = Float64[]
    for N in N_list
        trial_times = Float64[]
        for trial in 1:trials
            vx = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vy = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vstack = hcat(vx, vy)
            wg = fill(1.0 / N, N)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=false,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len)
            push!(trial_times, elapsed[])
        end
        median_time = median(trial_times)
        push!(times_landau, median_time)
        println("  N=$N  median time=$(median_time) s")
    end

    # Lenard-Balescu collision operator timings
    println("\n" * "="^40)
    println("Computing Lenard-Balescu collision complexity...")
    println("="^40)
    times_lb = Float64[]
    for N in N_list
        trial_times = Float64[]
        for trial in 1:trials
            vx = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vy = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vstack = hcat(vx, vy)
            wg = fill(1.0 / N, N)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len)
            push!(trial_times, elapsed[])
        end
        median_time = median(trial_times)
        push!(times_lb, median_time)
        println("  N=$N  median time=$(median_time) s")
    end

    # Estimate scaling log-log slope for both operators (exclude smallest N to better capture asymptotic behavior)
    logN = log.(float.(N_list[2:end]))
    logT_landau = log.(times_landau[2:end])
    logT_lb = log.(times_lb[2:end])
    
    slope_landau = cov(logN, logT_landau) / var(logN)
    slope_lb = cov(logN, logT_lb) / var(logN)

    println("\n" * "="^40)
    println("Complexity Analysis Summary")
    println("="^40)
    println("Landau Collision:")
    for (N, t) in zip(N_list, times_landau)
        println("  N=$N  time=$t s")
    end
    println("  Estimated slope ~ $(round(slope_landau, digits=3)) (1 ~ linear, 2 ~ quadratic)")
    
    println("\nLenard-Balescu Collision:")
    for (N, t) in zip(N_list, times_lb)
        println("  N=$N  time=$t s")
    end
    println("  Estimated slope ~ $(round(slope_lb, digits=3)) (1 ~ linear, 2 ~ quadratic)")
    
    # Create combined plot
    ref_n = N_list
    ref_n1 = times_landau[1] * (ref_n ./ ref_n[1])
    ref_n2 = times_landau[1] * (ref_n ./ ref_n[1]).^2

    p = plot(N_list, times_landau;
        xscale=:log10,
        yscale=:log10,
        marker=:circle,
        markersize=6,
        label="Landau (slope ≈ $(round(slope_landau, digits=2)))",
        xlabel="N (particles)",
        ylabel="Collision time (s)",
        title="Computational Complexity",
        grid=false,
        titlefontsize=14,
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=9,
        legend = :topleft,
        linewidth=2,
        widen=true)
    
    plot!(p, N_list, times_lb;
        marker=:square,
        markersize=6,
        label="Lenard-Balescu (slope ≈ $(round(slope_lb, digits=2)))",
        linewidth=2)
    
    plot!(p, ref_n, ref_n1; linestyle=:dot, linewidth=1.5, label="O(N)", color=:black, alpha=0.7)
    plot!(p, ref_n, ref_n2; linestyle=:dash, linewidth=1.5, label="O(N²)", color=:black, alpha=0.7)
    
    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved complexity comparison plot to $out_png")
    
    return (N_list=N_list, times_landau=times_landau, times_lb=times_lb, 
            slope_landau=slope_landau, slope_lb=slope_lb)
end


# Run the complexity comparison
run_complexity_comparison();
