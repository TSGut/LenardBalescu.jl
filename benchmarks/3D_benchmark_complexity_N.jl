# 3D computational complexity benchmark for Landau and Lenard-Balescu collision operators.
# Compares scaling of both operators vs number of particles N.

using LenardBalescu
using Random, Statistics
using Plots

function run_complexity_comparison_3d(; N_list=[80, 120, 180, 260, 360, 500, 700, 900], seed=1,
                                      trials=3,
                                      radial_symmetry=true,
                                      out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_3d_complexity_N.png"))
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 3
    par.Nv = 10
    par.Npc = 1
    par.N = par.Nv^par.dim * par.Npc
    update_kernel_params!(par)

    _, _, _, Vgrid = build_velocity_grid(par)
    dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=radial_symmetry)

    println("\n" * "="^44)
    println("Computing 3D Landau complexity vs N...")
    println("="^44)
    times_landau = Float64[]
    for N in N_list
        trial_times = Float64[]
        for _ in 1:trials
            vstack = rand(rng, N, 3) .* (2 * par.Lv) .- par.Lv
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
        tmed = median(trial_times)
        push!(times_landau, tmed)
        println("  N=$N  median time=$(tmed) s")
    end

    println("\n" * "="^44)
    println("Computing 3D Lenard-Balescu complexity vs N...")
    println("="^44)
    times_lb = Float64[]
    for N in N_list
        trial_times = Float64[]
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
        println("  N=$N  median time=$(tmed) s")
    end

    logN = log.(float.(N_list[2:end]))
    slope_landau = cov(logN, log.(times_landau[2:end])) / var(logN)
    slope_lb = cov(logN, log.(times_lb[2:end])) / var(logN)

    ref_n = N_list
    ref_n1 = times_landau[1] * (ref_n ./ ref_n[1])
    ref_n2 = times_landau[1] * (ref_n ./ ref_n[1]).^2

    p = plot(N_list, times_landau;
        xscale=:log10, yscale=:log10,
        marker=:circle, markersize=6,
        label="3D Landau (slope ≈ $(round(slope_landau, digits=2)))",
        xlabel="N (particles)", ylabel="Collision time (s)",
        title="3D Complexity vs N",
        linewidth=2, grid=false, legend=:topleft, widen=true)

    plot!(p, N_list, times_lb;
        marker=:square, markersize=6,
        label="3D Lenard-Balescu (slope ≈ $(round(slope_lb, digits=2)))",
        linewidth=2)

    plot!(p, ref_n, ref_n1; linestyle=:dot, linewidth=1.5, label="O(N)", color=:black, alpha=0.7)
    plot!(p, ref_n, ref_n2; linestyle=:dash, linewidth=1.5, label="O(N²)", color=:black, alpha=0.7)

    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved 3D complexity plot to $out_png")

    return (N_list=N_list, times_landau=times_landau, times_lb=times_lb,
            slope_landau=slope_landau, slope_lb=slope_lb)
end

run_complexity_comparison_3d();
