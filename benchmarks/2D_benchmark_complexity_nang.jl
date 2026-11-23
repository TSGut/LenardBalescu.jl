# Computational complexity test for Lenard-Balescu collision operator vs nang in 2D.
# Fixes N and varies angular quadrature order (nang) for non-radial dielectric preprocessing.

using LenardBalescu
using Random, Statistics
using Plots

function run_nang_complexity_2d(; nang_values=8:20:160,
                                N=1600,
                                seed=1,
                                trials=5,
                                out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_2d_nang_complexity.png"))
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 2
    par.N = N
    par.Nv = Int(round(sqrt(N)))
    par.Npc = Int(max(1, N ÷ max(par.Nv^2, 1)))

    # Keep k-space parameters fixed so nang scaling is isolated.
    par.k_perp_order = 120
    par.k_perp_max = 100.0
    par.k_parallel_width = 0.02
    par.k_parallel_order = 3
    update_kernel_params!(par)

    _, _, _, Vgrid = build_velocity_grid(par)
    dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=false)

    println("\n" * "="^60)
    println("Computing 2D Lenard-Balescu complexity vs nang (fixed N=$N)...")
    println("="^60)

    times_lb = Float64[]
    nang_list = collect(nang_values)
    for nang in nang_list
        trial_times = Float64[]
        par.nang = nang
        update_kernel_params!(par)

        for _ in 1:trials
            vx = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vy = rand(rng, N) .* 2 .* par.Lv .- par.Lv
            vstack = hcat(vx, vy)
            wg = fill(1.0 / N, N)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                radial_symmetry=false,
                verbose=0,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len)
            push!(trial_times, elapsed[])
        end

        tmed = median(trial_times)
        push!(times_lb, tmed)
        println("  nang=$(nang)  median time=$(tmed) s")
    end

    logNang = log.(float.(nang_list[2:end]))
    logT = log.(times_lb[2:end])
    slope = cov(logNang, logT) / var(logNang)

    println("\n" * "="^60)
    println("2D nang Complexity Summary")
    println("="^60)
    for (nang, t) in zip(nang_list, times_lb)
        println("  nang=$(nang)  time=$(t) s")
    end
    println("  Estimated slope ~ $(round(slope, digits=3))")

    ref_nang = nang_list
    ref_n1 = times_lb[1] * (ref_nang ./ ref_nang[1])
    ref_n2 = times_lb[1] * (ref_nang ./ ref_nang[1]).^2

    p = plot(nang_list, times_lb;
        xscale=:log10,
        yscale=:log10,
        marker=:square,
        markersize=6,
        label="2D LB non-radial (slope ≈ $(round(slope, digits=2)))",
        xlabel="nang",
        ylabel="Collision time (s)",
        title="2D LB Complexity vs nang (fixed N=$N)",
        grid=false,
        legend=:topleft,
        linewidth=2,
        widen=true)
    plot!(p, ref_nang, ref_n1; linestyle=:dot, linewidth=1.5, label="O(nang)", color=:black, alpha=0.7)
    plot!(p, ref_nang, ref_n2; linestyle=:dash, linewidth=1.5, label="O(nang²)", color=:black, alpha=0.7)

    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved 2D nang complexity plot to $out_png")

    return (nang_values=nang_list, times_lb=times_lb, slope=slope)
end

run_nang_complexity_2d();
