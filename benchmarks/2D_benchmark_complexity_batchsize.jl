# 2D batchsize complexity benchmark for RBM batching.
# Measures runtime as batchsize changes (including batchsize=0 for full interaction)
# for both Landau and Lenard-Balescu simulation paths.

using LenardBalescu
using Random, Statistics
using Plots

function benchmark_batchsize_2d(; batchsize_list=nothing,
                                batchsize_step=4,
                                seed=1,
                                trials=3,
                                ntot=4,
                                Nv=20,
                                Npc=2,
                                dt=0.01,
                                radial_symmetry=false,
                                out_png=joinpath(@__DIR__, "output", "benchmark", "benchmark_2d_batchsize_complexity.png"))
    rng = MersenneTwister(seed)

    par_base = SetParam()
    par_base.dim = 2
    par_base.Nv = Nv
    par_base.Npc = Npc
    par_base.N = par_base.Nv^par_base.dim * par_base.Npc
    par_base.dt = dt
    par_base.ntot = ntot
    par_base.T = par_base.ntot * par_base.dt
    update_kernel_params!(par_base)

    N = par_base.N

    if isnothing(batchsize_list)
        @assert batchsize_step > 0 "batchsize_step must be > 0"
        batchsize_list = collect(0:batchsize_step:N)
        if batchsize_list[end] != N
            push!(batchsize_list, N)
        end
    else
        batchsize_list = sort(unique(Int.(batchsize_list)))
        batchsize_list = [p for p in batchsize_list if 0 <= p <= N]
        @assert !isempty(batchsize_list) "batchsize_list has no values in [0, N]"
        if 0 ∉ batchsize_list
            pushfirst!(batchsize_list, 0)
        end
        if N ∉ batchsize_list
            push!(batchsize_list, N)
        end
    end

    println("\n" * "="^56)
    println("2D batchsize complexity benchmark")
    println("="^56)
    println("Particles: N=$N, steps=$ntot, trials=$trials")
    println("Sweep: batchsize from 0 to N in step=$batchsize_step")
    println("batchsize=0 means full interaction (no RBM splitting)")

    # Requested batch size for plotting, with full-interaction cases mapped to N.
    # In the solver, batchsize<=1 or >=N behaves as full interaction.
    effective_p = [((p >= 2) && (p < N)) ? p : N for p in batchsize_list]

    # Landau timings
    println("\n" * "-"^56)
    println("Benchmarking Landau path vs batchsize...")
    println("-"^56)
    times_landau = Float64[]
    for p in batchsize_list
        trial_means = Float64[]
        for _ in 1:trials
            par = deepcopy(par_base)
            par.batchsize = p

            result = run_simulation(
                par=par,
                rng=rng,
                use_lenard_balescu=false,
                init_particles=(N, dV, Vgrid, par, rng) -> initialize_particles_maxwellian(N, par, rng),
                verbose=0,
                savecsv=false,
                outdir=mktempdir()
            )
            push!(trial_means, mean(result.collision_times))
        end
        tmed = median(trial_means)
        push!(times_landau, tmed)
        println("  requested batchsize=$p: median collision time/step = $tmed s")
    end

    # Lenard-Balescu timings
    println("\n" * "-"^56)
    println("Benchmarking Lenard-Balescu path vs batchsize...")
    println("-"^56)
    times_lb = Float64[]
    for p in batchsize_list
        trial_means = Float64[]
        for _ in 1:trials
            par = deepcopy(par_base)
            par.batchsize = p

            result = run_simulation(
                par=par,
                rng=rng,
                use_lenard_balescu=true,
                radial_symmetry=radial_symmetry,
                init_particles=(N, dV, Vgrid, par, rng) -> initialize_particles_maxwellian(N, par, rng),
                verbose=0,
                savecsv=false,
                outdir=mktempdir()
            )
            push!(trial_means, mean(result.collision_times))
        end
        tmed = median(trial_means)
        push!(times_lb, tmed)
        println("  requested batchsize=$p: median collision time/step = $tmed s")
    end

    # Speedups relative to full interaction baseline (batchsize=0 entry)
    base_idx = findfirst(==(0), batchsize_list)
    baseline_landau = times_landau[base_idx]
    baseline_lb = times_lb[base_idx]
    speedup_landau = baseline_landau ./ times_landau
    speedup_lb = baseline_lb ./ times_lb

    println("\n" * "="^56)
    println("Speedups relative to batchsize=0")
    println("="^56)
    for i in eachindex(batchsize_list)
        println("  batchsize=$(batchsize_list[i]): Landau x$(round(speedup_landau[i], digits=3)), LB x$(round(speedup_lb[i], digits=3))")
    end

    # Plot runtime vs requested batch size (log-log).
    ref_p = float.(effective_p)
    ref_linear = times_landau[1] .* (ref_p ./ ref_p[1])

    p = scatter(ref_p, times_landau;
        xscale=:log10,
        yscale=:log10,
        marker=:circle,
        markersize=6,
        label="Landau",
        xlabel="Requested batch size p (full interaction shown at N)",
        ylabel="Median collision time per step (s)",
        title="2D Runtime vs Batchsize (N=$N)",
        linewidth=2,
        grid=false,
        widen=true)

    scatter!(p, ref_p, times_lb;
        marker=:square,
        markersize=6,
        label="Lenard-Balescu",
        linewidth=2)

    mkpath(dirname(out_png))
    savefig(p, out_png)
    println("Saved batchsize complexity plot to $out_png")

    return (
        N=N,
        batchsize_list=batchsize_list,
        effective_batchsize=effective_p,
        times_landau=times_landau,
        times_lb=times_lb,
        speedup_landau=speedup_landau,
        speedup_lb=speedup_lb
    )
end

benchmark_batchsize_2d(batchsize_step=50);
