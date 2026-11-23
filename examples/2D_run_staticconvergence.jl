# Convergence of static Lenard-Balescu to Landau as α → 0.
#
# The static dielectric ε_static(k) = 1 + α V̂(k) (Coulomb: V̂(k) = 1/|k|²)
# replaces the dynamic dielectric in the Lenard-Balescu kernel.
# As α → 0 the static-LB kernel should converge to the Landau kernel with β = β₀ = β_eff_num(0) (the effective β of the static-LB kernel at α=0).

using LenardBalescu
using Random, Statistics, LinearAlgebra
using Plots, LaTeXStrings

function run_static_convergence(;
    seed              = 1,
    Nv                = 40,
    Npc               = 3,
    ntot              = 100,
    dt                = 0.01,
    k_perp_order      = 50,
    k_perp_max        = 100.0,
    k_parallel_width  = 0.02,
    k_parallel_order  = 3,
    radial_symmetry   = false,
    alpha_values      = [0.0, 1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0],
    out_dir           = joinpath(@__DIR__, "output", "static_convergence"),
    verbose           = 1)

    mkpath(out_dir)

    # ----------------------------------------------------------------
    # Parameter setup
    # ----------------------------------------------------------------
    function make_par(; nv=Nv, alpha_val=1.0, beta_val=NaN)
        par = SetParam()
        par.dim              = 2
        par.Nv               = nv
        par.Npc              = Npc
        par.N                = nv^2 * Npc
        par.T                = ntot * dt
        par.dt               = dt
        par.ntot             = ntot
        par.batchsize        = 0
        par.k_perp_order     = k_perp_order
        par.k_perp_max       = k_perp_max
        par.k_parallel_width = k_parallel_width
        par.k_parallel_order = k_parallel_order
        # γ = -3: Coulomb interaction in 2D.  The LB kernel uses V̂(k) = 1/|k|², and
        # after integrating out the δ(k‖) constraint the effective velocity-space
        # dependence is ∝ 1/|q|.  The Landau tensor β|q|^γ(|q|²I - q⊗q) matches
        # this only when γ = -3.  The default γ = 0 (Maxwell molecules) is a
        # different interaction and must NOT be used here.
        par.gam              = -3.0
        par.alpha            = alpha_val
        # Override β when specified (used by matched Landau references).
        if !isnan(beta_val)
            par.beta = beta_val
        end
        update_kernel_params!(par)
        return par
    end

    par0 = make_par()
    _, dV0, _, Vgrid0 = build_velocity_grid(par0)

    # Precompute k-space quadrature once (same grid used by every static-LB run).
    # beta_eff_num(alpha) is the Landau beta that makes the Landau kernel identical
    # to the static-LB kernel for this quadrature.  It must be computed numerically
    # because the α=0 integral ∫ dk/k² is IR-divergent: beta_eff_num(0) depends
    # on the minimum k node of the quadrature and does not converge to a fixed value.
    _kp, _kperp, _nk, _approx, _p11, _p22, _p12, _kang =
        LenardBalescu.precompute_k_space(k_perp_order, k_perp_max,
                                         k_parallel_width, k_parallel_order)
    # beta_eff_num(alpha): the Landau β that the static-LB kernel evaluates to at this α,
    # computed from the same numerical quadrature.  At α=0 this is the exact Landau limit.
    function beta_eff_num(alpha)
        V_sq = 1.0 ./ (_nk.^2 .+ 1e-15)
        eps_sq = max.((1.0 .+ alpha .* V_sq).^2, 1e-10)
        K_yy_raw = sum(_p22 ./ eps_sq .* _approx)
        return par0.beta * K_yy_raw
    end
    # Fixed Landau reference: calibrated at α=0 (the unscreened Coulomb limit).
    # All α-sweep Landau runs use this SAME β₀, so the 'error' measures the effect
    # of the screening.
    beta_ref = beta_eff_num(0.0)
    if verbose >= 1
        println("\nFixed Landau reference: β₀ = $(round(beta_ref, digits=6))  (from β_eff_num(0))")
        println(rpad("alpha",8), rpad("beta_eff_num",14), rpad("rel_change",12))
        for a in collect(alpha_values)
            be = beta_eff_num(a)
            println(rpad(round(a,digits=3),8),
                    rpad(round(be,digits=6),14),
                    rpad(round((be-beta_ref)/beta_ref*100, digits=2), 8), "%")
        end
        println()
    end

    alpha_list = collect(alpha_values)
    n_alpha    = length(alpha_list)

    # Snapshot steps (fixed — ntot is the same for all runs)
    snap_interval = max(1, ntot ÷ 10)
    snap_steps    = collect(snap_interval:snap_interval:ntot)
    n_snaps       = length(snap_steps)
    snap_times    = snap_steps .* dt

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    function rel_l2_error(f, g_ref, dV)
        numer = sqrt(sum((f .- g_ref).^2) * dV^2)
        denom = sqrt(sum(g_ref.^2) * dV^2)
        return denom > 0 ? numer / denom : 0.0
    end

    function kl_divergence(f, g_ref, dV)
        mask = (f .> 0.0) .& (g_ref .> 0.0)
        any(mask) || return 0.0
        return sum(f[mask] .* log.(f[mask] ./ g_ref[mask])) * dV^2
    end

    # Format an α value as a LaTeX scientific-notation string, e.g. 10^{-5}.
    function fmt_alpha(alpha)
        iszero(alpha) && return L"\alpha = 0"
        e = floor(Int, log10(abs(alpha)))
        m = alpha / 10.0^e
        isapprox(m, 1.0; atol=1e-10) && return latexstring("\\alpha = 10^{$e}")
        return latexstring("\\alpha = $(round(m, digits=2)) \\cdot 10^{$e}")
    end

    # Pre-generate an identical initial particle distribution for a given par.
    # Returns a closure for the init_particles kwarg of run_simulation.
    # Both Landau and static-LB runs receive the same closure so they start
    # from identical particle distributions.
    function make_init_fn(par)
        rng_init = MersenneTwister(seed)
        _, dV_init, _, Vgrid_init = build_velocity_grid(par)
        Npart_init = par.Nv^2 * par.Npc
        vstack0, wg0 = initialize_particles_stratified_BKW(
            Npart_init, dV_init, Vgrid_init, par, rng_init)
        return (N, dV, Vg, p, rng_) -> (copy(vstack0), copy(wg0))
    end

    # Run one simulation and collect f_grid snapshots at snap_steps.
    # par_run.alpha must be set by the caller before passing.
    function collect_snapshots(par_run, use_lb, static_lb; init_fn=nothing)
        rng = MersenneTwister(seed)
        snaps = Dict{Int, Vector{Float64}}()
        vx_final = Float64[]
        vy_final = Float64[]
        function cb(k, vx, vy, wg, p, Vg, Vcc)
            if k in snap_steps
                vst = hcat(vx, vy)
                snaps[k] = f_recon(vst, wg, Vg, p)
            end
            if k == par_run.ntot
                append!(vx_final, vx)
                append!(vy_final, vy)
            end
        end
        run_simulation(;
            par                = par_run,
            use_lenard_balescu = use_lb,
            static_dielectric  = static_lb,
            radial_symmetry    = radial_symmetry,
            rng                = rng,
            step_callback      = cb,
            verbose            = verbose,
            savecsv            = false,
            trackconserved     = false,
            enforce_energy     = true,
            outdir             = mktempdir(),
            init_particles     = init_fn,
        )
        return snaps, vx_final, vy_final
    end

    # Pre-generate particles once for the default Nv (shared by all α-sweep runs)
    init_fn_default = make_init_fn(par0)

    # ----------------------------------------------------------------
    # Sweep over α: static-LB(α) vs FIXED Landau reference (β = β₀ = β_eff_num(0)).
    # Error measures the effect of screening; should decrease as α → 0.
    # ----------------------------------------------------------------
    l2_traj = zeros(n_alpha, n_snaps)
    kl_traj = zeros(n_alpha, n_snaps)

    # Landau reference is run once with β = β₀ (independent of α)
    par_landau_ref = make_par(beta_val=beta_ref)
    landau_snaps_ref, vx_landau_final, vy_landau_final = collect_snapshots(par_landau_ref, false, false; init_fn=init_fn_default)

    vx_static0_final = Float64[]
    vy_static0_final = Float64[]
    for (i, alpha) in enumerate(alpha_list)
        if verbose >= 1
            println("\n=== α = $(round(alpha, digits=4))  ($i/$n_alpha) ===")
        end
        # Single fixed Landau reference (β₀) reused across all α
        landau_snaps_i = landau_snaps_ref
        # Static-LB run
        par_i = make_par(alpha_val=alpha)
        static_snaps, vx_s, vy_s = collect_snapshots(par_i, true, true; init_fn=init_fn_default)
        if iszero(alpha)
            vx_static0_final = vx_s
            vy_static0_final = vy_s
        end
        for (j, step) in enumerate(snap_steps)
            f_s = get(static_snaps, step, nothing)
            f_l = get(landau_snaps_i, step, nothing)
            if f_s !== nothing && f_l !== nothing
                l2_traj[i, j] = rel_l2_error(f_s, f_l, dV0)
                kl_traj[i, j] = kl_divergence(f_s, f_l, dV0)
            end
        end
    end

    l2_final = l2_traj[:, end]
    kl_final = kl_traj[:, end]
    # Shield log-scale entropy plots from exact zeros (notably α=0).
    kl_plot_floor = 1e-16
    kl_final_plot = max.(kl_final, kl_plot_floor)
    kl_traj_plot = max.(kl_traj, kl_plot_floor)

    # For the log x-axis on α plots, floor α=0 to a small positive value.
    alpha_plot_floor = 1e-16
    alpha_list_log = max.(alpha_list, alpha_plot_floor)

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    _plot_defaults = (
        guidefontsize  = 14,
        tickfontsize   = 12,
        legendfontsize = 11,
        titlefontsize  = 13,
        linewidth      = 2,
        framestyle     = :box,
        grid           = true,
        minorgrid      = false,
    )

    ylabel_l2 = "relative error"
    ylabel_kl = "relative entropy"
    T_final_str = "t = $(round(ntot*dt, digits=2))"

    colors_traj = palette(:viridis, n_alpha + 2)[2:end-1]

    # Explicit x-ticks for the α-axis: one tick per data point.
    alpha_xtick_labs = [iszero(a) ? L"0" : latexstring("10^{$(floor(Int, log10(a)))}") for a in alpha_list]

    # ---- Plot 1: L2 error vs α at final T ----
    p1 = plot(alpha_list_log, l2_final;
        xlabel       = L"\alpha",
        ylabel       = ylabel_l2,
        legend       = false,
        marker       = :circle,
        markersize   = 5,
        xscale       = :log10,
        yscale       = :log10,
        xticks       = (alpha_list_log, alpha_xtick_labs),
        xrotation    = 45,
        color        = :steelblue,
        _plot_defaults...)

    savefig(p1, joinpath(out_dir, "l2_vs_alpha.pdf"))
    savefig(p1, joinpath(out_dir, "l2_vs_alpha.png"))

    # ---- Plot 2: KL divergence vs α at final T ----
    p2 = plot(alpha_list_log, kl_final_plot;
        xlabel       = L"\alpha",
        ylabel       = ylabel_kl,
        legend       = false,
        marker       = :circle,
        markersize   = 5,
        xscale       = :log10,
        yscale       = :log10,
        xticks       = (alpha_list_log, alpha_xtick_labs),
        xrotation    = 45,
        color        = :darkorange,
        _plot_defaults...)

    savefig(p2, joinpath(out_dir, "kl_vs_alpha.pdf"))
    savefig(p2, joinpath(out_dir, "kl_vs_alpha.png"))

    # ---- Plots 3 & 4: error vs time for all α ----
    p3 = plot(;
        xlabel        = L"t",
        ylabel        = ylabel_l2,
        yscale        = :log10,
        legend        = :outerright,
        right_margin  = 30Plots.mm,
        _plot_defaults...)
    for (i, alpha) in enumerate(alpha_list)
        plot!(p3, snap_times, l2_traj[i, :];
            label      = fmt_alpha(alpha),
            color      = colors_traj[i],
            marker     = :circle,
            markersize = 4)
    end
    savefig(p3, joinpath(out_dir, "l2_vs_time.pdf"))
    savefig(p3, joinpath(out_dir, "l2_vs_time.png"))

    p4 = plot(;
        xlabel        = L"t",
        ylabel        = ylabel_kl,
        yscale        = :log10,
        legend        = :outerright,
        right_margin  = 30Plots.mm,
        _plot_defaults...)
    for (i, alpha) in enumerate(alpha_list)
        plot!(p4, snap_times, kl_traj_plot[i, :];
            label      = fmt_alpha(alpha),
            color      = colors_traj[i],
            marker     = :circle,
            markersize = 4)
    end
    savefig(p4, joinpath(out_dir, "kl_vs_time.pdf"))
    savefig(p4, joinpath(out_dir, "kl_vs_time.png"))

    # ---- Combined summary figure ----
    p_combined = plot(p1, p2, p3, p4;
        layout        = (2, 2),
        size          = (1600, 900),
        left_margin   = 8Plots.mm,
        bottom_margin = 8Plots.mm,
        dpi           = 300)

    savefig(p_combined, joinpath(out_dir, "static_convergence_summary.pdf"))
    savefig(p_combined, joinpath(out_dir, "static_convergence_summary.png"))

    # ---- Scatter plots: initial and final particle states ----
    # Initial state (from shared init closure)
    vstack_init, _ = init_fn_default(par0.Nv^2 * par0.Npc, dV0, Vgrid0, par0, nothing)
    p_scatter_init = scatter(vstack_init[:, 1], vstack_init[:, 2];
        xlabel = L"v_x", ylabel = L"v_y", title = "Particles: initial state",
        markersize = 2, markerstrokewidth = 0, legend = false, aspect_ratio = 1,
        _plot_defaults...)
    savefig(p_scatter_init, joinpath(out_dir, "scatter_initial.pdf"))
    savefig(p_scatter_init, joinpath(out_dir, "scatter_initial.png"))

    p_scatter_landau = scatter(vx_landau_final, vy_landau_final;
        xlabel = L"v_x", ylabel = L"v_y", title = "Particles: final state (Landau)",
        markersize = 2, markerstrokewidth = 0, legend = false, aspect_ratio = 1,
        _plot_defaults...)
    savefig(p_scatter_landau, joinpath(out_dir, "scatter_final_landau.pdf"))
    savefig(p_scatter_landau, joinpath(out_dir, "scatter_final_landau.png"))

    p_scatter_static = scatter(vx_static0_final, vy_static0_final;
        xlabel = L"v_x", ylabel = L"v_y", title = "Particles: final state (static LB, α=0)",
        markersize = 2, markerstrokewidth = 0, legend = false, aspect_ratio = 1,
        _plot_defaults...)
    savefig(p_scatter_static, joinpath(out_dir, "scatter_final_staticLB.pdf"))
    savefig(p_scatter_static, joinpath(out_dir, "scatter_final_staticLB.png"))

    if verbose >= 1
        println("\nPlots saved to: $out_dir")
        println("Final relative L² errors:  min=$(minimum(l2_final))  max=$(maximum(l2_final))")
        println("Final KL div:              min=$(minimum(kl_final))  max=$(maximum(kl_final))")
    end

    return (
        alpha_values = alpha_list,
        snap_times   = snap_times,
        l2_traj      = l2_traj,
        kl_traj      = kl_traj,
        l2_final     = l2_final,
        kl_final     = kl_final,
    )
end

run_static_convergence()
