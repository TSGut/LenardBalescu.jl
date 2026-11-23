# Convergence study for the Landau operator on the 2D BKW problem.
#
# The BKW solution is the exact analytical solution to the homogeneous
# Boltzmann equation with Maxwell molecules (γ=0). It provides a clean
# reference for measuring numerical error as a function of resolution.
#
# Three sweeps are performed:
#   1. Nv sweep  — grid resolution (spline reconstruction accuracy)
#   2. Npc sweep — particles per cell (statistical noise)
#   3. dt sweep  — time-step size (time integration accuracy)
#
# Each sweep measures the relative L2 error:
#   ε = ||f_recon(T) - f_BKW(T)||_L2 / ||f_BKW(T)||_L2
# against the exact BKW analytical solution at the final time T.
#
# Run with:
#   julia --project examples/2D_convergence_bkw_landau.jl
#

using LenardBalescu
using Random
using Plots, LaTeXStrings

function run_bkw_landau_convergence(;
    seed             = 1,
    T_final          = 1.0,

    # Nv sweep: vary grid resolution, fix Npc and dt
    Nv_values        = [10, 15, 20, 30, 40],
    Npc_nv           = 4,
    dt_nv            = 0.05,

    # Npc sweep: vary particles per cell, fix Nv and dt
    Npc_values       = [1, 2, 4, 8, 16],
    Nv_npc           = 20,
    dt_npc           = 0.05,

    # dt sweep: vary time step, fix Nv and Npc
    dt_values        = [0.1, 0.05, 0.025, 0.01],
    Nv_dt            = 30,
    Npc_dt           = 4,

    out_dir          = joinpath(@__DIR__, "output", "convergence_bkw_landau"),
    verbose          = 1)

    mkpath(out_dir)

    # ----------------------------------------------------------------
    # Helper: build a Params struct for Landau/BKW runs
    # ----------------------------------------------------------------
    function make_par(; Nv, Npc, dt)
        par = SetParam()
        par.dim      = 2
        par.Nv       = Nv
        par.Npc      = Npc
        par.N        = Nv^2 * Npc
        par.dt       = dt
        par.T        = T_final
        par.ntot     = Int(ceil(T_final / dt))
        par.batchsize = 0
        # Default gam=0 (Maxwell molecules), beta=0.1, c=0.5, temp=1.0 —
        # these are the BKW parameters used by BKW_vector and
        # initialize_particles_stratified_BKW.
        update_kernel_params!(par)
        return par
    end

    # ----------------------------------------------------------------
    # Helper: run one Landau simulation and return the L2 error vs BKW
    # ----------------------------------------------------------------
    function run_one(par)
        rng = MersenneTwister(seed)
        _, dV, _, Vgrid = build_velocity_grid(par)

        # Exact BKW at the final time
        f_exact = BKW_vector(T_final, par.temp, par.c, par.beta, Vgrid)
        norm_exact = sqrt(sum(f_exact.^2) * dV^2)

        # Run Landau simulation from BKW initial condition
        result = run_simulation(;
            par                = par,
            use_lenard_balescu = false,
            rng                = rng,
            init_particles     = initialize_particles_stratified_BKW,
            verbose            = 0,
            savecsv            = false,
            trackconserved     = false,
            enforce_energy     = false,
            outdir             = mktempdir(),
        )

        # Reconstruct f at final time from returned particle state
        vstack_final = hcat(result.vx, result.vy)
        f_final = f_recon(vstack_final, result.wg, Vgrid, par)

        err = sqrt(sum((f_final .- f_exact).^2) * dV^2)
        return err / norm_exact
    end

    _defaults = (
        guidefontsize  = 14,
        tickfontsize   = 12,
        legendfontsize = 11,
        titlefontsize  = 13,
        linewidth      = 2,
        framestyle     = :box,
        grid           = true,
    )

    # ================================================================
    # Sweep 1: Nv (grid resolution)
    # ================================================================
    verbose >= 1 && println("\n=== Sweep 1: Nv (grid resolution) ===")
    err_nv = Float64[]
    for Nv in Nv_values
        par = make_par(Nv=Nv, Npc=Npc_nv, dt=dt_nv)
        e = run_one(par)
        verbose >= 1 && println("  Nv=$Nv  N=$(par.N)  ε=$e")
        push!(err_nv, e)
    end

    # ================================================================
    # Sweep 2: Npc (particles per cell)
    # ================================================================
    verbose >= 1 && println("\n=== Sweep 2: Npc (particles per cell) ===")
    err_npc = Float64[]
    for Npc in Npc_values
        par = make_par(Nv=Nv_npc, Npc=Npc, dt=dt_npc)
        e = run_one(par)
        verbose >= 1 && println("  Npc=$Npc  N=$(par.N)  ε=$e")
        push!(err_npc, e)
    end

    # ================================================================
    # Sweep 3: dt (time step)
    # ================================================================
    verbose >= 1 && println("\n=== Sweep 3: dt (time step) ===")
    err_dt = Float64[]
    for dt in dt_values
        par = make_par(Nv=Nv_dt, Npc=Npc_dt, dt=dt)
        e = run_one(par)
        verbose >= 1 && println("  dt=$dt  ntot=$(par.ntot)  ε=$e")
        push!(err_dt, e)
    end

    # ================================================================
    # Plots
    # ================================================================

    # --- Nv convergence ---
    p1 = plot(Nv_values, err_nv;
        xlabel     = L"N_v",
        ylabel     = L"\varepsilon_{L^2}",
        title      = "Convergence in grid resolution (Npc=$Npc_nv, dt=$dt_nv)",
        marker     = :circle, markersize = 6,
        yscale     = :log10, xscale = :log10,
        label      = "Landau/BKW",
        color      = :steelblue,
        _defaults...)
    # Reference slopes
    x_ref = float.(Nv_values)
    for (exp, ls, lbl) in [(-2, :dash, L"N_v^{-2}"), (-4, :dot, L"N_v^{-4}")]
        c_ref = err_nv[end] / x_ref[end]^exp
        plot!(p1, x_ref, c_ref .* x_ref.^exp; label=lbl, color=:gray, linestyle=ls, linewidth=1)
    end
    savefig(p1, joinpath(out_dir, "convergence_Nv.pdf"))
    savefig(p1, joinpath(out_dir, "convergence_Nv.png"))

    # --- Npc convergence ---
    N_total = Nv_npc^2 .* Npc_values
    p2 = plot(N_total, err_npc;
        xlabel     = L"N_{\mathrm{particles}}",
        ylabel     = L"\varepsilon_{L^2}",
        title      = "Convergence in particle count (Nv=$Nv_npc, dt=$dt_npc)",
        marker     = :circle, markersize = 6,
        yscale     = :log10, xscale     = :log10,
        label      = "Landau/BKW",
        color      = :darkorange,
        _defaults...)
    x_ref2 = float.(N_total)
    c_ref2 = err_npc[end] / x_ref2[end]^(-0.5)
    plot!(p2, x_ref2, c_ref2 .* x_ref2.^(-0.5); label=L"N^{-1/2}", color=:gray, linestyle=:dash, linewidth=1)
    savefig(p2, joinpath(out_dir, "convergence_Npc.pdf"))
    savefig(p2, joinpath(out_dir, "convergence_Npc.png"))

    # --- dt convergence ---
    p3 = plot(dt_values, err_dt;
        xlabel     = L"\Delta t",
        ylabel     = L"\varepsilon_{L^2}",
        title      = "Convergence in time step (Nv=$Nv_dt, Npc=$Npc_dt)",
        marker     = :circle, markersize = 6,
        yscale     = :log10, xscale     = :log10,
        label      = "Landau/BKW",
        color      = :seagreen,
        _defaults...)
    x_ref3 = float.(dt_values)
    for (exp, ls, lbl) in [(1, :dash, L"\Delta t"), (2, :dot, L"\Delta t^2")]
        c_ref3 = err_dt[end] / x_ref3[end]^exp
        plot!(p3, x_ref3, c_ref3 .* x_ref3.^exp; label=lbl, color=:gray, linestyle=ls, linewidth=1)
    end
    savefig(p3, joinpath(out_dir, "convergence_dt.pdf"))
    savefig(p3, joinpath(out_dir, "convergence_dt.png"))

    # --- Combined ---
    p_all = plot(p1, p2, p3;
        layout        = (1, 3),
        size          = (1500, 450),
        left_margin   = 8Plots.mm,
        bottom_margin = 8Plots.mm,
        dpi           = 200)
    savefig(p_all, joinpath(out_dir, "convergence_summary.pdf"))
    savefig(p_all, joinpath(out_dir, "convergence_summary.png"))

    verbose >= 1 && println("\nPlots saved to: $out_dir")

    return (
        Nv_values  = Nv_values,
        err_nv     = err_nv,
        Npc_values = Npc_values,
        err_npc    = err_npc,
        dt_values  = dt_values,
        err_dt     = err_dt,
    )
end

run_bkw_landau_convergence()
