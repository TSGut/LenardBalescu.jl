# Lenard-Balescu collision demo with two concentric ring initial condition.
# Shows relaxation of a non-equilibrium distribution with two ring populations.
using LenardBalescu
using Random, Statistics
using Plots

function run_lenardbalescu_animation_rings(; ntot=80, seed=1, batchsize=0,
                                          out_gif=joinpath(@__DIR__, "output", "lenard_balescu_rings", "lb_rings.gif"),
                                          out_dir=joinpath(@__DIR__, "output", "lenard_balescu_rings", "frames"),
                                          save_pngs=true, plot_grid=true,
                                          ring1_fraction=0.15, ring1_radius=1.3, ring1_width=0.15,
                                          ring2_fraction=0.15, ring2_radius=2.7, ring2_width=0.15,
                                          radial_symmetry=true)
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 2
    par.Nv = 25    # number of velocity grid cells per dimension
    par.Npc = 2    # particles per cell (stratified sampling density)
    par.N = par.Nv * par.Nv * par.Npc  # total particle count
    par.dt = 0.001  # time step size
    par.T = ntot * par.dt  # total simulated time
    par.ntot = ntot  # number of time steps
    par.batchsize = batchsize

    # Explicitly declare k-space integration parameters for Lenard-Balescu kernel
    par.k_perp_order = 40      # Number of Gauss-Legendre quadrature points for k_perp (resolution; default: 15)
    par.k_perp_max = 20.0      # Maximum k_perp value (integration range)
    par.k_parallel_width = 0.035 # Hat-delta width for k_parallel (delta function approximation)
    par.k_parallel_order = 3    # Number of points for k_parallel (resolution of delta approximation)

    update_kernel_params!(par)

    # Custom initialization function for double rings
    function init_double_ring(Npart, dV, Vgrid, par, rng)
        return initialize_particles_stratified_double_ring(Npart, dV, Vgrid, par, rng;
                                                           include_bulk=false,
                                                           ring1_fraction=ring1_fraction,
                                                           ring1_radius=ring1_radius,
                                                           ring1_width=ring1_width,
                                                           ring2_fraction=ring2_fraction,
                                                           ring2_radius=ring2_radius,
                                                           ring2_width=ring2_width)
    end

    # Pre-compute visualization axis limits
    if plot_grid
        _, dV, Vc, Vgrid = build_velocity_grid(par)
        vstack0, wg_init = init_double_ring(par.N, dV, Vgrid, par, rng)
        f_grid0 = f_recon(vstack0, wg_init, Vgrid, par)
        clims = (minimum(f_grid0), maximum(f_grid0))

        r_max = par.Lv * sqrt(2)
        r_lim = (-r_max, r_max)

        r_centers0, f_binned0 = compute_radial_binned_f(f_grid0, Vc, par)
        _, r_min0, r_max0, f_interp0, f_deriv0 = fit_bspline(r_centers0, f_binned0)
        r_eval0 = range(r_min0, r_max0, length=200)
        f_smooth0 = f_interp0.(r_eval0)
        f_deriv_vals0 = f_deriv0.(r_eval0)

        f_min = min(minimum(f_binned0), minimum(f_smooth0))
        f_max = max(maximum(f_binned0), maximum(f_smooth0))
        f_margin = 0.25 * (f_max - f_min)
        f_radial_lim = (f_min - f_margin, f_max + f_margin)

        deriv_min = minimum(f_deriv_vals0)
        deriv_max = maximum(f_deriv_vals0)
        deriv_margin = 0.25 * (deriv_max - deriv_min)
        deriv_lim = (deriv_min - deriv_margin, deriv_max + deriv_margin)
    end

    if save_pngs
        mkpath(out_dir)
    end

    # Storage for animation frames and conserved quantities
    frames = []
    mass_hist = Float64[]
    energy_hist = Float64[]
    momentum_hist = Vector{Float64}[]

    # Callback function for visualization at each step
    function visualization_callback(k, vx, vy, wg, par, Vgrid, Vc)
        vstack = hcat(vx, vy)
        # Compute and store conserved quantities
        cq = compute_conserved_quantities(vstack, wg, par)
        push!(mass_hist, cq.mass)
        push!(energy_hist, cq.energy)
        push!(momentum_hist, cq.momentum)

        f_grid = f_recon(vstack, wg, Vgrid, par)

        if plot_grid
            # Particle positions plot
            p_particles = scatter(vx, vy;
                xlims=(-par.Lv, par.Lv),
                ylims=(-par.Lv, par.Lv),
                markersize=2,
                markerstrokewidth=0,
                alpha=0.5,
                label="",
                xlabel="vx",
                ylabel="vy",
                title="Particles (t=$(round(k*par.dt, digits=2)))",
                aspect_ratio=:equal,
                titlefontsize=12,
                guidefontsize=10,
                tickfontsize=9,
                right_margin=2Plots.mm,
                left_margin=6Plots.mm,
                bottom_margin=6Plots.mm,
                widen=true)

            # Reconstructed distribution heatmap
            f_mat = reshape(f_grid, par.Nv, par.Nv)
            p_heatmap = heatmap(Vc, Vc, f_mat;
                xlabel="vx",
                ylabel="vy",
                title="f(vx, vy)",
                aspect_ratio=:equal,
                color=:viridis,
                clims=clims,
                titlefontsize=12,
                guidefontsize=10,
                tickfontsize=9,
                colorbar_titlefontsize=10,
                right_margin=8Plots.mm,
                left_margin=6Plots.mm,
                bottom_margin=6Plots.mm,
                widen=true)

            # Radial distribution analysis
            r_centers, f_binned = compute_radial_binned_f(f_grid, Vc, par)
            coeffs, r_min, r_max, f_interp, f_deriv = fit_bspline(r_centers, f_binned)

            r_smooth = range(r_min, r_max, length=200)
            f_smooth = f_interp.(r_smooth)
            f_prime_smooth = f_deriv.(r_smooth)

            f_prime_fd = diff(f_smooth) ./ diff(r_smooth)
            r_smooth_fd = r_smooth[1:end-1] .+ diff(r_smooth) ./ 2

            # Radial distribution plot
            p_radial_interp = scatter(r_centers, f_binned;
                label="Binned data",
                markersize=3,
                xlabel="r",
                ylabel="f(r)",
                title="Radial distribution",
                xlims=r_lim,
                ylims=f_radial_lim,
                titlefontsize=12,
                guidefontsize=10,
                tickfontsize=9,
                legendfontsize=9,
                right_margin=2Plots.mm,
                left_margin=6Plots.mm,
                bottom_margin=6Plots.mm,
                widen=true)
            plot!(p_radial_interp, r_smooth, f_smooth;
                linestyle=:solid,
                linewidth=2,
                label="Cubic B-spline",
                legend=:topright)

            # Derivative plot
            p_radial_deriv = plot(r_smooth, f_prime_smooth;
                xlabel="r",
                ylabel="df/dr",
                title="Radial derivative",
                linewidth=2,
                label="Polynomial",
                xlims=r_lim,
                ylims=deriv_lim,
                titlefontsize=12,
                guidefontsize=10,
                tickfontsize=9,
                right_margin=2Plots.mm,
                left_margin=6Plots.mm,
                bottom_margin=6Plots.mm,
                widen=true)
            plot!(p_radial_deriv, r_smooth_fd, f_prime_fd;
                linewidth=2,
                linestyle=:dash,
                label="Finite difference",
                legend=:topright)

            # Combine into 2x2 grid
            p_combined = plot(p_particles, p_heatmap, p_radial_interp, p_radial_deriv;
                layout=(2, 2),
                size=(1000, 900),
                plot_title="Lenard-Balescu Collision (Double Ring)",
                plot_titlefontsize=14,
                widen=true)

            if save_pngs
                frame_name = string("frame_", lpad(k, 4, '0'), ".png")
                frame_path = joinpath(out_dir, frame_name)
                savefig(p_combined, frame_path)
            end

            push!(frames, p_combined)
        end
        return nothing
    end

    # Run simulation with visualization callback and custom initialization
    result = run_simulation(; par=par, rng=rng, use_lenard_balescu=true,
                            outdir=joinpath(@__DIR__, "output", "lenard_balescu_rings"),
                            step_callback=visualization_callback,
                            init_particles=init_double_ring,
                            verbose=1,
                            trackconserved=true,
                            radial_symmetry=radial_symmetry)

    # Plot conserved quantities at the end in a 2x2 grid (bottom right)
    steps = 0:(length(mass_hist)-1)
    p_mass = plot(steps, mass_hist, xlabel="Step", ylabel="Mass", title="Mass vs Step", legend=false)
    energy_min = minimum(energy_hist)
    energy_max = maximum(energy_hist)
    p_energy = plot(steps, energy_hist, xlabel="Step", ylabel="Energy", title="Energy vs Step", legend=false, ylims=(energy_min-0.1, energy_max+0.1))
    px = [m[1] for m in momentum_hist]
    py = [m[2] for m in momentum_hist]
    p_momentum = plot(steps, px, label="p_x", xlabel="Step", ylabel="Momentum", title="Momentum vs Step")
    plot!(p_momentum, steps, py, label="p_y")
    p_entropy = plot(steps, result.entropy, xlabel="Step", ylabel="Entropy", title="Entropy vs Step", legend=false)
    p_conserved = plot(p_mass, p_energy, p_momentum, p_entropy; layout=(2,2), size=(1000,900), margin=6Plots.mm, widen=true)
    parent_dir = joinpath(@__DIR__, "output", "lenard_balescu_rings")
    mkpath(parent_dir)
    conserved_png_path = joinpath(parent_dir, "conserved_final.png")
    savefig(p_conserved, conserved_png_path)
    @info "Conserved quantities plot saved to: $conserved_png_path"

    p_L1 = plot(result.time, result.L1; xlabel="Time", ylabel="L1 Error", title="L1 Convergence", legend=false)
    p_L2 = plot(result.time, result.L2; xlabel="Time", ylabel="L2 Error", title="L2 Convergence", legend=false)
    p_errors = plot(p_L1, p_L2; layout=(1,2), size=(1000,450), margin=6Plots.mm, widen=true)
    errors_png_path = joinpath(parent_dir, "errors_final.png")
    savefig(p_errors, errors_png_path)

    pos_L1 = (result.time .> 0.0) .& (result.L1 .> 0.0)
    pos_L2 = (result.time .> 0.0) .& (result.L2 .> 0.0)
    p_L1_log = plot(result.time[pos_L1], result.L1[pos_L1];
        xlabel="Time", ylabel="L1 Error", title="L1 Convergence (Log-Log)",
        legend=false, xscale=:log10, yscale=:log10)
    p_L2_log = plot(result.time[pos_L2], result.L2[pos_L2];
        xlabel="Time", ylabel="L2 Error", title="L2 Convergence (Log-Log)",
        legend=false, xscale=:log10, yscale=:log10)
    p_errors_loglog = plot(p_L1_log, p_L2_log; layout=(1,2), size=(1000,450), margin=6Plots.mm, widen=true)
    errors_loglog_png_path = joinpath(parent_dir, "errors_loglog_final.png")
    savefig(p_errors_loglog, errors_loglog_png_path)
    @info "Error plots saved to:" linear=errors_png_path loglog=errors_loglog_png_path

    # Create animation from frames
    if plot_grid
        anim = Animation()
        for p in frames
            frame(anim, p)
        end
        mkpath(dirname(out_gif))
        gif(anim, out_gif, fps=10)
    end
    return nothing
end

# Convenience function with default parameters
run_lenardbalescu_animation_rings(ntot=100, seed=1, batchsize=0, radial_symmetry=true); # full interactions
