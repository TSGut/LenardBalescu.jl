# 3D Lenard-Balescu demo: radial-symmetry path with Maxwellian initial condition.

using LenardBalescu
using Random
using Statistics
using Plots

function run_lenardbalescu_demo_3d_maxwell_radial(; ntot=20, seed=1, batchsize=0,
    out_dir=joinpath(@__DIR__, "output", "lenard_balescu_3d_maxwell_radial"),
    out_gif=joinpath(@__DIR__, "output", "lenard_balescu_3d_maxwell_radial", "lb_3d_particles_static.gif"),
    out_gif_rotating=joinpath(@__DIR__, "output", "lenard_balescu_3d_maxwell_radial", "lb_3d_particles_rotating.gif"),
    out_gif_combined=joinpath(@__DIR__, "output", "lenard_balescu_3d_maxwell_radial", "lb_3d_particles_combined.gif"),
    make_plots=true,
    make_animation=true,
    make_combined_evolution=true,
    max_anim_particles=8000,
    anim_fps=18,
    rotate_turns=0.1,
    projection_bins=60)

    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 3
    par.Nv = 10
    par.Npc = 1
    par.N = par.Nv^par.dim * par.Npc
    par.dt = 0.01
    par.ntot = ntot
    par.T = ntot * par.dt
    par.batchsize = batchsize

    par.k_perp_order = 20
    par.k_perp_max = 50.0
    par.k_parallel_width = 0.05
    par.k_parallel_order = 3

    update_kernel_params!(par)
    mkpath(out_dir)

    anim_static = Animation()
    anim_rotating = Animation()
    anim_combined = Animation()

    mass_hist = Float64[]
    energy_hist = Float64[]
    entropy_hist = Float64[]
    px_hist = Float64[]
    py_hist = Float64[]
    pz_hist = Float64[]

    edges = collect(range(-par.Lv, par.Lv; length=projection_bins + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    dV_grid = 2.0 * par.Lv / par.Nv

    function projected_density_2d(x::AbstractVector{Float64}, y::AbstractVector{Float64}, edges::Vector{Float64})
        nb = length(edges) - 1
        H = zeros(Float64, nb, nb)
        inv_dx = 1.0 / (edges[2] - edges[1])
        x0 = edges[1]
        @inbounds for i in eachindex(x)
            bx = clamp(floor(Int, (x[i] - x0) * inv_dx) + 1, 1, nb)
            by = clamp(floor(Int, (y[i] - x0) * inv_dx) + 1, 1, nb)
            H[by, bx] += 1.0
        end
        s = sum(H)
        if s > 0.0
            H ./= s
        end
        return H
    end

    function fixed_pad_ylims(v::AbstractVector{Float64}; pad=0.01)
        return (minimum(v) - pad, maximum(v) + pad)
    end

    function step_callback(k, vstack, wg, par, Vgrid, Vc)
        if !make_animation && !make_combined_evolution
            return nothing
        end
        vx = @view vstack[:, 1]
        vy = @view vstack[:, 2]
        vz = @view vstack[:, 3]
        np = length(vx)
        stride = max(1, Int(ceil(np / max_anim_particles)))
        idx = 1:stride:np
        speed = sqrt.(vx[idx].^2 .+ vy[idx].^2 .+ vz[idx].^2)

        p3_static = scatter3d(vx[idx], vy[idx], vz[idx];
            marker_z=speed,
            color=:viridis,
            markersize=2,
            markerstrokewidth=0,
            xlabel="v_x", ylabel="v_y", zlabel="v_z",
            xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), zlims=(-par.Lv, par.Lv),
            title="3D LB radial Maxwell (step=$k, t=$(round(k * par.dt, digits=3)) s)",
            camera=(45, 25),
            legend=false,
            size=(900, 750))

        az = 45.0 + 360.0 * rotate_turns * (k / max(par.ntot, 1))
        p3_rot = scatter3d(vx[idx], vy[idx], vz[idx];
            marker_z=speed,
            color=:viridis,
            markersize=2,
            markerstrokewidth=0,
            xlabel="v_x", ylabel="v_y", zlabel="v_z",
            xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), zlims=(-par.Lv, par.Lv),
            title="3D LB radial Maxwell rotating camera (step=$k, t=$(round(k * par.dt, digits=3)) s)",
            camera=(az, 25),
            legend=false,
            size=(900, 750))

        if make_animation
            frame(anim_static, p3_static)
            frame(anim_rotating, p3_rot)
        end

        if make_combined_evolution
            cq = compute_conserved_quantities(vstack, wg, par)
            push!(mass_hist, cq.mass)
            push!(energy_hist, cq.energy)
            push!(px_hist, cq.momentum[1])
            push!(py_hist, cq.momentum[2])
            push!(pz_hist, cq.momentum[3])
            f_grid = f_recon(vstack, wg, Vgrid, par)
            push!(entropy_hist, LenardBalescu.compute_entropy(f_grid, dV_grid; dim=par.dim))

            Hxy = projected_density_2d(vx, vy, edges)
            Hxz = projected_density_2d(vx, vz, edges)
            Hyz = projected_density_2d(vy, vz, edges)

            step_idx = 0:(length(mass_hist)-1)
            p_mass = plot(step_idx, mass_hist;
                xlabel="Step", ylabel="Mass", title="Mass", legend=false,
                xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(mass_hist))
            p_energy = plot(step_idx, energy_hist;
                xlabel="Step", ylabel="Energy", title="Energy", legend=false,
                xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(energy_hist))
            p_entropy = plot(step_idx, entropy_hist;
                xlabel="Step", ylabel="Entropy", title="Entropy", legend=false,
                xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(entropy_hist))
            p_momentum = plot(step_idx, px_hist;
                label="p_x", xlabel="Step", ylabel="Momentum", title="Momentum",
                xlims=(0, max(par.ntot, 1)))
            plot!(p_momentum, step_idx, py_hist; label="p_y")
            plot!(p_momentum, step_idx, pz_hist; label="p_z")
            plot!(p_momentum; ylims=fixed_pad_ylims(vcat(px_hist, py_hist, pz_hist)))

            p_xy = heatmap(centers, centers, Hxy;
                xlabel="v_x", ylabel="v_y", title="XY density", color=:viridis,
                xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), aspect_ratio=:equal)
            p_xz = heatmap(centers, centers, Hxz;
                xlabel="v_x", ylabel="v_z", title="XZ density", color=:viridis,
                xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), aspect_ratio=:equal)
            p_yz = heatmap(centers, centers, Hyz;
                xlabel="v_y", ylabel="v_z", title="YZ density", color=:viridis,
                xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), aspect_ratio=:equal)

            p_combined = plot(
                p3_static, p3_rot, p_entropy,
                p_xy, p_xz, p_yz,
                p_mass, p_energy, p_momentum;
                layout=(3, 3),
                size=(1850, 1600),
                margin=6Plots.mm,
                plot_title="3D Lenard-Balescu (Radial) Maxwell Evolution (step=$k)",
                plot_titlefontsize=14,
                widen=true
            )
            frame(anim_combined, p_combined)
        end

        return nothing
    end

    result = run_simulation(; par=par,
        rng=rng,
        use_lenard_balescu=true,
        radial_symmetry=true,
        outdir=out_dir,
        savecsv=true,
        trackconserved=true,
        step_callback=step_callback,
        init_particles=(N, dV, Vgrid, par, rng) -> initialize_particles_maxwellian(N, par, rng),
        verbose=1)

    if make_animation
        mkpath(dirname(out_gif))
        gif(anim_static, out_gif; fps=anim_fps)
        gif(anim_rotating, out_gif_rotating; fps=anim_fps)
    end

    if make_combined_evolution
        mkpath(dirname(out_gif_combined))
        gif(anim_combined, out_gif_combined; fps=anim_fps)
    end

    if make_plots
        steps = 0:(length(result.mass)-1)
        p_mass = plot(steps, result.mass;
            xlabel="Step", ylabel="Mass", title="Mass vs Step", legend=false,
            xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(result.mass))
        p_energy = plot(steps, result.energy;
            xlabel="Step", ylabel="Energy", title="Energy vs Step", legend=false,
            xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(result.energy))

        px = [m[1] for m in result.momentum]
        py = [m[2] for m in result.momentum]
        pz = [m[3] for m in result.momentum]
        p_momentum = plot(steps, px;
            label="p_x", xlabel="Step", ylabel="Momentum", title="Momentum vs Step",
            xlims=(0, max(par.ntot, 1)))
        plot!(p_momentum, steps, py; label="p_y")
        plot!(p_momentum, steps, pz; label="p_z")
        plot!(p_momentum; ylims=fixed_pad_ylims(vcat(px, py, pz)))

        p_entropy = plot(steps, result.entropy;
            xlabel="Step", ylabel="Entropy", title="Entropy vs Step", legend=false,
            xlims=(0, max(par.ntot, 1)), ylims=fixed_pad_ylims(result.entropy))
        p_err = plot(result.time, result.L1;
            xlabel="Time", ylabel="Error", title="L1/L2 Convergence", label="L1")
        plot!(p_err, result.time, result.L2; label="L2")
        err_path = joinpath(out_dir, "error_convergence_final.png")
        savefig(p_err, err_path)

        pos_L1 = (result.time .> 0.0) .& (result.L1 .> 0.0)
        pos_L2 = (result.time .> 0.0) .& (result.L2 .> 0.0)
        p_err_loglog = plot(result.time[pos_L1], result.L1[pos_L1];
            xlabel="Time", ylabel="Error", title="L1/L2 Convergence (Log-Log)",
            label="L1", xscale=:log10, yscale=:log10)
        plot!(p_err_loglog, result.time[pos_L2], result.L2[pos_L2]; label="L2")
        err_loglog_path = joinpath(out_dir, "error_convergence_loglog_final.png")
        savefig(p_err_loglog, err_loglog_path)
        @info "Saved images" linear=err_path loglog=err_loglog_path

        vmag = sqrt.(result.vx.^2 .+ result.vy.^2 .+ result.vz.^2)
        p_speed = histogram(vmag; bins=40, xlabel="|v|", ylabel="Count", title="Final speed distribution", legend=false)
        speed_path = joinpath(out_dir, "speed_hist_final.png")
        savefig(p_speed, speed_path)
        @info "Saved image" path=speed_path

        bins2d = projection_bins
        p_xy = histogram2d(result.vx, result.vy;
            bins=bins2d, xlabel="v_x", ylabel="v_y", title="Projection heatmap: XY",
            xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), color=:viridis, aspect_ratio=:equal)
        p_xz = histogram2d(result.vx, result.vz;
            bins=bins2d, xlabel="v_x", ylabel="v_z", title="Projection heatmap: XZ",
            xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), color=:viridis, aspect_ratio=:equal)
        p_yz = histogram2d(result.vy, result.vz;
            bins=bins2d, xlabel="v_y", ylabel="v_z", title="Projection heatmap: YZ",
            xlims=(-par.Lv, par.Lv), ylims=(-par.Lv, par.Lv), color=:viridis, aspect_ratio=:equal)

        p_summary = plot(
            p_mass, p_energy, p_momentum,
            p_entropy, p_err, p_speed,
            p_xy, p_xz, p_yz;
            layout=(3, 3),
            size=(1800, 1500),
            margin=6Plots.mm,
            plot_title="3D Lenard-Balescu (Radial) Maxwell Summary",
            plot_titlefontsize=14,
            widen=true)

        summary_path = joinpath(out_dir, "summary_final.png")
        savefig(p_summary, summary_path)
        @info "Saved image" path=summary_path
    end

    return nothing
end

run_lenardbalescu_demo_3d_maxwell_radial(ntot=100, seed=1, batchsize=0) # full interactions
