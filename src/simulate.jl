"""
    partition_random_batches(rng, N, batchsize)

Randomly partition `1:N` into balanced batches used by RBM acceleration.

For `batchsize < 2` or `batchsize >= N`, returns a single batch containing all
indices and `redistributed_remainder = 0`.

# Returns
- `batches::Vector{Vector{Int}}`: random balanced partition of particle indices
- `redistributed_remainder::Int`: number of batches that received one extra index
"""
function partition_random_batches(rng::AbstractRNG, N::Int, batchsize::Int)
    @assert N >= 0 "N must be non-negative"
    if N == 0
        return Vector{Vector{Int}}(), 0
    end

    perm = randperm(rng, N)
    if batchsize < 2 || batchsize >= N
        return [perm], 0
    end

    nbatches = cld(N, batchsize)
    base_size = fld(N, nbatches)
    redistributed_remainder = N - base_size * nbatches

    batches = Vector{Vector{Int}}(undef, nbatches)
    offset = 1
    @inbounds for q in 1:nbatches
        m = base_size + (q <= redistributed_remainder ? 1 : 0)
        batches[q] = perm[offset:(offset + m - 1)]
        offset += m
    end
    return batches, redistributed_remainder
end

"""
        run_simulation(; par=SetParam(), outdir="../data", snapshot_interval=:auto,
                                     rng=MersenneTwister(0), use_lenard_balescu=true,
                                     step_callback=nothing, init_particles=nothing,
                                     verbose=1, savecsv=false, trackconserved=false,
                                     enforce_energy=true, radial_symmetry=false)

Run a kinetic collision simulation using either the Lenard-Balescu or Landau operator.

By default (`use_lenard_balescu=true`), uses the Lenard-Balescu collision operator with screening.
Set `use_lenard_balescu=false` to run the Landau operator.

# Arguments
- `par` (default: `SetParam()`): simulation parameters (`Params` struct)
        - `par.batchsize` controls random-batch RBM acceleration.
            Use `0` (default) or `1` to disable batching (full interaction).
            For `2 <= par.batchsize < N`, each timestep randomly repartitions particles
            into balanced batches and interactions are only computed inside each batch,
            with per-batch normalization `(N-1)/(m-1)` for batch size `m`.
- `outdir` (default: `"../data"`): output directory for data files
- `snapshot_interval` (default: `:auto`): how often to save snapshots (`:auto` or integer)
- `rng` (default: `MersenneTwister(0)`): random number generator
- `use_lenard_balescu` (default: `true`): choose Lenard-Balescu (`true`) or Landau (`false`)
- `step_callback` (default: `nothing`): optional callback; supported signatures are
    `(k, vstack, wg, par, Vgrid, Vc)`, `(k, vx, vy, wg, par, Vgrid, Vc)`, and for 3D
    `(k, vx, vy, vz, wg, par, Vgrid, Vc)`
- `init_particles`: custom initialization function with signature `(Npart, dV, Vgrid, par, rng) -> (vstack, wg)`
    where `vstack` is `N×2` in 2D or `N×3` in 3D. If `nothing`, defaults to stratified BKW (2D)
    or Maxwellian (3D).
- `verbose` (default: `1`): verbosity level (`0` silent, `1` progress, `2` detailed)
- `savecsv` (default: `false`): when true, write CSV outputs to `outdir`
- `trackconserved` (default: `false`): when true, compute and return conserved/entropy histories
- `enforce_energy` (default: `true`): when true, rescales particle deviations each step to match
    the initial weighted variance (energy proxy)
- `radial_symmetry` (default: `false`): Lenard-Balescu dielectric preprocessing mode
- `static_dielectric` (default: `false`): when `true`, use static dielectric ε_static(k)=1+α V̂(k) with α=`par.alpha` instead of the dynamic G/HG path (skips per-step dielectric precomputation)

# Returns
Named tuple with simulation history and final particle state.
Always includes `L1`, `L2`, `time`, `collision_times`, `vx`, `vy`, `vz`, `wg`.

If `trackconserved=true`, also includes `mass`, `momentum`, `energy`, `entropy`, and
final deltas (`delta_mass`, `delta_momentum`, `delta_energy`, `delta_entropy`).
"""

function run_simulation(; par::Params=SetParam(), outdir::AbstractString="../data",
                        snapshot_interval::Union{Int, Symbol}=:auto,
                        rng::AbstractRNG=MersenneTwister(0),
                        use_lenard_balescu::Bool=true,
                        step_callback::Union{Function,Nothing}=nothing,
                        init_particles::Union{Function,Nothing}=nothing,
                        verbose::Int=1,
                        savecsv::Bool=false,
                        trackconserved::Bool=false,
                        enforce_energy::Bool=true,
                        radial_symmetry::Bool=false,
                        static_dielectric::Bool=false)
    # Log operator choice
    if verbose >= 1 && !use_lenard_balescu
        println("Note: Using simple Landau operator (not Lenard-Balescu).")
    end

    # Validate simulation dimension early
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    @assert par.batchsize >= 0 "Params.batchsize must be >= 0 (got=$(par.batchsize))"
    @assert par.spline_order in (0, 1, 2, 3) "Params.spline_order must be one of 0,1,2,3 (got=$(par.spline_order))"
    
    # Use default initialization if none provided
    if isnothing(init_particles)
        if par.dim == 2
            if verbose >= 1
                println("Note: Using default BKW distribution for initialization.")
            end
            init_particles = initialize_particles_stratified_BKW
        else
            if verbose >= 1
                println("Note: Using default Maxwellian initialization for dim=3.")
            end
            init_particles = (Npart, dV, Vgrid, par, rng) ->
                initialize_particles_maxwellian(Npart, par, rng)
        end
    end
    
    # Setup output directory and initialize parameters
    mkpath(outdir)

    Npart = (par.N > 0) ? par.N : par.Nv^par.dim * par.Npc
    dt = par.dt
    T = par.T
    ntot = (par.ntot > 0) ? par.ntot : Int(ceil(T / dt))

    rbm_enabled_summary = (par.batchsize >= 2) && (par.batchsize < Npart)
    effective_batchsize_summary = rbm_enabled_summary ? par.batchsize : Npart
    nbatches_per_step_summary = rbm_enabled_summary ? cld(Npart, effective_batchsize_summary) : 1

    update_kernel_params!(par)


    # Print parameter summary with box
    if verbose >= 1
        println("──────────────────────────────────────────────────────")
        println("  SIMULATION PARAMETERS")
        println("──────────────────────────────────────────────────────")
        println("  Dimension: dim = $(par.dim)D")
        println("  Velocity grid: Nv = $(par.Nv) (velocity range ±$(par.Lv))")
        println("  Particles: N = $Npart ($(par.Npc) per grid cell)")
        println("  Time: T = $T s, dt = $dt s ($ntot steps)")
        if rbm_enabled_summary
            println("  Batching mode: RBM (requested batch size = $effective_batchsize_summary, batches/step = $nbatches_per_step_summary)")
        else
            println("  Batching mode: Full interaction (effective batch size = $effective_batchsize_summary)")
        end
        if use_lenard_balescu
            if static_dielectric
                println("  Kernel: Lenard-Balescu (static, α=$(par.alpha))")
            else
                println("  Kernel: Lenard-Balescu")
                println("  Radial symmetry assumed: $radial_symmetry")
            end
        else
            println("  Kernel: Landau γ = $(par.gam), β = $(par.beta)")
        end
        println("──────────────────────────────────────────────────────")
    end

    V, dV, Vc, Vgrid = build_velocity_grid(par)
    if use_lenard_balescu && !static_dielectric
        dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=radial_symmetry)
    else
        # Unused on Landau path or static-dielectric path; keep concrete values to satisfy keyword types.
        dielectric_r0 = 0.0
        dielectric_dr = 1.0
        dielectric_inv_dr = 1.0
        dielectric_len = 1
    end

    # Initialize particle positions and weights.
    vstack, wg = init_particles(Npart, dV, Vgrid, par, rng)
    @assert ndims(vstack) == 2 "init_particles must return a 2D matrix for vstack"
    @assert size(vstack, 2) == par.dim "init_particles vstack must have $(par.dim) columns for dim=$(par.dim)"
    vx = @view vstack[:, 1]
    vy = @view vstack[:, 2]
    vz = par.dim == 3 ? (@view vstack[:, 3]) : nothing
    if length(vx) != Npart
        Npart = length(vx)
    end

    rbm_enabled = (par.batchsize >= 2) && (par.batchsize < Npart)
    effective_batchsize = rbm_enabled ? par.batchsize : Npart
    nbatches_per_step = rbm_enabled ? cld(Npart, effective_batchsize) : 1
    redistributed_remainder = rbm_enabled ? (Npart - fld(Npart, nbatches_per_step) * nbatches_per_step) : 0
    
    # Compute Maxwellian reference matched to the particle ensemble's
    # conserved quantities (mass, momentum, temperature).
    cq0 = compute_conserved_quantities(vstack, wg, par)
    mass0 = cq0.mass
    mom0 = cq0.momentum
    energy0 = cq0.energy
    # mean velocity
    u = mom0 ./ mass0
    # Target variance (scalar) about the mean for energy rescaling
    if par.dim == 2
        sigma_target = sum(wg .* ((vx .- u[1]).^2 .+ (vy .- u[2]).^2))
    else
        sigma_target = sum(wg .* ((vx .- u[1]).^2 .+ (vy .- u[2]).^2 .+ (vz .- u[3]).^2))
    end
    sigma_target = max(sigma_target, eps(Float64))
    # Effective equilibrium temperature from thermal variance about the mean:
    # sum(wg * |v-u|^2) = dim * mass * T for Maxwellian in `dim` dimensions.
    T_eff = sigma_target / (par.dim * mass0)
    T_eff = max(T_eff, eps(Float64))
    grid_views = component_grid_views(Vgrid, par)
    r2 = Vector{Float64}(undef, length(grid_views[1]))
    if par.dim == 2
        vx_grid, vy_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = (vx_grid[k] - u[1])^2 + (vy_grid[k] - u[2])^2
        end
    else
        vx_grid, vy_grid, vz_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = (vx_grid[k] - u[1])^2 + (vy_grid[k] - u[2])^2 + (vz_grid[k] - u[3])^2
        end
    end
    f0_flat = exp.(-r2 ./ (2.0 * T_eff))
    norm = mass0 / (sum(f0_flat) * dV ^ par.dim)
    f_analytic = f0_flat .* norm

    # Initialize error history arrays
    L1_hist = Float64[]
    L2_hist = Float64[]
    time_hist = Float64[]
    coll_time_hist = Float64[]

    # Compute and (optionally) save initial state errors
    f_grid = f_recon(vstack, wg, Vgrid, par)
    push!(L1_hist, L1_error(f_grid, f_analytic, dV; dim=par.dim))
    push!(L2_hist, L2_error(f_grid, f_analytic, dV; dim=par.dim))
    push!(time_hist, 0.0)
    if savecsv
        write_vector_csv("f_recon_t0.csv", f_grid; dir=outdir)
    end

    # Prepare conserved quantity history if requested
    mass_hist = trackconserved ? Float64[] : nothing
    momentum_hist = trackconserved ? Vector{Vector{Float64}}() : nothing
    energy_hist = trackconserved ? Float64[] : nothing
    entropy_hist = trackconserved ? Float64[] : nothing
    if trackconserved
        cq = compute_conserved_quantities(vstack, wg, par)
        push!(mass_hist, cq.mass)
        push!(momentum_hist, cq.momentum)
        push!(energy_hist, cq.energy)

        # Initial entropy
        S0 = compute_entropy(f_grid, dV; dim=par.dim)
        push!(entropy_hist, S0)
    end

    # Call callback for initial state (step 0)
    if !isnothing(step_callback)
        if applicable(step_callback, 0, vstack, wg, par, Vgrid, Vc)
            step_callback(0, vstack, wg, par, Vgrid, Vc)
        elseif par.dim == 2 && applicable(step_callback, 0, vx, vy, wg, par, Vgrid, Vc)
            step_callback(0, vx, vy, wg, par, Vgrid, Vc)
        elseif par.dim == 3 && applicable(step_callback, 0, vx, vy, vz, wg, par, Vgrid, Vc)
            step_callback(0, vx, vy, vz, wg, par, Vgrid, Vc)
        else
            error("step_callback has unsupported signature. Supported forms are (k, vstack, wg, par, Vgrid, Vc), (k, vx, vy, wg, par, Vgrid, Vc), and for 3D (k, vx, vy, vz, wg, par, Vgrid, Vc).")
        end
    end

    interval = snapshot_interval === :auto ? max(1, ntot ÷ 10) : Int(snapshot_interval)

    # Main time stepping loop
    if verbose >= 1
        println("\nRunning simulation...")
    end
    iter_times = Float64[]  # Track iteration times for ETA calculation
    loop_start_time = time()
    for k in 1:ntot
        iter_start = time()
        t = k * dt

        if verbose >= 2
            println("\n  Step $k/$ntot, t = $t s")
        end

        # Compute collision operator and advection step
        if verbose >= 2
            print("    Computing collision operator...")
        end
        elapsed = Ref(0.0)
        if rbm_enabled
            batches, _ = partition_random_batches(rng, Npart, effective_batchsize)
            sv = zeros(Float64, Npart, par.dim)
            t_batch_start = time_ns()
            if verbose >= 2
                batch_sizes = [length(b) for b in batches]
                println("    RBM batches this step: $(length(batches)) (min size=$(minimum(batch_sizes)), max size=$(maximum(batch_sizes)))")
            end

            # Compute global fields from all particles, then batch only pairwise interactions.
            fields = compute_collision_fields(vstack, par, wg)
            kernel_state = precompute_collision_kernel_state(
                vstack, par, wg;
                Vgrid=Vgrid,
                use_lenard_balescu=use_lenard_balescu,
                radial_symmetry=radial_symmetry,
                static_dielectric=static_dielectric,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len
            )

            for idx in batches
                m = length(idx)
                if m <= 1
                    continue
                end
                sv_batch = collision_pairwise_subset(
                    vstack, par, wg,
                    fields.Fx, fields.Fy, fields.Fz,
                    idx, idx;
                    use_lenard_balescu=use_lenard_balescu,
                    kernel_state=kernel_state
                )
                alpha_batch = (Npart - 1) / (m - 1)
                @views sv[idx, :] .= alpha_batch .* sv_batch
            end
            elapsed[] = (time_ns() - t_batch_start) / 1.0e9
        else
            sv = Collision(
                vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=use_lenard_balescu,
                verbose=verbose,
                radial_symmetry=radial_symmetry,
                static_dielectric=static_dielectric,
                dielectric_r0=dielectric_r0,
                dielectric_dr=dielectric_dr,
                dielectric_inv_dr=dielectric_inv_dr,
                dielectric_len=dielectric_len
            )
        end
        push!(coll_time_hist, elapsed[])

        # Apply collision accelerations (sv is N x dim)
        vstack .-= dt .* sv

        if verbose >= 2
            println(" done ($(round(elapsed[]*1000, digits=1))ms)")
        end

        # Optional: enforce energy (variance) by rescaling deviations
        if enforce_energy
            small = eps(Float64)
            if par.dim == 2
                sigma_star = sum(wg .* ((vx .- u[1]).^2 .+ (vy .- u[2]).^2))
            else
                sigma_star = sum(wg .* ((vx .- u[1]).^2 .+ (vy .- u[2]).^2 .+ (vz .- u[3]).^2))
            end
            if sigma_star > small
                α = sqrt(sigma_target / sigma_star)
                # Rescale deviations about the mean for each component
                for j in 1:par.dim
                    vstack[:, j] .-= u[j]
                    vstack[:, j] .*= α
                    vstack[:, j] .+= u[j]
                end
            else
                @warn "Skipping energy rescale at step $k: sigma_star ≈ $sigma_star"
            end
        end

        # Track conserved quantities if requested
        if trackconserved
            cq = compute_conserved_quantities(vstack, wg, par)
            push!(mass_hist, cq.mass)
            push!(momentum_hist, cq.momentum)
            push!(energy_hist, cq.energy)
        end

        # Call user-provided callback for custom visualization/logging
        if !isnothing(step_callback)
            if applicable(step_callback, k, vstack, wg, par, Vgrid, Vc)
                step_callback(k, vstack, wg, par, Vgrid, Vc)
            elseif par.dim == 2 && applicable(step_callback, k, vx, vy, wg, par, Vgrid, Vc)
                step_callback(k, vx, vy, wg, par, Vgrid, Vc)
            elseif par.dim == 3 && applicable(step_callback, k, vx, vy, vz, wg, par, Vgrid, Vc)
                step_callback(k, vx, vy, vz, wg, par, Vgrid, Vc)
            else
                error("step_callback has unsupported signature. Supported forms are (k, vstack, wg, par, Vgrid, Vc), (k, vx, vy, wg, par, Vgrid, Vc), and for 3D (k, vx, vy, vz, wg, par, Vgrid, Vc).")
            end
        end

        # Reconstruct distribution and compute errors
        f_grid_k = f_recon(vstack, wg, Vgrid, par)
        # Maxwellian reference is time-independent for conserved quantities
        f_analytic_t = f_analytic

        L1k = L1_error(f_grid_k, f_analytic_t, dV; dim=par.dim)
        L2k = L2_error(f_grid_k, f_analytic_t, dV; dim=par.dim)
        push!(L1_hist, L1k)
        push!(L2_hist, L2k)
        push!(time_hist, t)
        if trackconserved # Entropy at this step
            S_k = compute_entropy(f_grid_k, dV; dim=par.dim)
            push!(entropy_hist, S_k)
        end

        # Save snapshots at regular intervals (optional)
        if savecsv && (k % interval == 0)
            fname = "f_recon_t" * string(k) * ".csv"
            write_vector_csv(fname, f_grid_k; dir=outdir)
        end
        
        # Track iteration time
        iter_time = time() - iter_start
        push!(iter_times, iter_time)
        if length(iter_times) > 10
            popfirst!(iter_times)  # Keep only last 10
        end
        
        # Update progress bar every 10 steps
        if verbose >= 1 && (k % 10 == 0 || k == ntot)
            progress = k / ntot
            bar_length = 30
            filled = Int(floor(progress * bar_length))
            bar = repeat("=", filled) * repeat(" ", bar_length - filled)
            percent = Int(round(progress * 100))
            
            # Calculate ETA based on average of last 10 iterations
            if length(iter_times) > 0
                avg_time = sum(iter_times) / length(iter_times)
                remaining = ntot - k
                eta_seconds = avg_time * remaining
                eta_str = if eta_seconds < 60
                    string(round(eta_seconds, digits=1), "s")
                elseif eta_seconds < 3600
                    string(round(eta_seconds / 60, digits=1), "m")
                else
                    string(round(eta_seconds / 3600, digits=1), "h")
                end
                print("\r  [$bar] $percent% ($k/$ntot) | $(round(avg_time*1000, digits=1))ms/iter | ETA: $eta_str")
            else
                print("\r  [$bar] $percent% ($k/$ntot steps)")
            end
        end
    end
    if verbose >= 1
        println()
        # Print total simulation run time
        total_time = time() - loop_start_time
        time_str = if total_time < 60
            string(round(total_time, digits=2), " seconds")
        elseif total_time < 3600
            mins = floor(total_time / 60)
            secs = total_time - mins * 60
            string(Int(mins), "m ", round(secs, digits=1), "s")
        else
            hours = floor(total_time / 3600)
            mins = floor((total_time - hours * 3600) / 60)
            string(Int(hours), "h ", Int(mins), "m")
        end
        println("Simulation completed in $time_str.")
    end

    # Write error history to CSV (optional)
    if savecsv
        errors_path = joinpath(outdir, "errors.csv")
        open(errors_path, "w") do io
            write(io, "t,L1,L2\n")
            for i in eachindex(time_hist)
                println(io, "$(time_hist[i]),$(L1_hist[i]),$(L2_hist[i])")
            end
        end
    end

    coll_path = joinpath(outdir, "collision_times.csv")
    # Write collision operator timing data (optional)
    if savecsv
        open(coll_path, "w") do io
            write(io, "step,time_s\n")
            for k in eachindex(coll_time_hist)
                println(io, "$k,$(coll_time_hist[k])")
            end
        end
    end

        # Prepare output
          result = (vx=vx, vy=vy, vz=vz, wg=wg, L1=L1_hist, L2=L2_hist,
              time=time_hist, collision_times=coll_time_hist,
              mass=mass_hist, momentum=momentum_hist, energy=energy_hist,
              entropy=entropy_hist)

        # If conservation tracked, add deltas (final - initial) and print them
        if trackconserved && mass_hist !== nothing && !isempty(mass_hist)
            delta_mass = mass_hist[end] - mass_hist[1]
            delta_momentum = momentum_hist[end] .- momentum_hist[1]
            delta_energy = energy_hist[end] - energy_hist[1]
            # Add entropy delta if available
            if entropy_hist !== nothing && !isempty(entropy_hist)
                delta_entropy = entropy_hist[end] - entropy_hist[1]
                result = merge(result, (delta_mass=delta_mass, delta_momentum=delta_momentum, delta_energy=delta_energy, delta_entropy=delta_entropy))
                println("Δmass = ", delta_mass)
                println("Δmomentum = ", delta_momentum)
                println("Δenergy = ", delta_energy)
                println("Δentropy = ", delta_entropy)
            else
                result = merge(result, (delta_mass=delta_mass, delta_momentum=delta_momentum, delta_energy=delta_energy))
                println("Δmass = ", delta_mass)
                println("Δmomentum = ", delta_momentum)
                println("Δenergy = ", delta_energy)
            end
        end
        return result
end
