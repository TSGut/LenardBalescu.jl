"""

    Collision(vstack, par, wg, elapsed_s=nothing;
              Vgrid=nothing, use_lenard_balescu=true, verbose=0,
              radial_symmetry=true, static_dielectric=false,
              dielectric_r0, dielectric_dr,
              dielectric_inv_dr, dielectric_len)

Lenard-Balescu (or Landau) collision operator for particle velocities.

# Arguments
- `vstack`: `N×dim` particle velocities (`dim = par.dim`, supports 2 or 3)
- `par`: Simulation parameters
- `wg`: Particle weights
- `elapsed_s`: (optional) Ref for computation time
- `Vgrid`: (optional) velocity grid used for reconstruction/precomputation
- `use_lenard_balescu`: Use Lenard-Balescu (true) or Landau (false)
- `verbose`: Verbosity level
- `radial_symmetry`: Controls dielectric preprocessing mode for Lenard-Balescu
- `static_dielectric`: Use static dielectric ε_static(k)=1+αV̂(k) with α=`par.alpha` (skips G/HG precomputation)
- `dielectric_r0`: Precomputed interpolation-grid origin (first grid value)
- `dielectric_dr`: Precomputed interpolation-grid spacing
- `dielectric_inv_dr`: Precomputed inverse spacing (`1/dr`) for fast indexing
- `dielectric_len`: Precomputed interpolation-grid length

# Returns
- `N×dim` matrix `sv`, with `dv/dt = -sv` (the driver subtracts `dt * sv`)

# Notes
- Landau path supports `dim = 2` and `dim = 3`.
- Lenard-Balescu path supports `dim = 2` and `dim = 3` in both radial
    (`radial_symmetry=true`) and generic (`radial_symmetry=false`) modes.
"""
function Collision(vstack::AbstractMatrix{Float64},
                   par::Params, wg::AbstractVector{Float64},
                   elapsed_s::Union{Nothing, Base.RefValue{Float64}}=nothing;
                   Vgrid::Union{Nothing, AbstractArray{Float64}}=nothing,
                   use_lenard_balescu::Bool=true,
                   verbose::Int=0,
                   radial_symmetry::Bool=true,
                   static_dielectric::Bool=false,
                   dielectric_r0::Float64,
                   dielectric_dr::Float64,
                   dielectric_inv_dr::Float64,
                   dielectric_len::Int)
    measure_time = elapsed_s !== nothing
    if measure_time
        tstart = time_ns()
    end
    # Validate simulation dimension and vstack shape
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    N, dim = size(vstack)
    @assert dim == par.dim "vstack second dimension must match par.dim"
    @assert length(wg) == N "Expected wg length to match number of particles"
    svxj = zeros(Float64, N)
    svyj = zeros(Float64, N)
    if par.dim == 3
        svzj = zeros(Float64, N)
    end

    # Views for velocity components
    vx = @view vstack[:, 1]
    vy = @view vstack[:, 2]
    vz = par.dim == 3 ? (@view vstack[:, 3]) : nothing
    if N == 0
        if elapsed_s !== nothing
            elapsed_s[] = 0.0
        end
        return zeros(Float64, 0, par.dim)
    end

    NV = par.Nv
    Lv = par.Lv
    Ncells = NV^par.dim
    cell_h = (par.cell_h > 0.0) ? par.cell_h : (2.0 * Lv / NV)
    epsi = par.epsi

    order = par.spline_order

    # Default kernel
    kernel = BasisKernel(order, epsi)

    use_cell_list = par.use_cell_list
    if use_cell_list
        # Build spatial cell list
        cell_particles = [Int[] for _ in 1:Ncells]
        if par.dim == 2
            @inbounds for p in 1:N
                xnorm = (vx[p] + Lv) / (2.0 * Lv)
                ynorm = (vy[p] + Lv) / (2.0 * Lv)
                ix = clamp(floor(Int, xnorm * NV), 0, NV - 1)
                iy = clamp(floor(Int, ynorm * NV), 0, NV - 1)
                c = ix + iy * NV + 1
                push!(cell_particles[c], p)
            end
        else  # par.dim == 3
            @inbounds for p in 1:N
                xnorm = (vx[p] + Lv) / (2.0 * Lv)
                ynorm = (vy[p] + Lv) / (2.0 * Lv)
                znorm = (vz[p] + Lv) / (2.0 * Lv)
                ix = clamp(floor(Int, xnorm * NV), 0, NV - 1)
                iy = clamp(floor(Int, ynorm * NV), 0, NV - 1)
                iz = clamp(floor(Int, znorm * NV), 0, NV - 1)
                c = ix + iy * NV + iz * NV * NV + 1
                push!(cell_particles[c], p)
            end
        end

        rCells = neighbor_radius_cells(kernel, cell_h; gaussian_cutoff_sigma=par.gaussian_cutoff_sigma)
        # Disable cell list if neighbor radius covers entire grid.
        if rCells >= NV
            use_cell_list = false
        else
            neighbor_cells = [Int[] for _ in 1:Ncells]
            if par.dim == 2
                for c in 1:Ncells
                    ix = (c - 1) % NV
                    iy = (c - 1) ÷ NV
                    neigh = Int[]
                    for dy in -rCells:rCells
                        for dx in -rCells:rCells
                            nx = ix + dx
                            ny = iy + dy
                            if 0 <= nx < NV && 0 <= ny < NV
                                push!(neigh, nx + ny * NV + 1)
                            end
                        end
                    end
                    neighbor_cells[c] = neigh
                end
            else  # par.dim == 3
                for c in 1:Ncells
                    ix = mod(c - 1, NV)
                    iy = div(mod(c - 1, NV * NV), NV)
                    iz = div(c - 1, NV * NV)
                    neigh = Int[]
                    for dz in -rCells:rCells
                        for dy in -rCells:rCells
                            for dx in -rCells:rCells
                                nx = ix + dx
                                ny = iy + dy
                                nz = iz + dz
                                if 0 <= nx < NV && 0 <= ny < NV && 0 <= nz < NV
                                    push!(neigh, nx + ny * NV + nz * NV * NV + 1)
                                end
                            end
                        end
                    end
                    neighbor_cells[c] = neigh
                end
            end
        end
    end

    tiny = 1e-12
    f_tilde = zeros(Float64, N)
    denom = zeros(Float64, N)

    # Compute density-weighted sums for use in collision operator
    if !use_cell_list
        if par.dim == 2
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                s = 0.0
                for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    pval = psi(kernel, qx, qy)
                    s += wg[k] * pval
                end
                f_tilde[i] = s
            end
            @inbounds for k in 1:N
                vxk = vx[k]
                vyk = vy[k]
                s = 0.0
                for m in 1:N
                    qx = vxk - vx[m]
                    qy = vyk - vy[m]
                    pval = psi(kernel, qx, qy)
                    s += wg[m] * pval
                end
                denom[k] = s
            end
        else
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                s = 0.0
                for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    pval = psi(kernel, qx, qy, qz)
                    s += wg[k] * pval
                end
                f_tilde[i] = s
            end
            @inbounds for k in 1:N
                vxk = vx[k]
                vyk = vy[k]
                vzk = vz[k]
                s = 0.0
                for m in 1:N
                    qx = vxk - vx[m]
                    qy = vyk - vy[m]
                    qz = vzk - vz[m]
                    pval = psi(kernel, qx, qy, qz)
                    s += wg[m] * pval
                end
                denom[k] = s
            end
        end
    else
        if par.dim == 2
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    s = 0.0
                    for nc in neighbor_cells[c]
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            s += wg[k] * psi(kernel, qx, qy)
                        end
                    end
                    f_tilde[i] = s
                end
            end
            for c in 1:Ncells
                vec_k = cell_particles[c]
                if isempty(vec_k)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for k in vec_k
                    vxk = vx[k]
                    vyk = vy[k]
                    s = 0.0
                    for nc in cells_neigh
                        for m in cell_particles[nc]
                            qx = vxk - vx[m]
                            qy = vyk - vy[m]
                            s += wg[m] * psi(kernel, qx, qy)
                        end
                    end
                    denom[k] = s
                end
            end
        else  # par.dim == 3
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    vzi = vz[i]
                    s = 0.0
                    for nc in neighbor_cells[c]
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            s += wg[k] * psi(kernel, qx, qy, qz)
                        end
                    end
                    f_tilde[i] = s
                end
            end
            for c in 1:Ncells
                vec_k = cell_particles[c]
                if isempty(vec_k)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for k in vec_k
                    vxk = vx[k]
                    vyk = vy[k]
                    vzk = vz[k]
                    s = 0.0
                    for nc in cells_neigh
                        for m in cell_particles[nc]
                            qx = vxk - vx[m]
                            qy = vyk - vy[m]
                            qz = vzk - vz[m]
                            s += wg[m] * psi(kernel, qx, qy, qz)
                        end
                    end
                    denom[k] = s
                end
            end
        end
    end

    @inbounds for i in 1:N
        if !isfinite(f_tilde[i]) || abs(f_tilde[i]) < tiny
            f_tilde[i] = tiny
        end
        if !isfinite(denom[i]) || abs(denom[i]) < tiny
            denom[i] = tiny
        end
    end

    Fx = zeros(Float64, N)
    Fy = zeros(Float64, N)
    Fz = par.dim == 3 ? zeros(Float64, N) : nothing
    # Compute advection coefficients (collision operator gradients)
    if !use_cell_list
        if par.dim == 2
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                f_tilde_i = f_tilde[i]
                A_x = 0.0
                A_y = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    gx, gy = grad(kernel, qx, qy)
                    A_x += wg[k] * gx
                    A_y += wg[k] * gy
                end
                termA_x = A_x / f_tilde_i
                termA_y = A_y / f_tilde_i

                B_x = 0.0
                B_y = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    gx, gy = grad(kernel, qx, qy)
                    B_x += wg[k] * (gx / denom[k])
                    B_y += wg[k] * (gy / denom[k])
                end
                Fx[i] = termA_x + B_x
                Fy[i] = termA_y + B_y
            end
        else
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                f_tilde_i = f_tilde[i]
                A_x = 0.0
                A_y = 0.0
                A_z = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    gx, gy, gz = grad(kernel, qx, qy, qz)
                    A_x += wg[k] * gx
                    A_y += wg[k] * gy
                    A_z += wg[k] * gz
                end
                termA_x = A_x / f_tilde_i
                termA_y = A_y / f_tilde_i
                termA_z = A_z / f_tilde_i

                B_x = 0.0
                B_y = 0.0
                B_z = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    gx, gy, gz = grad(kernel, qx, qy, qz)
                    B_x += wg[k] * (gx / denom[k])
                    B_y += wg[k] * (gy / denom[k])
                    B_z += wg[k] * (gz / denom[k])
                end
                Fx[i] = termA_x + B_x
                Fy[i] = termA_y + B_y
                Fz[i] = termA_z + B_z
            end
        end
    else
        if par.dim == 2
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    f_tilde_i = f_tilde[i]
                    A_x = 0.0
                    A_y = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            gx, gy = grad(kernel, qx, qy)
                            A_x += wg[k] * gx
                            A_y += wg[k] * gy
                        end
                    end
                    termA_x = A_x / f_tilde_i
                    termA_y = A_y / f_tilde_i

                    B_x = 0.0
                    B_y = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            gx, gy = grad(kernel, qx, qy)
                            B_x += wg[k] * (gx / denom[k])
                            B_y += wg[k] * (gy / denom[k])
                        end
                    end
                    Fx[i] = termA_x + B_x
                    Fy[i] = termA_y + B_y
                end
            end
        else  # par.dim == 3
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    vzi = vz[i]
                    f_tilde_i = f_tilde[i]
                    A_x = 0.0
                    A_y = 0.0
                    A_z = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            gx, gy, gz = grad(kernel, qx, qy, qz)
                            A_x += wg[k] * gx
                            A_y += wg[k] * gy
                            A_z += wg[k] * gz
                        end
                    end
                    termA_x = A_x / f_tilde_i
                    termA_y = A_y / f_tilde_i
                    termA_z = A_z / f_tilde_i

                    B_x = 0.0
                    B_y = 0.0
                    B_z = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            gx, gy, gz = grad(kernel, qx, qy, qz)
                            B_x += wg[k] * (gx / denom[k])
                            B_y += wg[k] * (gy / denom[k])
                            B_z += wg[k] * (gz / denom[k])
                        end
                    end
                    Fx[i] = termA_x + B_x
                    Fy[i] = termA_y + B_y
                    Fz[i] = termA_z + B_z
                end
            end
        end
    end
    # Normalize advection coefficients by particle mass
    mass = par.m
    if mass == 0.0
        @warn "A particle's mass is zero, defaulting to 1.0 for collision operator normalization."
        mass = 1.0
    end
    Fx ./= mass
    Fy ./= mass
    if par.dim == 3
        Fz ./= mass
    end

    g = par.gam
    beta = par.beta

    if Vgrid === nothing
        _, _, Vc, Vgrid = build_velocity_grid(par)
    end
    if use_lenard_balescu
        if Vgrid !== nothing && !(@isdefined Vc)
            Vc = range(-par.Lv + par.Lv/par.Nv, par.Lv - par.Lv/par.Nv, length=par.Nv)
        end
        if par.dim == 2
            Vx, Vy = component_grid_views(Vgrid, par)
            Vz = nothing
        else
            Vx, Vy, Vz = component_grid_views(Vgrid, par)
        end
    end
    
    if use_lenard_balescu && !static_dielectric
        f_grid = f_recon(vstack, wg, Vgrid, par)
        if radial_symmetry
            if par.dim == 2
                r_centers, f_binned = compute_radial_binned_f(f_grid, Vc, par)
            else
                r_centers, f_binned = compute_radial_binned_f(f_grid, Vgrid, par)
            end
            # Fit spline and get interpolation functions
            _, _, _, _, f_radial_deriv = fit_bspline(r_centers, f_binned)
            # Precompute radial distribution derivative values on radial grid
            f_radial_deriv_grid = [f_radial_deriv(r) for r in r_centers]
        else
            r_centers = nothing
            f_radial_deriv_grid = nothing
        end
    else
        r_centers = nothing
        f_radial_deriv_grid = nothing
    end

    # Precompute dielectric data once, handle both radial and generic cases
    G = nothing
    HG = nothing
    I_integral = 0.0
    k_parallel = nothing
    k_perp = nothing
    k_perp2 = nothing
    k_perp3 = nothing
    norm_k = nothing
    approx = nothing
    phi_p_11 = nothing
    phi_p_22 = nothing
    phi_p_12 = nothing
    k_angle_local = nothing
    inv_norm_k4 = nothing
    r0_ext = 0.0
    dr_ext = 1.0
    inv_dr_ext = 1.0
    len_r_ext = 0
    r_ext = nothing
    if use_lenard_balescu
        if !static_dielectric
            dielectric_grid = par.dim == 2 ? (Vx, Vy) : (Vx, Vy, Vz)
            G, HG, r_ext, I_integral = precompute_dielectric_data(
                r_centers, f_radial_deriv_grid, par.dim, par.nang;
                radial_symmetry=radial_symmetry,
                f=f_grid,
                grid=dielectric_grid,
                rho_points_3d=par.dielectric_rho_points_3d,
                psi_points_3d=par.dielectric_psi_points_3d
            )
        else
            # Static dielectric: skip G/HG precomputation; kernel uses ε_static(k)=1+α V̂(k)
            G = zeros(Float64, 1)
            HG = zeros(Float64, 1)
            I_integral = 0.0
        end
        # Precompute k-space grid and delta approximation using user parameters
        if par.dim == 2
            k_parallel, k_perp, norm_k, approx, phi_p_11, phi_p_22, phi_p_12, k_angle_local = precompute_k_space(
                par.k_perp_order, par.k_perp_max, par.k_parallel_width, par.k_parallel_order;
                k_perp_scale=par.k_perp_scale, potential=par.potential)
        else
            k_parallel, k_perp2, k_perp3, norm_k, approx, inv_norm_k4 = precompute_k_space_3d(
                par.k_perp_order, par.k_perp_max, par.k_parallel_width, par.k_parallel_order;
                k_perp_scale=par.k_perp_scale, potential=par.potential)
        end
        # Use precomputed interpolation-grid metadata passed by caller.
        r0_ext = dielectric_r0
        dr_ext = dielectric_dr
        inv_dr_ext = dielectric_inv_dr
        len_r_ext = dielectric_len
    end
    static_alpha_val = (use_lenard_balescu && static_dielectric) ? par.alpha : NaN

    # Apply collision operator with kernel computation
    if use_lenard_balescu
        if par.dim == 2
            Threads.@threads for i in 1:N
                accx = 0.0
                accy = 0.0
                vxi = vx[i]
                vyi = vy[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                @inbounds for j in 1:N
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]

                    r = sqrt(Bij_x * Bij_x + Bij_y * Bij_y)
                    if r > 1e-15
                        kernel_xx, kernel_xy, kernel_yy = compute_lenard_balescu_kernel!(
                            r, vxi, vyi, vx[j], vy[j],
                            G, HG, I_integral,
                            r0_ext, dr_ext, inv_dr_ext, len_r_ext,
                            k_parallel, k_perp, norm_k, approx, phi_p_11, phi_p_22, phi_p_12,
                            k_angle_local;
                            static_alpha=static_alpha_val,
                            constant_fourier=is_constant_fourier(par.potential)
                        )
                        axL = beta * kernel_xx
                        axR = beta * kernel_xy
                        ayL = beta * kernel_xy
                        ayR = beta * kernel_yy
                    else
                        axL, axR, ayL, ayR = 0.0, 0.0, 0.0, 0.0
                    end

                    accx += wg[j] * (axL * Bij_x + axR * Bij_y)
                    accy += wg[j] * (ayL * Bij_x + ayR * Bij_y)
                end
                @inbounds svxj[i] = accx
                @inbounds svyj[i] = accy
            end
        else
            Threads.@threads for i in 1:N
                accx = 0.0
                accy = 0.0
                accz = 0.0
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                Fi_z = Fz[i]
                @inbounds for j in 1:N
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]
                    Bij_z = Fi_z - Fz[j]

                    r = sqrt(Bij_x * Bij_x + Bij_y * Bij_y + Bij_z * Bij_z)
                    if r > 1e-15
                        a11, a12, a13,
                        a21, a22, a23,
                        a31, a32, a33 = compute_lenard_balescu_kernel_3d!(
                            r,
                            vxi, vyi, vzi, vx[j], vy[j], vz[j],
                            G, HG, I_integral,
                            r0_ext, dr_ext, inv_dr_ext, len_r_ext,
                            k_parallel, k_perp2, k_perp3,
                            norm_k, approx, inv_norm_k4;
                            static_alpha=static_alpha_val,
                            constant_fourier=is_constant_fourier(par.potential)
                        )
                        a11 *= beta; a12 *= beta; a13 *= beta
                        a21 *= beta; a22 *= beta; a23 *= beta
                        a31 *= beta; a32 *= beta; a33 *= beta
                    else
                        a11, a12, a13 = 0.0, 0.0, 0.0
                        a21, a22, a23 = 0.0, 0.0, 0.0
                        a31, a32, a33 = 0.0, 0.0, 0.0
                    end

                    accx += wg[j] * (a11 * Bij_x + a12 * Bij_y + a13 * Bij_z)
                    accy += wg[j] * (a21 * Bij_x + a22 * Bij_y + a23 * Bij_z)
                    accz += wg[j] * (a31 * Bij_x + a32 * Bij_y + a33 * Bij_z)
                end
                @inbounds svxj[i] = accx
                @inbounds svyj[i] = accy
                @inbounds svzj[i] = accz
            end
        end
    else # use simple Landau kernel
        if par.dim == 2
            Threads.@threads for i in 1:N
                accx = 0.0
                accy = 0.0
                vxi = vx[i]
                vyi = vy[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                @inbounds for j in 1:N
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]

                    axL, axR, ayL, ayR = getA_Landau(vxi, vyi, vx[j], vy[j], g, beta)
                    accx += wg[j] * (axL * Bij_x + axR * Bij_y)
                    accy += wg[j] * (ayL * Bij_x + ayR * Bij_y)
                end
                @inbounds svxj[i] = accx
                @inbounds svyj[i] = accy
            end
        else
            Threads.@threads for i in 1:N
                accx = 0.0
                accy = 0.0
                accz = 0.0
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                Fi_z = Fz[i]
                @inbounds for j in 1:N
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]
                    Bij_z = Fi_z - Fz[j]

                    a11, a12, a13,
                    a21, a22, a23,
                    a31, a32, a33 = getA_Landau_3D(vxi, vyi, vzi, vx[j], vy[j], vz[j], g, beta)

                    accx += wg[j] * (a11 * Bij_x + a12 * Bij_y + a13 * Bij_z)
                    accy += wg[j] * (a21 * Bij_x + a22 * Bij_y + a23 * Bij_z)
                    accz += wg[j] * (a31 * Bij_x + a32 * Bij_y + a33 * Bij_z)
                end
                @inbounds svxj[i] = accx
                @inbounds svyj[i] = accy
                @inbounds svzj[i] = accz
            end
        end
    end

    total_time = measure_time ? (time_ns() - tstart) / 1.0e9 : 0.0
    if elapsed_s !== nothing
        elapsed_s[] = total_time
    end
    if par.dim == 2
        return hcat(svxj, svyj)
    else
        return hcat(svxj, svyj, svzj)
    end
end

"""
    compute_collision_fields(vstack, par, wg)

Compute the global advection fields `(Fx, Fy, Fz)` used by the collision
operator. This includes the cell-list-accelerated `f_tilde/denom/F` stages and
is intended to be evaluated on the full particle set.
"""
function compute_collision_fields(vstack::AbstractMatrix{Float64},
                                  par::Params,
                                  wg::AbstractVector{Float64})
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    N, dim = size(vstack)
    @assert dim == par.dim "vstack second dimension must match par.dim"
    @assert length(wg) == N "Expected wg length to match number of particles"

    Fx = zeros(Float64, N)
    Fy = zeros(Float64, N)
    Fz = par.dim == 3 ? zeros(Float64, N) : nothing
    if N == 0
        return (Fx=Fx, Fy=Fy, Fz=Fz)
    end

    vx = @view vstack[:, 1]
    vy = @view vstack[:, 2]
    vz = par.dim == 3 ? (@view vstack[:, 3]) : nothing

    NV = par.Nv
    Lv = par.Lv
    Ncells = NV^par.dim
    cell_h = (par.cell_h > 0.0) ? par.cell_h : (2.0 * Lv / NV)
    epsi = par.epsi

    order = par.spline_order
    kernel = BasisKernel(order, epsi)

    use_cell_list = par.use_cell_list
    if use_cell_list
        # Build spatial cell list
        cell_particles = [Int[] for _ in 1:Ncells]
        if par.dim == 2
            @inbounds for p in 1:N
                xnorm = (vx[p] + Lv) / (2.0 * Lv)
                ynorm = (vy[p] + Lv) / (2.0 * Lv)
                ix = clamp(floor(Int, xnorm * NV), 0, NV - 1)
                iy = clamp(floor(Int, ynorm * NV), 0, NV - 1)
                c = ix + iy * NV + 1
                push!(cell_particles[c], p)
            end
        else
            @inbounds for p in 1:N
                xnorm = (vx[p] + Lv) / (2.0 * Lv)
                ynorm = (vy[p] + Lv) / (2.0 * Lv)
                znorm = (vz[p] + Lv) / (2.0 * Lv)
                ix = clamp(floor(Int, xnorm * NV), 0, NV - 1)
                iy = clamp(floor(Int, ynorm * NV), 0, NV - 1)
                iz = clamp(floor(Int, znorm * NV), 0, NV - 1)
                c = ix + iy * NV + iz * NV * NV + 1
                push!(cell_particles[c], p)
            end
        end

        rCells = neighbor_radius_cells(kernel, cell_h; gaussian_cutoff_sigma=par.gaussian_cutoff_sigma)
        if rCells >= NV
            use_cell_list = false
        else
            neighbor_cells = [Int[] for _ in 1:Ncells]
            if par.dim == 2
                for c in 1:Ncells
                    ix = (c - 1) % NV
                    iy = (c - 1) ÷ NV
                    neigh = Int[]
                    for dy in -rCells:rCells
                        for dx in -rCells:rCells
                            nx = ix + dx
                            ny = iy + dy
                            if 0 <= nx < NV && 0 <= ny < NV
                                push!(neigh, nx + ny * NV + 1)
                            end
                        end
                    end
                    neighbor_cells[c] = neigh
                end
            else
                for c in 1:Ncells
                    ix = mod(c - 1, NV)
                    iy = div(mod(c - 1, NV * NV), NV)
                    iz = div(c - 1, NV * NV)
                    neigh = Int[]
                    for dz in -rCells:rCells
                        for dy in -rCells:rCells
                            for dx in -rCells:rCells
                                nx = ix + dx
                                ny = iy + dy
                                nz = iz + dz
                                if 0 <= nx < NV && 0 <= ny < NV && 0 <= nz < NV
                                    push!(neigh, nx + ny * NV + nz * NV * NV + 1)
                                end
                            end
                        end
                    end
                    neighbor_cells[c] = neigh
                end
            end
        end
    end

    tiny = 1e-12
    f_tilde = zeros(Float64, N)
    denom = zeros(Float64, N)

    if !use_cell_list
        if par.dim == 2
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                s = 0.0
                for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    pval = psi(kernel, qx, qy)
                    s += wg[k] * pval
                end
                f_tilde[i] = s
            end
            @inbounds for k in 1:N
                vxk = vx[k]
                vyk = vy[k]
                s = 0.0
                for m in 1:N
                    qx = vxk - vx[m]
                    qy = vyk - vy[m]
                    pval = psi(kernel, qx, qy)
                    s += wg[m] * pval
                end
                denom[k] = s
            end
        else
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                s = 0.0
                for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    pval = psi(kernel, qx, qy, qz)
                    s += wg[k] * pval
                end
                f_tilde[i] = s
            end
            @inbounds for k in 1:N
                vxk = vx[k]
                vyk = vy[k]
                vzk = vz[k]
                s = 0.0
                for m in 1:N
                    qx = vxk - vx[m]
                    qy = vyk - vy[m]
                    qz = vzk - vz[m]
                    pval = psi(kernel, qx, qy, qz)
                    s += wg[m] * pval
                end
                denom[k] = s
            end
        end
    else
        if par.dim == 2
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    s = 0.0
                    for nc in neighbor_cells[c]
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            s += wg[k] * psi(kernel, qx, qy)
                        end
                    end
                    f_tilde[i] = s
                end
            end
            for c in 1:Ncells
                vec_k = cell_particles[c]
                if isempty(vec_k)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for k in vec_k
                    vxk = vx[k]
                    vyk = vy[k]
                    s = 0.0
                    for nc in cells_neigh
                        for m in cell_particles[nc]
                            qx = vxk - vx[m]
                            qy = vyk - vy[m]
                            s += wg[m] * psi(kernel, qx, qy)
                        end
                    end
                    denom[k] = s
                end
            end
        else
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    vzi = vz[i]
                    s = 0.0
                    for nc in neighbor_cells[c]
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            s += wg[k] * psi(kernel, qx, qy, qz)
                        end
                    end
                    f_tilde[i] = s
                end
            end
            for c in 1:Ncells
                vec_k = cell_particles[c]
                if isempty(vec_k)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for k in vec_k
                    vxk = vx[k]
                    vyk = vy[k]
                    vzk = vz[k]
                    s = 0.0
                    for nc in cells_neigh
                        for m in cell_particles[nc]
                            qx = vxk - vx[m]
                            qy = vyk - vy[m]
                            qz = vzk - vz[m]
                            s += wg[m] * psi(kernel, qx, qy, qz)
                        end
                    end
                    denom[k] = s
                end
            end
        end
    end

    @inbounds for i in 1:N
        if !isfinite(f_tilde[i]) || abs(f_tilde[i]) < tiny
            f_tilde[i] = tiny
        end
        if !isfinite(denom[i]) || abs(denom[i]) < tiny
            denom[i] = tiny
        end
    end

    if !use_cell_list
        if par.dim == 2
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                f_tilde_i = f_tilde[i]
                A_x = 0.0
                A_y = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    gx, gy = grad(kernel, qx, qy)
                    A_x += wg[k] * gx
                    A_y += wg[k] * gy
                end
                termA_x = A_x / f_tilde_i
                termA_y = A_y / f_tilde_i

                B_x = 0.0
                B_y = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    gx, gy = grad(kernel, qx, qy)
                    B_x += wg[k] * (gx / denom[k])
                    B_y += wg[k] * (gy / denom[k])
                end
                Fx[i] = termA_x + B_x
                Fy[i] = termA_y + B_y
            end
        else
            @inbounds for i in 1:N
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                f_tilde_i = f_tilde[i]
                A_x = 0.0
                A_y = 0.0
                A_z = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    gx, gy, gz = grad(kernel, qx, qy, qz)
                    A_x += wg[k] * gx
                    A_y += wg[k] * gy
                    A_z += wg[k] * gz
                end
                termA_x = A_x / f_tilde_i
                termA_y = A_y / f_tilde_i
                termA_z = A_z / f_tilde_i

                B_x = 0.0
                B_y = 0.0
                B_z = 0.0
                @inbounds for k in 1:N
                    qx = vxi - vx[k]
                    qy = vyi - vy[k]
                    qz = vzi - vz[k]
                    gx, gy, gz = grad(kernel, qx, qy, qz)
                    B_x += wg[k] * (gx / denom[k])
                    B_y += wg[k] * (gy / denom[k])
                    B_z += wg[k] * (gz / denom[k])
                end
                Fx[i] = termA_x + B_x
                Fy[i] = termA_y + B_y
                Fz[i] = termA_z + B_z
            end
        end
    else
        if par.dim == 2
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    f_tilde_i = f_tilde[i]
                    A_x = 0.0
                    A_y = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            gx, gy = grad(kernel, qx, qy)
                            A_x += wg[k] * gx
                            A_y += wg[k] * gy
                        end
                    end
                    termA_x = A_x / f_tilde_i
                    termA_y = A_y / f_tilde_i

                    B_x = 0.0
                    B_y = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            gx, gy = grad(kernel, qx, qy)
                            B_x += wg[k] * (gx / denom[k])
                            B_y += wg[k] * (gy / denom[k])
                        end
                    end
                    Fx[i] = termA_x + B_x
                    Fy[i] = termA_y + B_y
                end
            end
        else
            for c in 1:Ncells
                vec_i = cell_particles[c]
                if isempty(vec_i)
                    continue
                end
                cells_neigh = neighbor_cells[c]
                for i in vec_i
                    vxi = vx[i]
                    vyi = vy[i]
                    vzi = vz[i]
                    f_tilde_i = f_tilde[i]
                    A_x = 0.0
                    A_y = 0.0
                    A_z = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            gx, gy, gz = grad(kernel, qx, qy, qz)
                            A_x += wg[k] * gx
                            A_y += wg[k] * gy
                            A_z += wg[k] * gz
                        end
                    end
                    termA_x = A_x / f_tilde_i
                    termA_y = A_y / f_tilde_i
                    termA_z = A_z / f_tilde_i

                    B_x = 0.0
                    B_y = 0.0
                    B_z = 0.0
                    for nc in cells_neigh
                        for k in cell_particles[nc]
                            qx = vxi - vx[k]
                            qy = vyi - vy[k]
                            qz = vzi - vz[k]
                            gx, gy, gz = grad(kernel, qx, qy, qz)
                            B_x += wg[k] * (gx / denom[k])
                            B_y += wg[k] * (gy / denom[k])
                            B_z += wg[k] * (gz / denom[k])
                        end
                    end
                    Fx[i] = termA_x + B_x
                    Fy[i] = termA_y + B_y
                    Fz[i] = termA_z + B_z
                end
            end
        end
    end

    mass = par.m
    if mass == 0.0
        @warn "A particle's mass is zero, defaulting to 1.0 for collision operator normalization."
        mass = 1.0
    end
    Fx ./= mass
    Fy ./= mass
    if par.dim == 3
        Fz ./= mass
    end

    return (Fx=Fx, Fy=Fy, Fz=Fz)
end

"""
    precompute_collision_kernel_state(vstack, par, wg;
                                      Vgrid, use_lenard_balescu,
                                      radial_symmetry, static_dielectric,
                                      dielectric_r0, dielectric_dr,
                                      dielectric_inv_dr, dielectric_len)

Precompute data reused by pairwise collision evaluation. For Landau mode this
returns `nothing`; for Lenard-Balescu it returns a NamedTuple of dielectric and
k-space precomputations. When `static_dielectric=true`, G/HG computation is skipped
and `static_alpha = par.alpha` is embedded in the state for the static ε_static(k) path.
"""
function precompute_collision_kernel_state(vstack::AbstractMatrix{Float64},
                                           par::Params,
                                           wg::AbstractVector{Float64};
                                           Vgrid::AbstractArray{Float64},
                                           use_lenard_balescu::Bool=true,
                                           radial_symmetry::Bool=true,
                                           static_dielectric::Bool=false,
                                           dielectric_r0::Float64,
                                           dielectric_dr::Float64,
                                           dielectric_inv_dr::Float64,
                                           dielectric_len::Int)
    if !use_lenard_balescu
        return nothing
    end

    static_alpha_val = static_dielectric ? par.alpha : NaN

    Nv = par.Nv
    Vc = range(-par.Lv + par.Lv / Nv, par.Lv - par.Lv / Nv, length=Nv)
    if par.dim == 2
        Vx, Vy = component_grid_views(Vgrid, par)
        Vz = nothing
    else
        Vx, Vy, Vz = component_grid_views(Vgrid, par)
    end

    if !static_dielectric
        f_grid = f_recon(vstack, wg, Vgrid, par)
        if radial_symmetry
            if par.dim == 2
                r_centers, f_binned = compute_radial_binned_f(f_grid, Vc, par)
            else
                r_centers, f_binned = compute_radial_binned_f(f_grid, Vgrid, par)
            end
            _, _, _, _, f_radial_deriv = fit_bspline(r_centers, f_binned)
            f_radial_deriv_grid = [f_radial_deriv(r) for r in r_centers]
        else
            r_centers = nothing
            f_radial_deriv_grid = nothing
        end

        dielectric_grid = par.dim == 2 ? (Vx, Vy) : (Vx, Vy, Vz)
        G, HG, _, I_integral = precompute_dielectric_data(
            r_centers, f_radial_deriv_grid, par.dim, par.nang;
            radial_symmetry=radial_symmetry,
            f=f_grid,
            grid=dielectric_grid,
            rho_points_3d=par.dielectric_rho_points_3d,
            psi_points_3d=par.dielectric_psi_points_3d
        )
    else
        # Static dielectric: skip G/HG; kernel uses ε_static(k)=1+αV̂(k)
        G = zeros(Float64, 1)
        HG = zeros(Float64, 1)
        I_integral = 0.0
    end
    if par.dim == 2
        k_parallel, k_perp, norm_k, approx, phi_p_11, phi_p_22, phi_p_12, k_angle_local = precompute_k_space(
            par.k_perp_order, par.k_perp_max, par.k_parallel_width, par.k_parallel_order;
            k_perp_scale=par.k_perp_scale, potential=par.potential
        )
        return (
            G=G,
            HG=HG,
            I_integral=I_integral,
            k_parallel=k_parallel,
            k_perp=k_perp,
            norm_k=norm_k,
            approx=approx,
            phi_p_11=phi_p_11,
            phi_p_22=phi_p_22,
            phi_p_12=phi_p_12,
            k_angle_local=k_angle_local,
            r0_ext=dielectric_r0,
            dr_ext=dielectric_dr,
            inv_dr_ext=dielectric_inv_dr,
            len_r_ext=dielectric_len,
            static_alpha=static_alpha_val,
            constant_fourier=is_constant_fourier(par.potential)
        )
    else
        k_parallel, k_perp2, k_perp3, norm_k, approx, inv_norm_k4 = precompute_k_space_3d(
            par.k_perp_order, par.k_perp_max, par.k_parallel_width, par.k_parallel_order;
            k_perp_scale=par.k_perp_scale, potential=par.potential
        )
        return (
            G=G,
            HG=HG,
            I_integral=I_integral,
            k_parallel=k_parallel,
            k_perp2=k_perp2,
            k_perp3=k_perp3,
            norm_k=norm_k,
            approx=approx,
            inv_norm_k4=inv_norm_k4,
            r0_ext=dielectric_r0,
            dr_ext=dielectric_dr,
            inv_dr_ext=dielectric_inv_dr,
            len_r_ext=dielectric_len,
            static_alpha=static_alpha_val,
            constant_fourier=is_constant_fourier(par.potential)
        )
    end
end

"""
    collision_pairwise_subset(vstack, par, wg, Fx, Fy, Fz, idx_i, idx_j;
                              use_lenard_balescu=true,
                              kernel_state=nothing)

Evaluate only the pairwise interaction stage for particle indices `idx_i`
against `idx_j` using precomputed global fields.
"""
function collision_pairwise_subset(vstack::AbstractMatrix{Float64},
                                   par::Params,
                                   wg::AbstractVector{Float64},
                                   Fx::AbstractVector{Float64},
                                   Fy::AbstractVector{Float64},
                                   Fz::Union{Nothing, AbstractVector{Float64}},
                                   idx_i::AbstractVector{Int},
                                   idx_j::AbstractVector{Int};
                                   use_lenard_balescu::Bool=true,
                                   kernel_state=nothing)
    N, dim = size(vstack)
    @assert dim == par.dim "vstack second dimension must match par.dim"
    @assert length(wg) == N "Expected wg length to match number of particles"
    @assert length(Fx) == N && length(Fy) == N "Field lengths must match particle count"
    if par.dim == 3
        @assert !isnothing(Fz) && length(Fz) == N "Fz must be provided for 3D"
    end

    vx = @view vstack[:, 1]
    vy = @view vstack[:, 2]
    vz = par.dim == 3 ? (@view vstack[:, 3]) : nothing

    m_i = length(idx_i)
    svx = zeros(Float64, m_i)
    svy = zeros(Float64, m_i)
    svz = par.dim == 3 ? zeros(Float64, m_i) : nothing

    g = par.gam
    beta = par.beta

    if use_lenard_balescu
        @assert kernel_state !== nothing "kernel_state is required for Lenard-Balescu pairwise stage"
    end

    if use_lenard_balescu
        if par.dim == 2
            Threads.@threads for ii in 1:m_i
                i = idx_i[ii]
                accx = 0.0
                accy = 0.0
                vxi = vx[i]
                vyi = vy[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                @inbounds for j in idx_j
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]

                    r = sqrt(Bij_x * Bij_x + Bij_y * Bij_y)
                    if r > 1e-15
                        kernel_xx, kernel_xy, kernel_yy = compute_lenard_balescu_kernel!(
                            r, vxi, vyi, vx[j], vy[j],
                            kernel_state.G, kernel_state.HG, kernel_state.I_integral,
                            kernel_state.r0_ext, kernel_state.dr_ext,
                            kernel_state.inv_dr_ext, kernel_state.len_r_ext,
                            kernel_state.k_parallel, kernel_state.k_perp,
                            kernel_state.norm_k, kernel_state.approx,
                            kernel_state.phi_p_11, kernel_state.phi_p_22,
                            kernel_state.phi_p_12, kernel_state.k_angle_local;
                            static_alpha=kernel_state.static_alpha,
                            constant_fourier=kernel_state.constant_fourier
                        )
                        axL = beta * kernel_xx
                        axR = beta * kernel_xy
                        ayL = beta * kernel_xy
                        ayR = beta * kernel_yy
                    else
                        axL, axR, ayL, ayR = 0.0, 0.0, 0.0, 0.0
                    end

                    accx += wg[j] * (axL * Bij_x + axR * Bij_y)
                    accy += wg[j] * (ayL * Bij_x + ayR * Bij_y)
                end
                @inbounds svx[ii] = accx
                @inbounds svy[ii] = accy
            end
        else
            Threads.@threads for ii in 1:m_i
                i = idx_i[ii]
                accx = 0.0
                accy = 0.0
                accz = 0.0
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                Fi_z = Fz[i]
                @inbounds for j in idx_j
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]
                    Bij_z = Fi_z - Fz[j]

                    r = sqrt(Bij_x * Bij_x + Bij_y * Bij_y + Bij_z * Bij_z)
                    if r > 1e-15
                        a11, a12, a13,
                        a21, a22, a23,
                        a31, a32, a33 = compute_lenard_balescu_kernel_3d!(
                            r,
                            vxi, vyi, vzi, vx[j], vy[j], vz[j],
                            kernel_state.G, kernel_state.HG, kernel_state.I_integral,
                            kernel_state.r0_ext, kernel_state.dr_ext,
                            kernel_state.inv_dr_ext, kernel_state.len_r_ext,
                            kernel_state.k_parallel, kernel_state.k_perp2,
                            kernel_state.k_perp3,
                            kernel_state.norm_k, kernel_state.approx,
                            kernel_state.inv_norm_k4;
                            static_alpha=kernel_state.static_alpha,
                            constant_fourier=kernel_state.constant_fourier
                        )
                        a11 *= beta; a12 *= beta; a13 *= beta
                        a21 *= beta; a22 *= beta; a23 *= beta
                        a31 *= beta; a32 *= beta; a33 *= beta
                    else
                        a11, a12, a13 = 0.0, 0.0, 0.0
                        a21, a22, a23 = 0.0, 0.0, 0.0
                        a31, a32, a33 = 0.0, 0.0, 0.0
                    end

                    accx += wg[j] * (a11 * Bij_x + a12 * Bij_y + a13 * Bij_z)
                    accy += wg[j] * (a21 * Bij_x + a22 * Bij_y + a23 * Bij_z)
                    accz += wg[j] * (a31 * Bij_x + a32 * Bij_y + a33 * Bij_z)
                end
                @inbounds svx[ii] = accx
                @inbounds svy[ii] = accy
                @inbounds svz[ii] = accz
            end
        end
    else
        if par.dim == 2
            Threads.@threads for ii in 1:m_i
                i = idx_i[ii]
                accx = 0.0
                accy = 0.0
                vxi = vx[i]
                vyi = vy[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                @inbounds for j in idx_j
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]
                    axL, axR, ayL, ayR = getA_Landau(vxi, vyi, vx[j], vy[j], g, beta)
                    accx += wg[j] * (axL * Bij_x + axR * Bij_y)
                    accy += wg[j] * (ayL * Bij_x + ayR * Bij_y)
                end
                @inbounds svx[ii] = accx
                @inbounds svy[ii] = accy
            end
        else
            Threads.@threads for ii in 1:m_i
                i = idx_i[ii]
                accx = 0.0
                accy = 0.0
                accz = 0.0
                vxi = vx[i]
                vyi = vy[i]
                vzi = vz[i]
                Fi_x = Fx[i]
                Fi_y = Fy[i]
                Fi_z = Fz[i]
                @inbounds for j in idx_j
                    Bij_x = Fi_x - Fx[j]
                    Bij_y = Fi_y - Fy[j]
                    Bij_z = Fi_z - Fz[j]

                    a11, a12, a13,
                    a21, a22, a23,
                    a31, a32, a33 = getA_Landau_3D(vxi, vyi, vzi, vx[j], vy[j], vz[j], g, beta)

                    accx += wg[j] * (a11 * Bij_x + a12 * Bij_y + a13 * Bij_z)
                    accy += wg[j] * (a21 * Bij_x + a22 * Bij_y + a23 * Bij_z)
                    accz += wg[j] * (a31 * Bij_x + a32 * Bij_y + a33 * Bij_z)
                end
                @inbounds svx[ii] = accx
                @inbounds svy[ii] = accy
                @inbounds svz[ii] = accz
            end
        end
    end

    if par.dim == 2
        return hcat(svx, svy)
    else
        return hcat(svx, svy, svz)
    end
end

"""
    precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=true)

Precompute fixed interpolation-grid metadata used by fused Lenard-Balescu
kernels: `(r0, dr, inv_dr, len)`.

This depends on grid geometry (`par`, `Vgrid`, `radial_symmetry`) and can be
computed once per simulation run.
"""
function precompute_dielectric_lookup_grid(par::Params, Vgrid::AbstractArray{Float64};
                                           radial_symmetry::Bool=true)
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    grid_views = component_grid_views(Vgrid, par)

    if radial_symmetry
        r_bins = radial_bin_edges(par)
        n_bins = length(r_bins) - 1
        r_centers = (r_bins[1:end-1] .+ r_bins[2:end]) ./ 2.0
        r_abs_max = maximum(r_centers)
        n_plot = 2 * n_bins
        r_plot = range(-r_abs_max, r_abs_max, length=n_plot)

        L = maximum(r_plot)
        n_orig = length(r_plot)
        n_ext = 2 * n_orig - 1
        r0 = -L
        if n_ext > 1
            dr = (2.0 * L) / (n_ext - 1)
            inv_dr = 1.0 / dr
        else
            dr = 1.0
            inv_dr = 1.0
        end
        return r0, dr, inv_dr, n_ext
    end

    Nv = par.Nv
    if par.dim == 2
        vx_grid, vy_grid = grid_views
        r_extent = maximum(sqrt.(vx_grid.^2 .+ vy_grid.^2))
        xuniq = sort!(unique(vx_grid))
        yuniq = sort!(unique(vy_grid))
        @assert length(xuniq) >= 2 && length(yuniq) >= 2 "Non-radial lookup requires at least 2 unique points per axis"
        dv_axis = min(xuniq[2] - xuniq[1], yuniq[2] - yuniq[1])
    else
        vx_grid, vy_grid, vz_grid = grid_views
        r_extent = maximum(sqrt.(vx_grid.^2 .+ vy_grid.^2 .+ vz_grid.^2))
        xuniq = sort!(unique(vx_grid))
        yuniq = sort!(unique(vy_grid))
        zuniq = sort!(unique(vz_grid))
        @assert length(xuniq) >= 2 && length(yuniq) >= 2 && length(zuniq) >= 2 "Non-radial lookup requires at least 2 unique points per axis"
        dv_axis = min(xuniq[2] - xuniq[1], min(yuniq[2] - yuniq[1], zuniq[2] - zuniq[1]))
    end
    @assert isfinite(dv_axis) && dv_axis > 0.0 "Invalid non-radial lookup spacing: Δv=$(dv_axis)"
    n_parallel = max(Nv, Int(ceil((2.0 * r_extent) / max(dv_axis, 1e-12))) + 1)
    if iseven(n_parallel)
        n_parallel += 1
    end
    x_centers = collect(range(-r_extent, r_extent; length=n_parallel))

    r0 = x_centers[1]
    len_r = length(x_centers)
    if len_r > 1
        dr = x_centers[2] - x_centers[1]
        inv_dr = 1.0 / dr
    else
        dr = 1.0
        inv_dr = 1.0
    end
    return r0, dr, inv_dr, len_r
end
