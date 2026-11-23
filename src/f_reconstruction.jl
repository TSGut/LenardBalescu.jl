"""
    f_recon(vstack, wg, Vgrid, par)

Reconstructs the distribution function on a velocity grid from particle positions and weights.

# Arguments
- `vstack`: particle-velocity matrix (one particle per row; columns follow `par.dim`)
- `wg`: Particle weights
- `Vgrid`: tensor grid of velocity-cell centres, with component index in the last dimension
- `par`: Simulation parameters

# Returns
- 1D array of reconstructed distribution values on the grid
"""
function f_recon(vstack::AbstractMatrix{Float64}, wg::AbstractVector{Float64},
                 Vgrid::AbstractArray{Float64}, par::Params)
    N = size(vstack, 1)
    @assert size(vstack, 2) == par.dim "vstack second dimension must match par.dim"
    grid_views = component_grid_views(Vgrid, par)
    M = length(grid_views[1])
    f = zeros(Float64, M)

    order = par.spline_order
    kernel = BasisKernel(order, par.epsi)

    # For each grid point, compute weighted sum of basis functions centered on particles
    if par.dim == 2
        vx_grid, vy_grid = grid_views
        Threads.@threads for m in 1:M
            vxg = vx_grid[m]
            vyg = vy_grid[m]
            s = 0.0
            @inbounds for i in 1:N
                qx = vxg - vstack[i, 1]
                qy = vyg - vstack[i, 2]
                s += wg[i] * psi(kernel, qx, qy)
            end
            f[m] = s
        end
    else
        vx_grid, vy_grid, vz_grid = grid_views
        Threads.@threads for m in 1:M
            vxg = vx_grid[m]
            vyg = vy_grid[m]
            vzg = vz_grid[m]
            s = 0.0
            @inbounds for i in 1:N
                qx = vxg - vstack[i, 1]
                qy = vyg - vstack[i, 2]
                qz = vzg - vstack[i, 3]
                s += wg[i] * psi(kernel, qx, qy, qz)
            end
            f[m] = s
        end
    end

    return f
end

function radial_bin_edges(par::Params; n_bins::Union{Nothing,Int}=nothing)
    r_max = par.Lv * sqrt(par.dim)
    if isnothing(n_bins)
        h = 2.0 * par.Lv / par.Nv
        n = ceil(Int, r_max / h)
        return range(0.0; step=h, length=n + 1)
    end
    @assert n_bins > 0 "n_bins must be positive"
    return range(0.0, r_max; length=n_bins + 1)
end

"""
    compute_radial_binned_f(f_grid::AbstractVector, Vc::AbstractVector, par::Params; n_bins=nothing)

Compatibility 2D wrapper for radial binning from 1D center coordinates `Vc`.
This method is defined for `par.dim == 2` and forwards to the
`(f_grid, Vgrid, par)` method after constructing a 2D `Vgrid`.

Returns the same symmetric radial grid and averaged values as the general method.
"""
function compute_radial_binned_f(f_grid::AbstractVector, Vc::AbstractVector, par::Params; n_bins::Union{Nothing,Int}=nothing)
    @assert par.dim == 2 "compute_radial_binned_f(f_grid, Vc, par) is only defined for dim=2"
    @assert length(Vc) == par.Nv "Vc length must equal par.Nv"

    # Build 2D grid centers in the same ordering used by build_velocity_grid
    Vgrid = zeros(Float64, par.Nv, par.Nv, 2)
    for j in 1:par.Nv
        for i in 1:par.Nv
            Vgrid[i, j, 1] = Vc[j]
            Vgrid[i, j, 2] = Vc[i]
        end
    end
    return compute_radial_binned_f(f_grid, Vgrid, par; n_bins=n_bins)
end

"""
    compute_radial_binned_f(f_grid::AbstractVector, Vgrid::AbstractArray, par::Params; n_bins=nothing)

Compute radially binned 1D distribution f(r) from a 2D or 3D velocity grid representation.
Uses unsigned radius r = ||v|| and averages f within radial shells.
By default, shells have width h = 2Lv/Nv and cover the velocity domain.
An explicit `n_bins` instead divides [0, Lv*sqrt(dim)] into that many shells.
Empty shells are assigned zero.

Returns `(r_plot, f_plot)` on a symmetric axis through the shell centres.
"""
function compute_radial_binned_f(f_grid::AbstractVector, Vgrid::AbstractArray{Float64}, par::Params; n_bins::Union{Nothing,Int}=nothing)
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    grid_views = component_grid_views(Vgrid, par)
    @assert length(f_grid) == length(grid_views[1]) "f_grid length must match number of Vgrid points"

    # Compute unsigned radial distance for each grid point
    r_bins = radial_bin_edges(par; n_bins=n_bins)
    n_bins = length(r_bins) - 1
    r_centers = (r_bins[1:end-1] .+ r_bins[2:end]) ./ 2
    f_binned = fill(0.0, n_bins)
    counts = fill(0, n_bins)

    if par.dim == 2
        vx_grid, vy_grid = grid_views
        @inbounds for m in eachindex(vx_grid)
            r = sqrt(vx_grid[m]^2 + vy_grid[m]^2)
            bin_idx = searchsortedlast(r_bins, r)
            if 1 <= bin_idx <= n_bins
                f_binned[bin_idx] += f_grid[m]
                counts[bin_idx] += 1
            end
        end
    else
        vx_grid, vy_grid, vz_grid = grid_views
        @inbounds for m in eachindex(vx_grid)
            r = sqrt(vx_grid[m]^2 + vy_grid[m]^2 + vz_grid[m]^2)
            bin_idx = searchsortedlast(r_bins, r)
            if 1 <= bin_idx <= n_bins
                f_binned[bin_idx] += f_grid[m]
                counts[bin_idx] += 1
            end
        end
    end

    # Average within bins
    for b in 1:n_bins
        if counts[b] > 0
            f_binned[b] /= counts[b]
        end
    end

    # Create uniform grid from -r_max to r_max
    r_abs_max = maximum(r_centers)
    n_plot = 2 * n_bins
    r_plot = range(-r_abs_max, r_abs_max, length=n_plot)

    # Map binned data to full range using symmetry
    f_plot = zeros(n_plot)
    for i in 1:n_plot
        r_abs = abs(r_plot[i])
        # Find nearest bins
        idx = searchsortedlast(r_centers, r_abs)
        if idx == 0
            f_plot[i] = f_binned[1]
        elseif idx == length(r_centers)
            f_plot[i] = f_binned[end]
        else
            # Linear interpolation
            t = (r_abs - r_centers[idx]) / (r_centers[idx+1] - r_centers[idx])
            f_plot[i] = f_binned[idx] * (1-t) + f_binned[idx+1] * t
        end
    end

    return r_plot, f_plot
end
