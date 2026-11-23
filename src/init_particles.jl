function initialize_particles_stratified_bumptail(N::Int, dV::Float64,
                                                   Vgrid::AbstractArray{Float64},
                                                   par::Params,
                                                   rng::AbstractRNG=Random.default_rng())

    # ======================================================================
    # HARD-CODED BUMP-ON-TAIL PARAMETERS (EDIT THESE)
    bump_frac  = 0.05                     # Fraction of bump particles
    bump_temp  = 0.3 * par.temp           # Bump temperature (σ_b^2)
    bump_center = par.dim == 2 ?
        [3.0, 0.0] :                      # Drift velocity (2D)
        [3.0, 0.0, 0.0]                   # Drift velocity (3D)
    # ======================================================================

    grid_views = component_grid_views(Vgrid, par)
    M = length(grid_views[1])
    if M <= 0
        error("initialize_particles_stratified_bumptail: Vgrid is empty")
    end

    dim = par.dim
    T  = par.temp
    Tb = bump_temp

    # Squared speeds for core and bump
    r2_core = Vector{Float64}(undef, M)
    r2_bump = Vector{Float64}(undef, M)

    if dim == 2
        vx, vy = grid_views
        @inbounds for k in eachindex(vx)
            r2_core[k] = vx[k]^2 + vy[k]^2
            r2_bump[k] =
                (vx[k] - bump_center[1])^2 +
                (vy[k] - bump_center[2])^2
        end
    else
        vx, vy, vz = grid_views
        @inbounds for k in eachindex(vx)
            r2_core[k] = vx[k]^2 + vy[k]^2 + vz[k]^2
            r2_bump[k] =
                (vx[k] - bump_center[1])^2 +
                (vy[k] - bump_center[2])^2 +
                (vz[k] - bump_center[3])^2
        end
    end

    # --- Properly normalized Maxwellians ---
    core_norm = (2π * T)^(-dim / 2)
    bump_norm = (2π * Tb)^(-dim / 2)

    f_core = core_norm .* exp.(-r2_core ./ (2 * T))
    f_bump = bump_norm .* exp.(-r2_bump ./ (2 * Tb))

    # Mixture
    f_vals = (1 - bump_frac) .* f_core .+ bump_frac .* f_bump

    # Convert density to particle counts
    mass = f_vals .* dV^dim
    n_frac = mass .* N
    n_int = floor.(n_frac)
    remainder = n_frac .- n_int

    sum_int = Int(round(sum(n_int)))
    deficit = max(N - sum_int, 0)

    # Distribute remaining particles deterministically
    rems = sort!([(remainder[k], k) for k in 1:M],
                 by = x -> x[1], rev = true)
    for i in 1:min(deficit, M)
        _, k = rems[i]
        n_int[k] += 1.0
    end

    # Correct the total particle count.
    total_assigned = Int(round(sum(n_int)))
    if total_assigned != N
        diff = N - total_assigned
        k = 1
        while diff != 0
            if diff > 0
                n_int[k] += 1.0
                diff -= 1
            elseif n_int[k] > 0
                n_int[k] -= 1.0
                diff += 1
            end
            k = k % M + 1
        end
    end

    # Sample uniformly inside each occupied cell
    vstack = zeros(Float64, N, dim)
    count = 0

    for k in 1:M
        ns = Int(round(n_int[k]))
        if ns <= 0
            continue
        end
        for _ in 1:ns
            count += 1
            count > N && break
            for d in 1:dim
                vstack[count, d] =
                    grid_views[d][k] - dV/2 + rand(rng) * dV
            end
        end
    end

    # Fill any remaining slots uniformly over the velocity domain.
    while count < N
        count += 1
        for d in 1:dim
            vstack[count, d] =
                -par.Lv + rand(rng) * (2 * par.Lv)
        end
    end

    # Uniform weights
    wg = fill(sum(mass) / N, N)

    return vstack, wg
end

"""
    initialize_particles_maxwellian(N, par, rng)

Initialize `N` particles with velocities sampled independently from a Maxwellian distribution with temperature `par.temp`.

# Arguments
- `N`: Total number of particles
- `par`: Simulation parameters (must contain `temp` and `dim`)
- `rng`: (optional) random number generator

# Returns
- `(vstack, wg)` where `vstack` is `N×2` for `par.dim == 2` and `N×3` for `par.dim == 3`
"""
function initialize_particles_maxwellian(N::Int, par::Params, rng::AbstractRNG=Random.default_rng())
    sigma = sqrt(par.temp)
    wg = fill(1.0 / N, N)
    if par.dim == 2
        vx = randn(rng, N) .* sigma
        vy = randn(rng, N) .* sigma
        vstack = hcat(vx, vy)
        return vstack, wg
    else # 3D
        vx = randn(rng, N) .* sigma
        vy = randn(rng, N) .* sigma
        vz = randn(rng, N) .* sigma
        vstack = hcat(vx, vy, vz)
        return vstack, wg
    end
end

"""
    initialize_particles_torus_3d(N, par, rng; major_radius=2.0, tube_radius=0.35)

Initialize `N` particles in 3D velocity space on a solid torus centered on the z-axis,
lying in the xy-plane. The torus major radius is `major_radius` and minor radius is
`tube_radius`.

# Arguments
- `N`: Total number of particles
- `par`: Simulation parameters (must have `dim == 3`)
- `rng`: Random number generator
- `major_radius`: Distance from origin to the centerline circle in the xy-plane
- `tube_radius`: Radius of the torus tube

# Returns
- `(vstack, wg)` with `vstack` shaped `N×3` and uniform particle weights
"""
function initialize_particles_torus_3d(N::Int, par::Params, rng::AbstractRNG=Random.default_rng();
                                       major_radius::Float64=2.0,
                                       tube_radius::Float64=0.35)
    @assert par.dim == 3 "initialize_particles_torus_3d requires par.dim == 3"
    @assert N > 0 "N must be positive"
    @assert major_radius > 0.0 "major_radius must be > 0"
    @assert tube_radius > 0.0 "tube_radius must be > 0"
    @assert major_radius + tube_radius <= par.Lv "major_radius + tube_radius must be <= par.Lv"

    vx = zeros(Float64, N)
    vy = zeros(Float64, N)
    vz = zeros(Float64, N)

    # Uniform major angle and uniform disk cross-section; the resulting
    # volume density is proportional to 1 / sqrt(vx^2 + vy^2).
    @inbounds for i in 1:N
        theta = 2π * rand(rng)
        phi = 2π * rand(rng)
        rho = tube_radius * sqrt(rand(rng))

        ctheta = cos(theta)
        stheta = sin(theta)
        cphi = cos(phi)
        sphi = sin(phi)

        ring = major_radius + rho * cphi
        vx[i] = ring * ctheta
        vy[i] = ring * stheta
        vz[i] = rho * sphi
    end

    wg = fill(1.0 / N, N)
    vstack = hcat(vx, vy, vz)
    return vstack, wg
end

"""
    initialize_particles_spherical_shell_3d(N, par, rng; shell_radius=2.0, thickness=0.3)

Initialize `N` particles in 3D velocity space inside a spherical shell centered at the origin.
The shell has mean radius `shell_radius` and radial thickness `thickness`.

Sampling is uniform in shell volume, with radius distributed over
`[shell_radius - thickness/2, shell_radius + thickness/2]` and isotropic direction.

# Arguments
- `N`: Total number of particles
- `par`: Simulation parameters (must have `dim == 3`)
- `rng`: Random number generator
- `shell_radius`: Mean shell radius
- `thickness`: Shell thickness (must be > 0)

# Returns
- `(vstack, wg)` with `vstack` shaped `N×3` and uniform particle weights
"""
function initialize_particles_spherical_shell_3d(N::Int, par::Params, rng::AbstractRNG=Random.default_rng();
                                                 shell_radius::Float64=2.0,
                                                 thickness::Float64=0.3)
    @assert par.dim == 3 "initialize_particles_spherical_shell_3d requires par.dim == 3"
    @assert N > 0 "N must be positive"
    @assert shell_radius > 0.0 "shell_radius must be > 0"
    @assert thickness > 0.0 "thickness must be > 0"

    r_in = shell_radius - 0.5 * thickness
    r_out = shell_radius + 0.5 * thickness
    @assert r_in >= 0.0 "thickness is too large: inner radius would be negative"
    @assert r_out <= par.Lv "shell outer radius must be <= par.Lv"

    vx = zeros(Float64, N)
    vy = zeros(Float64, N)
    vz = zeros(Float64, N)

    rin3 = r_in^3
    rout3 = r_out^3

    @inbounds for i in 1:N
        # Uniform in shell volume: r^3 is uniform on [r_in^3, r_out^3]
        r = (rin3 + (rout3 - rin3) * rand(rng))^(1/3)

        # Isotropic direction using cos(theta) and phi
        μ = 2.0 * rand(rng) - 1.0
        φ = 2π * rand(rng)
        s = sqrt(max(0.0, 1.0 - μ * μ))

        vx[i] = r * s * cos(φ)
        vy[i] = r * s * sin(φ)
        vz[i] = r * μ
    end

    wg = fill(1.0 / N, N)
    vstack = hcat(vx, vy, vz)
    return vstack, wg
end

"""
    initialize_particles_stratified_maxwellian(N, dV, Vgrid, par, rng)

Stratified initialization of `N` particles from a Maxwellian distribution on a 2D or 3D velocity grid.

This method distributes particles uniformly across grid cells according to local Maxwellian weights, rather than sampling independently from the Maxwellian. For independent samples, use `initialize_particles_maxwellian` instead. Neither initialization guarantees an exactly stationary numerical reconstruction at finite particle number and grid resolution.

# Arguments
- `N`: Total number of particles
- `dV`: Grid cell width
- `Vgrid`: matrix of velocity-grid cell centres (one centre per row), from `build_velocity_grid`
- `par`: Simulation parameters (must contain `temp`, `dim`, `Lv`)
- `rng`: (optional) random number generator

# Returns
- `(vstack, wg)` where `vstack` is N×2 or N×3 matrix (2D or 3D) and `wg` is weights
"""
function initialize_particles_stratified_maxwellian(N::Int, dV::Float64,
                                                    Vgrid::AbstractArray{Float64},
                                                    par::Params,
                                                    rng::AbstractRNG=Random.default_rng())
    grid_views = component_grid_views(Vgrid, par)
    M = length(grid_views[1])
    if M <= 0
        error("initialize_particles_stratified_maxwellian: Vgrid is empty")
    end
    sigma2 = par.temp
    r2 = Vector{Float64}(undef, M)
    if par.dim == 2
        vx_grid, vy_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = vx_grid[k]^2 + vy_grid[k]^2
        end
    else
        vx_grid, vy_grid, vz_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = vx_grid[k]^2 + vy_grid[k]^2 + vz_grid[k]^2
        end
    end
    f_vals = exp.(-r2 ./ (2 * sigma2)) ./ (2 * pi * sigma2)^(par.dim / 2)
    mass = f_vals .* dV ^ par.dim
    n_frac = (mass ./ sum(mass)) .* N
    n_int = floor.(n_frac)
    remainder = n_frac .- n_int
    sum_int = Int(round(sum(n_int)))
    deficit = N - sum_int
    if deficit < 0
        deficit = 0
    end
    rems = sort!([(remainder[k], k) for k in 1:M], by=x->x[1], rev=true)
    for i in 1:min(deficit, M)
        _, k = rems[i]
        n_int[k] += 1.0
    end
    total_assigned = Int(round(sum(n_int)))
    if total_assigned != N
        diff = N - total_assigned
        k = 1
        while diff != 0
            if diff > 0
                n_int[k] += 1.0
                diff -= 1
            elseif n_int[k] > 0
                n_int[k] -= 1.0
                diff += 1
            end
            k = k % M + 1
        end
    end
    dim = par.dim
    vstack = zeros(Float64, N, dim)
    count = 0
    component_views = grid_views
    for k in 1:M
        num_samples = Int(round(n_int[k]))
        if num_samples <= 0
            continue
        end
        for _ in 1:num_samples
            if count >= N
                break
            end
            count += 1
            for d in 1:dim
                vstack[count, d] = component_views[d][k] - dV / 2.0 + rand(rng) * dV
            end
        end
    end
    if count < N
        while count < N
            count += 1
            for d in 1:dim
                vstack[count, d] = -par.Lv + rand(rng) * (2.0 * par.Lv)
            end
        end
    end
    wg = fill(sum(mass) / N, N)
    return vstack, wg
end

"""
    initialize_particles_stratified_BKW(vx, vy, wg, N, dV, Vgrid, par, [rng])

Stratified initialization of N particles from the BKW distribution on a 2D or 3D velocity grid.

# Arguments
- `vx`, `vy`, `wg`: Preallocated vectors to be filled with particle velocities and weights
- `N`: Total number of particles
- `dV`: Grid cell width
- `Vgrid`: matrix of velocity-grid cell centres (one centre per row), from `build_velocity_grid`
- `par`: Simulation parameters (must contain `temp`, `c`, `beta`)
- `rng`: (optional) random number generator

# Returns
- `(vstack, wg)` after copying the generated 2D particles into the supplied vectors
"""
function initialize_particles_stratified_BKW(vx::AbstractVector, vy::AbstractVector,
                                              wg::AbstractVector, N::Int, dV::Float64,
                                              Vgrid::AbstractArray{Float64},
                                              par::Params,
                                              rng::AbstractRNG=Random.default_rng())
    vstack, wg_out = initialize_particles_stratified_BKW(N, dV, Vgrid, par, rng)
    vx_out = @view vstack[:, 1]
    vy_out = @view vstack[:, 2]
    resize!(vx, length(vx_out))
    resize!(vy, length(vy_out))
    resize!(wg, length(wg_out))
    vx .= vx_out
    vy .= vy_out
    wg .= wg_out
    return vstack, wg
end

function initialize_particles_stratified_BKW(N::Int, dV::Float64,
                                              Vgrid::AbstractArray{Float64},
                                              par::Params,
                                              rng::AbstractRNG=Random.default_rng())
    grid_views = component_grid_views(Vgrid, par)
    M = length(grid_views[1])
    @assert par.dim in (2, 3) "initialize_particles_stratified_BKW is only implemented for dim==2 or dim==3"

    f_flat = BKW_vector(0.0, par.temp, par.c, par.beta, Vgrid)
    min_f = minimum(f_flat)
    if min_f < -100eps(Float64)
        error("initialize_particles_stratified_BKW: BKW density is negative for these parameters; reduce par.c or increase the initial K value")
    end
    f_flat = max.(f_flat, 0.0)
    mass = f_flat .* dV ^ par.dim
    n_frac = mass .* N
    n_int = floor.(n_frac)
    remainder = n_frac .- n_int

    sum_int = Int(round(sum(n_int)))
    deficit = N - sum_int
    if deficit < 0
        deficit = 0
    end

    rems = sort!([(remainder[k], k) for k in 1:M], by=x->x[1], rev=true)
    for i in 1:min(deficit, M)
        _, k = rems[i]
        n_int[k] += 1.0
    end

    total_assigned = Int(round(sum(n_int)))
    if total_assigned != N
        diff = N - total_assigned
        k = 1
        while diff != 0
            if diff > 0
                n_int[k] += 1.0
                diff -= 1
            elseif n_int[k] > 0
                n_int[k] -= 1.0
                diff += 1
            end
            k = k % M + 1
        end
    end

    vstack = zeros(Float64, N, par.dim)
    count = 0
    for k in 1:M
        num_samples = Int(round(n_int[k]))
        if num_samples <= 0
            continue
        end
        cell_mins = ntuple(d -> grid_views[d][k] - dV / 2.0, par.dim)
        for _ in 1:num_samples
            if count >= N
                break
            end
            count += 1
            for d in 1:par.dim
                vstack[count, d] = cell_mins[d] + rand(rng) * dV
            end
        end
    end

    if count < N
        while count < N
            count += 1
            for d in 1:par.dim
                vstack[count, d] = -par.Lv + rand(rng) * (2.0 * par.Lv)
            end
        end
    end

    wg = fill(sum(mass) / N, N)
    return vstack, wg
end

"""
    initialize_particles_stratified_BKW_elliptical(N, dV, Vgrid, par, [rng]; stretch=2.0)

Stratified initialization of N particles from a BKW distribution stretched along the x axis (elliptical shape).

# Arguments
- `N`: Total number of particles
- `dV`: Grid cell width
- `Vgrid`: matrix of velocity-grid cell centres (one centre per row), from `build_velocity_grid`
- `par`: Simulation parameters (must contain `temp`, `c`, `beta`)
- `rng`: (optional) random number generator
- `stretch`: Stretch factor along x axis (default: 2.0)

# Returns
- `(vstack, wg)` where `vstack` is `N×dim`
"""
function initialize_particles_stratified_BKW_elliptical(N::Int, dV::Float64,
                                                        Vgrid::AbstractArray{Float64},
                                                        par::Params,
                                                        rng::AbstractRNG=Random.default_rng();
                                                        stretch::Float64=2.0)
    Vgrid_stretch = copy(Vgrid)
    selectdim(Vgrid_stretch, ndims(Vgrid_stretch), 1) ./= stretch
    vstack, wg = initialize_particles_stratified_BKW(N, dV, Vgrid_stretch, par, rng)
    vstack[:, 1] .*= stretch
    return vstack, wg
end

"""
    initialize_particles_stratified_double_ring(N, dV, Vgrid, par, [rng]; ...)

Stratified initialization of N particles from two concentric Gaussian rings (and optional central Maxwellian) in velocity space.

# Arguments
- `N`: Total number of particles
- `dV`: Grid cell width
- `Vgrid`: matrix of velocity-grid cell centres (one centre per row), from `build_velocity_grid`
- `par`: Simulation parameters (must contain `temp`)
- `rng`: (optional) random number generator
- `include_bulk`: Whether to include a central Maxwellian (default: true)
- `ring1_fraction`, `ring2_fraction`: Fractions of total density in each ring (default: 0.15)
- `ring1_radius`, `ring2_radius`: Radial positions of the rings (default: 1.3, 2.7)
- `ring1_width`, `ring2_width`: Widths (standard deviation) of the rings (default: 0.15)

# Returns
- `(vstack, wg)` where `vstack` is `N×dim`
"""
function initialize_particles_stratified_double_ring(N::Int, dV::Float64,
                                                       Vgrid::AbstractArray{Float64},
                                                       par::Params,
                                                       rng::AbstractRNG=Random.default_rng();
                                                       include_bulk::Bool=true,
                                                       ring1_fraction::Float64=0.15,
                                                       ring1_radius::Float64=1.3,
                                                       ring1_width::Float64=0.15,
                                                       ring2_fraction::Float64=0.15,
                                                       ring2_radius::Float64=2.7,
                                                       ring2_width::Float64=0.15)
    grid_views = component_grid_views(Vgrid, par)
    M = length(grid_views[1])
    dim = par.dim

    T0 = par.temp
    m = 1.0
    r2 = Vector{Float64}(undef, M)
    if par.dim == 2
        vx_grid, vy_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = vx_grid[k]^2 + vy_grid[k]^2
        end
    else
        vx_grid, vy_grid, vz_grid = grid_views
        @inbounds for k in eachindex(vx_grid)
            r2[k] = vx_grid[k]^2 + vy_grid[k]^2 + vz_grid[k]^2
        end
    end
    r = sqrt.(r2)

    if include_bulk
        f_bulk = (m / (2.0 * pi * T0)) .* exp.(-m .* r2 ./ (2.0 * T0))
        bulk_fraction = 1.0 - ring1_fraction - ring2_fraction
        if bulk_fraction < 0.0
            error("Ring fractions sum to more than 1.0: ring1_fraction=$ring1_fraction, ring2_fraction=$ring2_fraction")
        end
    else
        f_bulk = zeros(M)
        total_ring_fraction = ring1_fraction + ring2_fraction
        if total_ring_fraction <= 0.0
            error("Ring fractions must be positive when include_bulk=false")
        end
        ring1_fraction = ring1_fraction / total_ring_fraction
        ring2_fraction = ring2_fraction / total_ring_fraction
        bulk_fraction = 0.0
    end

    sigma1 = ring1_width
    A_ring1 = 1.0 / (2.0 * pi * ring1_radius * sigma1 * sqrt(2.0 * pi))
    f_ring1 = A_ring1 .* exp.(-(r .- ring1_radius).^2 ./ (2.0 * sigma1^2))

    sigma2 = ring2_width
    A_ring2 = 1.0 / (2.0 * pi * ring2_radius * sigma2 * sqrt(2.0 * pi))
    f_ring2 = A_ring2 .* exp.(-(r .- ring2_radius).^2 ./ (2.0 * sigma2^2))

    f_total = bulk_fraction .* f_bulk .+ ring1_fraction .* f_ring1 .+ ring2_fraction .* f_ring2
    mass = f_total .* dV ^ dim

    n_frac = mass .* N
    n_int = floor.(n_frac)
    remainder = n_frac .- n_int

    sum_int = Int(round(sum(n_int)))
    deficit = N - sum_int
    if deficit < 0
        deficit = 0
    end

    rems = sort!([(remainder[k], k) for k in 1:M], by=x->x[1], rev=true)
    for i in 1:min(deficit, M)
        _, k = rems[i]
        n_int[k] += 1.0
    end

    total_assigned = Int(round(sum(n_int)))
    if total_assigned != N
        diff = N - total_assigned
        k = 1
        while diff != 0
            if diff > 0
                n_int[k] += 1.0
                diff -= 1
            elseif n_int[k] > 0
                n_int[k] -= 1.0
                diff += 1
            end
            k = k % M + 1
        end
    end

    vstack = zeros(Float64, N, dim)
    count = 0
    for k in 1:M
        num_samples = Int(round(n_int[k]))
        if num_samples <= 0
            continue
        end
        for _ in 1:num_samples
            if count >= N
                break
            end
            count += 1
            for d in 1:dim
                vstack[count, d] = grid_views[d][k] - dV / 2.0 + rand(rng) * dV
            end
        end
    end

    if count < N
        while count < N
            count += 1
            for d in 1:dim
                vstack[count, d] = -par.Lv + rand(rng) * (2.0 * par.Lv)
            end
        end
    end

    wg = fill(sum(mass) / N, N)
    return vstack, wg
end
