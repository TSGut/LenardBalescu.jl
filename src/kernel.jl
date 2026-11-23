"""
    getA_Landau(vxi::Float64, vyi::Float64, vxj::Float64, vyj::Float64, g::Float64, beta::Float64)

Compute the four elements of the collision kernel matrix A for a velocity pair using the Landau operator.

# Arguments
 - `vxi`, `vyi`: Components of the first particle's velocity.
 - `vxj`, `vyj`: Components of the second particle's velocity.
 - `g`: Exponent parameter for the velocity difference.
 - `beta`: Scaling parameter for the kernel.

# Returns
 - `axL`, `axR`, `ayL`, `ayR`: The four elements of the collision kernel matrix A as a tuple.
"""
function getA_Landau(vxi::Float64, vyi::Float64, vxj::Float64, vyj::Float64, g::Float64, beta::Float64)
    # Compute velocity difference
    qx = vxi - vxj
    qy = vyi - vyj
    
    r = sqrt(qx * qx + qy * qy)
    nn = 0.0
    # Protect from 0^0
    if r > 0.0
        nn = r^g * beta
    end
    axL = nn * qy * qy
    axR = -nn * qx * qy
    ayL = -nn * qx * qy
    ayR = nn * qx * qx
    return axL, axR, ayL, ayR
end

"""
    getA_Landau_3D(vxi, vyi, vzi, vxj, vyj, vzj, g, beta)

Compute the 3×3 Landau tensor for a velocity pair in 3D:

    A = beta * |q|^g * (|q|^2 I - q q^T),   q = v_i - v_j

Returns the 9 tensor entries in row-major order.
"""
function getA_Landau_3D(vxi::Float64, vyi::Float64, vzi::Float64,
                        vxj::Float64, vyj::Float64, vzj::Float64,
                        g::Float64, beta::Float64)
    qx = vxi - vxj
    qy = vyi - vyj
    qz = vzi - vzj
    r2 = qx * qx + qy * qy + qz * qz
    if r2 <= 0.0
        return 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0
    end

    r = sqrt(r2)
    nn = beta * r^g

    a11 = nn * (r2 - qx * qx)
    a12 = -nn * (qx * qy)
    a13 = -nn * (qx * qz)
    a21 = a12
    a22 = nn * (r2 - qy * qy)
    a23 = -nn * (qy * qz)
    a31 = a13
    a32 = a23
    a33 = nn * (r2 - qz * qz)

    return a11, a12, a13,
           a21, a22, a23,
           a31, a32, a33
end


"""
    precompute_k_space(k_perp_order::Int, k_perp_max::Float64, k_parallel_width::Float64, k_parallel_order::Int)

Precompute 2D k-space grid and weights matching MATLAB reference:
- Sinh-mapped Gauss-Legendre in k_perp on [-k_perp_max, k_perp_max], concentrating nodes near |k| ~ k_perp_scale
- Hat function approximation for delta in k_parallel on [-k_parallel_width, k_parallel_width]

# Arguments
- `k_perp_order`: Number of Gauss-Legendre quadrature points for k_perp (resolution)
- `k_perp_max`: Maximum k_perp value (integration range)
- `k_parallel_width`: Hat-delta width for k_parallel (delta function approximation)
- `k_parallel_order`: Number of points for k_parallel (resolution of delta approximation)
- `k_perp_scale`: Characteristic k scale for sinh node clustering (default 1.0 = thermal scale)

# Returns
- `k_parallel`, `k_perp`, `norm_k`, `approx`, `phi_p_11`, `phi_p_22`, `phi_p_12`, `k_angle_local`
"""
function precompute_k_space(k_perp_order::Int, k_perp_max::Float64, k_parallel_width::Float64, k_parallel_order::Int;
                            k_perp_scale::Float64=1.0, potential::Symbol=:coulomb)

    # Sinh-mapped GL quadrature: k = k_perp_scale * sinh(t * α_sh), t ∈ [-1,1],
    # where α_sh = asinh(k_perp_max / k_perp_scale).  Concentrates nodes near
    # |k_perp| ~ k_perp_scale (thermal scale) while still covering [−k_perp_max, k_perp_max].
    nodes_t, weights_t = gauss_legendre_quadrature(k_perp_order, -1.0, 1.0)
    alpha_sh = asinh(k_perp_max / k_perp_scale)
    nodes_perp   = k_perp_scale .* sinh.(nodes_t .* alpha_sh)
    weights_perp = weights_t .* (k_perp_scale * alpha_sh) .* cosh.(nodes_t .* alpha_sh)

    # Delta approximation in k_parallel
    nodes_parallel = range(-k_parallel_width, k_parallel_width, length=k_parallel_order)
    d_delta = 2.0 * k_parallel_width / (k_parallel_order - 1)
    delta_approx = hat(collect(nodes_parallel), k_parallel_width)
    approx = (weights_perp .* (delta_approx' .* d_delta))

    # Meshgrid in rotated coordinates
    k_parallel = repeat(nodes_parallel', length(nodes_perp), 1)
    k_perp = repeat(nodes_perp, 1, length(nodes_parallel))
    norm_k = sqrt.(k_parallel.^2 .+ k_perp.^2)

    # Precompute Phi(p): V̂(k)=1/|k|² for :coulomb, V̂(k)=1 for :constant_fourier
    hat_V = is_constant_fourier(potential) ? ones(size(norm_k)) : 1.0 ./ (norm_k.^2 .+ 1e-15)
    hat_V_sq = hat_V.^2
    phi_p_11 = (k_parallel.^2) .* hat_V_sq
    phi_p_22 = (k_perp.^2) .* hat_V_sq
    phi_p_12 = (k_parallel .* k_perp) .* hat_V_sq

    # Precompute local-frame angle of each k-sample so callers can avoid atan in the hot loop.
    k_angle_local = atan.(k_perp, k_parallel)

    return k_parallel, k_perp, norm_k, approx, phi_p_11, phi_p_22, phi_p_12, k_angle_local
end

"""
    precompute_k_space_3d(k_perp_order, k_perp_max, k_parallel_width, k_parallel_order)

Precompute 3D k-space samples in a local frame `(k_parallel, k_perp2, k_perp3)`.
The two perpendicular directions use sinh-mapped Gauss-Legendre quadrature and `k_parallel`
uses the same hat-delta approximation as the 2D implementation.

`k_perp_order` is used directly as the quadrature order in each perpendicular
direction (`k_perp2`, `k_perp3`).
"""
function precompute_k_space_3d(k_perp_order::Int, k_perp_max::Float64,
                               k_parallel_width::Float64, k_parallel_order::Int;
                               k_perp_scale::Float64=1.0, potential::Symbol=:coulomb)
    n_perp = max(3, k_perp_order)

    nodes_t2, weights_t2 = gauss_legendre_quadrature(n_perp, -1.0, 1.0)
    alpha_sh = asinh(k_perp_max / k_perp_scale)
    nodes_perp2   = k_perp_scale .* sinh.(nodes_t2 .* alpha_sh)
    weights_perp2 = weights_t2 .* (k_perp_scale * alpha_sh) .* cosh.(nodes_t2 .* alpha_sh)
    nodes_perp3   = copy(nodes_perp2)
    weights_perp3 = copy(weights_perp2)

    nodes_parallel = collect(range(-k_parallel_width, k_parallel_width, length=k_parallel_order))
    d_delta = 2.0 * k_parallel_width / (k_parallel_order - 1)
    delta_approx = hat(nodes_parallel, k_parallel_width)

    n_tot = n_perp * n_perp * k_parallel_order
    k_parallel = zeros(Float64, n_tot)
    k_perp2 = zeros(Float64, n_tot)
    k_perp3 = zeros(Float64, n_tot)
    norm_k = zeros(Float64, n_tot)
    approx = zeros(Float64, n_tot)
    inv_norm_k4 = zeros(Float64, n_tot)

    idx = 1
    @inbounds for a in 1:n_perp
        k2 = nodes_perp2[a]
        w2 = weights_perp2[a]
        for b in 1:n_perp
            k3 = nodes_perp3[b]
            w3 = weights_perp3[b]
            for c in 1:k_parallel_order
                kp = nodes_parallel[c]
                w = w2 * w3 * delta_approx[c] * d_delta
                k2sq = k2 * k2
                k3sq = k3 * k3
                kn = sqrt(kp * kp + k2sq + k3sq)

                k_parallel[idx] = kp
                k_perp2[idx] = k2
                k_perp3[idx] = k3
                norm_k[idx] = kn
                approx[idx] = w
                # inv_norm_k4 stores V̂²(k): 1/|k|⁴ for :coulomb, 1 for :constant_fourier
                inv_norm_k4[idx] = is_constant_fourier(potential) ? 1.0 : 1.0 / ((kn * kn + 1e-15)^2)
                idx += 1
            end
        end
    end

    return k_parallel, k_perp2, k_perp3, norm_k, approx, inv_norm_k4
end

"""
    compute_hilbert_transform(x::AbstractVector, y::AbstractVector)

Compute Hilbert transform of y(x) via DSP.jl's FFT-based algorithm.

The Hilbert transform is: `H[y](x)` = (1/π) ∫ y(t)/(x-t) dt
Implementation adds four signal lengths of zeros on each side to reduce wrap-around artifacts.

# Arguments
- `x`: Grid points (must be uniformly spaced for FFT)
- `y`: Function values at grid points

# Returns
- `H_y::Vector`: Hilbert transform values (same length as y)
"""
function compute_hilbert_transform(x::AbstractVector, y::AbstractVector)
    # Check uniform spacing
    if length(x) < 2
        return similar(y)
    end
    
    dx = x[2] - x[1]
    for i in 2:(length(x)-1)
        if abs(x[i+1] - x[i] - dx) > 1e-10
            error("Hilbert transform requires uniformly spaced grid")
        end
    end
    
    # Symmetric zero-padding (4x signal length each side) to reduce
    # wrap-around artifacts, matching the MATLAB reference implementation.
    n = length(y)
    n_pad = 4 * n
    y_padded = [zeros(n_pad); y; zeros(n_pad)]

    # Use DSP.jl hilbert function (returns analytic signal via FFT)
    analytic_signal = hilbert(y_padded)

    # Extract imaginary part from the central segment
    H_y = imag(analytic_signal[n_pad+1:n_pad+n])

    return H_y
end

@inline function finite_difference_uniform(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    n = length(y)
    dydx = zeros(Float64, n)
    if n <= 1
        return dydx
    end
    dx = x[2] - x[1]
    @inbounds for i in 2:(n - 1)
        dydx[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx)
    end
    @inbounds dydx[1] = (y[2] - y[1]) / dx
    @inbounds dydx[end] = (y[end] - y[end - 1]) / dx
    return dydx
end

@inline function bilinear_interp_uniform(f::AbstractMatrix{Float64}, x::Float64, y::Float64,
                                         x0::Float64, dx::Float64, y0::Float64, dy::Float64)
    nx, ny = size(f)
    tx = (x - x0) / dx + 1.0
    ty = (y - y0) / dy + 1.0

    if !(1.0 <= tx <= nx && 1.0 <= ty <= ny)
        return 0.0
    end

    i0 = clamp(floor(Int, tx), 1, nx - 1)
    j0 = clamp(floor(Int, ty), 1, ny - 1)
    ax = tx - i0
    ay = ty - j0

    f00 = f[i0, j0]
    f10 = f[i0 + 1, j0]
    f01 = f[i0, j0 + 1]
    f11 = f[i0 + 1, j0 + 1]

    # Original equivalent form:
    # return (1.0 - ax) * (1.0 - ay) * f00 +
    #        ax * (1.0 - ay) * f10 +
    #        (1.0 - ax) * ay * f01 +
    #        ax * ay * f11
    f0 = muladd(ax, f10 - f00, f00)
    f1 = muladd(ax, f11 - f01, f01)
    return muladd(ay, f1 - f0, f0)
end

@inline function trilinear_interp_uniform(f::Array{Float64,3}, x::Float64, y::Float64, z::Float64,
                                          x0::Float64, dx::Float64,
                                          y0::Float64, dy::Float64,
                                          z0::Float64, dz::Float64)
    nx, ny, nz = size(f)
    tx = (x - x0) / dx + 1.0
    ty = (y - y0) / dy + 1.0
    tz = (z - z0) / dz + 1.0

    if !(1.0 <= tx <= nx && 1.0 <= ty <= ny && 1.0 <= tz <= nz)
        return 0.0
    end

    i0 = clamp(floor(Int, tx), 1, nx - 1)
    j0 = clamp(floor(Int, ty), 1, ny - 1)
    k0 = clamp(floor(Int, tz), 1, nz - 1)

    ax = tx - i0
    ay = ty - j0
    az = tz - k0

    c000 = f[i0, j0, k0]
    c100 = f[i0 + 1, j0, k0]
    c010 = f[i0, j0 + 1, k0]
    c110 = f[i0 + 1, j0 + 1, k0]
    c001 = f[i0, j0, k0 + 1]
    c101 = f[i0 + 1, j0, k0 + 1]
    c011 = f[i0, j0 + 1, k0 + 1]
    c111 = f[i0 + 1, j0 + 1, k0 + 1]

    # Original equivalent form:
    # c00 = (1.0 - ax) * c000 + ax * c100
    # c10 = (1.0 - ax) * c010 + ax * c110
    # c01 = (1.0 - ax) * c001 + ax * c101
    # c11 = (1.0 - ax) * c011 + ax * c111
    # c0 = (1.0 - ay) * c00 + ay * c10
    # c1 = (1.0 - ay) * c01 + ay * c11
    # return (1.0 - az) * c0 + az * c1
    c00 = muladd(ax, c100 - c000, c000)
    c10 = muladd(ax, c110 - c010, c010)
    c01 = muladd(ax, c101 - c001, c001)
    c11 = muladd(ax, c111 - c011, c011)
    c0 = muladd(ay, c10 - c00, c00)
    c1 = muladd(ay, c11 - c01, c01)

    return muladd(az, c1 - c0, c0)
end

"""
    compute_radial_integral_G(x::Float64, r::AbstractVector, f_prime::AbstractVector, dim::Int=2)

Compute the radial integral G(x) for the computed Hilbert transform.

For radially symmetric distributions:
    G(x) = S_{d-2} ∫_{|x|}^{∞} f'(r) (r² - x²)^{(d-3)/2} dr

where S_{d-2} is the surface area of unit sphere in d-1 dimensions.

# Arguments
- `x`: Evaluation point (typically v_parallel)
- `r`: Radial grid points (must be sorted)
- `f_prime`: Derivative values on radial grid
- `dim`: Dimension (2 or 3)

# Returns
- `G_x::Float64`: Value of G at x
"""
function compute_radial_integral_G(x::Float64, r::AbstractVector, f_prime::AbstractVector, dim::Int=2)
    @assert dim in (2, 3) "dim must be 2 or 3 (got=$(dim))"
    S = (dim == 2) ? 2.0 : (2.0 * π)  # S_{d-2}: S_0=2 for d=2, S_1=2π for d=3
    exponent = (dim - 3) / 2.0

    # Find indices where r >= |x|
    x_abs = abs(x)
    idx_start = if r isa AbstractRange
        r0 = first(r)
        dr = step(r)
        clamp(Int(floor((x_abs - r0) / dr)) + 1, 1, length(r))
    else
        searchsortedfirst(r, x_abs)
    end

    if idx_start > length(r)
        return 0.0
    end

    # Integrate from |x| to end of grid using trapezoidal rule
    r_subset = r[idx_start:end]
    f_prime_subset = f_prime[idx_start:end]

    if length(r_subset) < 2
        return 0.0
    end

    # Compute integrand: f'(r) * (r² - x²)^{exponent}
    integrand = similar(r_subset)
    @inbounds for i in eachindex(r_subset)
        r_val = r_subset[i]
        if r_val > x_abs + 1e-15
            integrand[i] = f_prime_subset[i] * (r_val^2 - x_abs^2)^exponent
        else
            integrand[i] = 0.0
        end
    end

    # Trapezoidal rule
    result = 0.0
    @inbounds for i in 1:(length(integrand)-1)
        dr = r_subset[i+1] - r_subset[i]
        result += 0.5 * (integrand[i] + integrand[i+1]) * dr
    end

    return S * result
end

"""
    extract_uniform_axis_spacing(vals::AbstractArray{Float64}) -> (v0::Float64, dv::Float64)

Extract the origin and uniform spacing from a velocity grid axis array.

Reads all unique values from `vals`, validates that they form a uniformly-spaced
sequence, and returns the minimum value and the spacing.

# Arguments
- `vals`: Array of grid coordinates for one velocity axis (may be multi-dimensional).

# Returns
- `v0`: Minimum coordinate value (grid origin for that axis).
- `dv`: Uniform grid spacing.

# Errors
Throws an error if fewer than 2 unique values are found, if the spacing is not
positive and finite, or if the axis is not uniformly spaced.
"""
@inline function extract_uniform_axis_spacing(vals::AbstractArray{Float64})
    uniq = sort!(unique(vec(vals)))
    if length(uniq) < 2
        error("Degenerate velocity axis: expected at least 2 unique grid points")
    end
    dv = uniq[2] - uniq[1]
    if !(isfinite(dv) && dv > 0.0)
        error("Invalid velocity axis spacing detected: Δv=$(dv)")
    end
    tol = 1e-10 * max(1.0, abs(dv))
    @inbounds for i in 2:(length(uniq)-1)
        if abs((uniq[i + 1] - uniq[i]) - dv) > tol
            error("Velocity axis is not uniformly spaced; non-radial dielectric requires a uniform grid")
        end
    end
    return uniq[1], dv
end

"""
    projected_line_points(r_extent::Float64, dv_par::Float64, Nv::Int) -> Int

Compute the number of points for the projected parallel velocity line used in
non-radial dielectric precomputation.

Ensures that the line from `-r_extent` to `r_extent` is sampled at least as
densely as the underlying velocity grid (spacing `dv_par`), and always returns
an odd count so that the grid is symmetric about zero.

# Arguments
- `r_extent`: Half-length of the projected line (maximum particle speed on grid).
- `dv_par`: Target parallel grid spacing (typically the minimum axis spacing).
- `Nv`: Minimum number of points (usually the grid resolution per axis).

# Returns
- Number of sample points (odd integer, ≥ `Nv`).
"""
@inline function projected_line_points(r_extent::Float64, dv_par::Float64, Nv::Int)
    if !isfinite(r_extent) || r_extent <= 0.0
        return max(3, Nv)
    end
    n = max(Nv, Int(ceil((2.0 * r_extent) / dv_par)) + 1)
    return iseven(n) ? (n + 1) : n
end

"""
    precompute_dielectric_data(r, f_prime, dim=2, nang=16;
                               radial_symmetry=true, f=nothing, grid=nothing,
                               rho_points_3d=0, psi_points_3d=0)

Precompute `G(x)` and `H[G](x)` for efficient dielectric response calculation.

Computes once:
- G(x) = S_{d-2} ∫_{|x|}^{∞} f'(r) (r²-x²)^{(d-3)/2} dr
- `H[G](x)` = Hilbert transform of `G`
- I = ∫_0^∞ f'(r) r^{d-2} dr

# Arguments
- `r`: Radial grid for distribution (`radial_symmetry=true` path; otherwise `nothing`)
- `f_prime`: Distribution derivative on radial grid (`radial_symmetry=true` path; otherwise `nothing`)
- `dim`: Dimension (`2` or `3`)
- `nang`: Angular sampling count for generic (non-radial) mode
- `radial_symmetry`: Select radial (`true`) or generic (`false`) dielectric preprocessing
- `f`: Reconstructed distribution on velocity grid (required in generic mode)
- `grid`: Velocity-grid coordinate tuple used in generic mode (`(Vx, Vy)` or `(Vx, Vy, Vz)`)
- `rho_points_3d`, `psi_points_3d`: Optional 3D non-radial quadrature resolutions

# Returns
- `G`: G values on extended grid
- `HG`: Hilbert[G] values on extended grid
- `r_ext`: Extended radial grid [-L, ..., L]
- `I_integral`: Normalization integral used in radial mode
"""
function precompute_dielectric_data(
    r::Union{Nothing,AbstractVector}, f_prime::Union{Nothing,AbstractVector}, dim::Int=2, nang::Int=16;
    radial_symmetry::Bool=true, f::Union{Nothing,AbstractArray}=nothing, grid::Union{Nothing,Tuple}=nothing,
    rho_points_3d::Int=0, psi_points_3d::Int=0)
    @assert dim in (2, 3) "dim must be 2 or 3 (got=$(dim))"

    if radial_symmetry
        # Radial mode: build a symmetric 1D lookup table G(r_parallel), then Hilbert-transform it.
        @assert length(r) == length(f_prime) "r and f_prime must have the same length"
        L = maximum(r)
        n_orig = length(r)
        n_extended = 2 * n_orig - 1
        r_extended = collect(LinRange(-L, L, n_extended))
        G_extended = similar(r_extended)
        @inbounds for i in eachindex(r_extended)
            G_extended[i] = compute_radial_integral_G(r_extended[i], r, f_prime, dim)
        end
        HG_extended = compute_hilbert_transform(r_extended, G_extended)

        # I is a radial half-line integral:
        #     I = ∫₀^∞ f'(r) r^(d-2) dr.
        # `compute_radial_binned_f` returns an even reconstruction on a signed
        # grid, so integrating over the whole grid would cancel I in 2D and
        # double it in 3D. Integrate only the nonnegative half and include the
        # origin, where f'(0)=0 for a smooth radial distribution.
        first_nonnegative = searchsortedfirst(r, 0.0)
        @assert first_nonnegative <= length(r) "radial grid must contain positive radii"
        I_integral = 0.0
        previous_r = 0.0
        previous_integrand = 0.0
        @inbounds for i in first_nonnegative:length(r)
            radius = r[i]
            integrand = f_prime[i] * radius^(dim - 2)
            radial_step = radius - previous_r
            @assert radial_step >= 0.0 "radial grid must be sorted"
            I_integral += 0.5 * (previous_integrand + integrand) * radial_step
            previous_r = radius
            previous_integrand = integrand
        end
        return G_extended, HG_extended, r_extended, I_integral
    else
        # Generic non-radial mode: precompute directional profiles indexed by angle.
        # For each direction e, build F_e(x)=∫_{u_perp} f(x e + u_perp) du_perp,
        # then G_e(x)=dF_e/dx and HG_e=Hilbert[G_e].
        @assert f !== nothing "f (distribution array) must be provided for non-radial case"
        @assert grid !== nothing "grid must be provided for non-radial case"

        if dim == 2
            @assert length(grid) == 2 "For dim=2 non-radial case, grid must be (Vx, Vy)"
            Vx, Vy = grid
            f_mat = reshape(f, size(Vx))
            Nv = size(Vx, 1)

            x0, dx_grid = extract_uniform_axis_spacing(Vx)
            y0, dy_grid = extract_uniform_axis_spacing(Vy)
            Lx = maximum(abs.(Vx))
            Ly = maximum(abs.(Vy))

            r_extent = maximum(sqrt.(Vx.^2 .+ Vy.^2))
            dv_par = min(dx_grid, dy_grid)
            n_parallel = projected_line_points(r_extent, dv_par, Nv)
            x_centers = collect(range(-r_extent, r_extent; length=n_parallel))

            theta_nodes = collect(range(0.0, 2π; length=nang + 1))[1:end-1]
            G_dir = zeros(Float64, nang, n_parallel)
            HG_dir = zeros(Float64, nang, n_parallel)

            n_perp = max(6, min(16, Nv))
            u_max = sqrt(2.0) * max(Lx, Ly)
            u_nodes, u_weights = gauss_legendre_quadrature(n_perp, -u_max, u_max)

            # Sweep each direction on the unit circle, compute F_local, then d/dx and Hilbert.
            Threads.@threads for it in eachindex(theta_nodes)
                theta = theta_nodes[it]
                kx = cos(theta)
                ky = sin(theta)
                e2x = -ky
                e2y = kx

                F_local = zeros(Float64, n_parallel)
                @inbounds for ix in 1:n_parallel
                    x_par = x_centers[ix]
                    acc = 0.0
                    for (u, wu) in zip(u_nodes, u_weights)
                        xx = x_par * kx + u * e2x
                        yy = x_par * ky + u * e2y
                        if abs(xx) <= Lx + 1e-12 && abs(yy) <= Ly + 1e-12
                            acc += wu * bilinear_interp_uniform(f_mat, xx, yy, x0, dx_grid, y0, dy_grid)
                        end
                    end
                    F_local[ix] = acc
                end

                G_local = finite_difference_uniform(x_centers, F_local)

                HG_local = compute_hilbert_transform(x_centers, G_local)
                @inbounds G_dir[it, :] .= G_local
                @inbounds HG_dir[it, :] .= HG_local
            end

            # Return directional lookup tables G(θ, x_parallel), HG(θ, x_parallel).
            return G_dir, HG_dir, x_centers, 0.0
        else
            @assert length(grid) == 3 "For dim=3 non-radial case, grid must be (Vx, Vy, Vz)"
            Vx, Vy, Vz = grid
            f_mat = reshape(f, size(Vx))
            Nv = size(Vx, 1)

            x0, dx_grid = extract_uniform_axis_spacing(Vx)
            y0, dy_grid = extract_uniform_axis_spacing(Vy)
            z0, dz_grid = extract_uniform_axis_spacing(Vz)
            Lx = maximum(abs.(Vx))
            Ly = maximum(abs.(Vy))
            Lz = maximum(abs.(Vz))

            r_extent = maximum(sqrt.(Vx.^2 .+ Vy.^2 .+ Vz.^2))
            dv_par = min(dx_grid, min(dy_grid, dz_grid))
            n_parallel = projected_line_points(r_extent, dv_par, Nv)
            x_centers = collect(range(-r_extent, r_extent; length=n_parallel))

            mu_nodes = collect(range(-1.0, 1.0; length=nang))
            phi_nodes = collect(range(0.0, 2π; length=nang + 1))[1:end-1]
            nmu = length(mu_nodes)
            nphi = length(phi_nodes)
            G_dir = zeros(Float64, nmu, nphi, n_parallel)
            HG_dir = zeros(Float64, nmu, nphi, n_parallel)

            # Cylindrical coordinates in u_perp: (ρ, ψ), du_perp = ρ dρ dψ.
            n_rho = rho_points_3d > 0 ? rho_points_3d : max(12, Nv)
            n_psi = psi_points_3d > 0 ? psi_points_3d : max(8, div(nang, 2))
            rho_max = sqrt(3.0) * max(Lx, max(Ly, Lz))
            rho_nodes, rho_weights = gauss_legendre_quadrature(n_rho, 0.0, rho_max)
            psi_nodes = collect(range(0.0, 2π; length=n_psi + 1))[1:end-1]
            dpsi = 2π / n_psi

            # Precompute Cartesian gradient fields once; later we interpolate these off-grid.
            dfdx_mat = zeros(Float64, size(f_mat))
            dfdy_mat = zeros(Float64, size(f_mat))
            dfdz_mat = zeros(Float64, size(f_mat))
            nx, ny, nz = size(f_mat)
            Threads.@threads for i in 1:nx
                @inbounds for j in 1:ny, k in 1:nz
                    if i == 1
                        dfdx_mat[i, j, k] = (f_mat[i + 1, j, k] - f_mat[i, j, k]) / dx_grid
                    elseif i == nx
                        dfdx_mat[i, j, k] = (f_mat[i, j, k] - f_mat[i - 1, j, k]) / dx_grid
                    else
                        dfdx_mat[i, j, k] = (f_mat[i + 1, j, k] - f_mat[i - 1, j, k]) / (2.0 * dx_grid)
                    end

                    if j == 1
                        dfdy_mat[i, j, k] = (f_mat[i, j + 1, k] - f_mat[i, j, k]) / dy_grid
                    elseif j == ny
                        dfdy_mat[i, j, k] = (f_mat[i, j, k] - f_mat[i, j - 1, k]) / dy_grid
                    else
                        dfdy_mat[i, j, k] = (f_mat[i, j + 1, k] - f_mat[i, j - 1, k]) / (2.0 * dy_grid)
                    end

                    if k == 1
                        dfdz_mat[i, j, k] = (f_mat[i, j, k + 1] - f_mat[i, j, k]) / dz_grid
                    elseif k == nz
                        dfdz_mat[i, j, k] = (f_mat[i, j, k] - f_mat[i, j, k - 1]) / dz_grid
                    else
                        dfdz_mat[i, j, k] = (f_mat[i, j, k + 1] - f_mat[i, j, k - 1]) / (2.0 * dz_grid)
                    end
                end
            end

            # Sweep the (μ, φ) sphere grid; for each direction integrate k·∇f over transverse plane.
            Threads.@threads for imu in eachindex(mu_nodes)
                μ = mu_nodes[imu]
                sθ = sqrt(max(0.0, 1.0 - μ * μ))
                for iphi in eachindex(phi_nodes)
                    φ = phi_nodes[iphi]
                    kx = sθ * cos(φ)
                    ky = sθ * sin(φ)
                    kz = μ

                    inv_norm = 1.0 / (sqrt(kx * kx + ky * ky + kz * kz) + 1e-30)
                    e1x = kx * inv_norm
                    e1y = ky * inv_norm
                    e1z = kz * inv_norm
                    e2x, e2y, e2z, e3x, e3y, e3z = orthonormal_basis_from_e1(e1x, e1y, e1z)

                    G_local = zeros(Float64, n_parallel)
                    @inbounds for ix in 1:n_parallel
                        x_par = x_centers[ix]
                        acc = 0.0
                        for (ρ, wρ) in zip(rho_nodes, rho_weights)
                            for ψ in psi_nodes
                                cψ = cos(ψ)
                                sψ = sin(ψ)
                                xx = x_par * e1x + ρ * cψ * e2x + ρ * sψ * e3x
                                yy = x_par * e1y + ρ * cψ * e2y + ρ * sψ * e3y
                                zz = x_par * e1z + ρ * cψ * e2z + ρ * sψ * e3z
                                if abs(xx) <= Lx + 1e-12 && abs(yy) <= Ly + 1e-12 && abs(zz) <= Lz + 1e-12
                                    dfdx = trilinear_interp_uniform(dfdx_mat, xx, yy, zz,
                                                                    x0, dx_grid, y0, dy_grid, z0, dz_grid)
                                    dfdy = trilinear_interp_uniform(dfdy_mat, xx, yy, zz,
                                                                    x0, dx_grid, y0, dy_grid, z0, dz_grid)
                                    dfdz = trilinear_interp_uniform(dfdz_mat, xx, yy, zz,
                                                                    x0, dx_grid, y0, dy_grid, z0, dz_grid)
                                    kdotgrad = e1x * dfdx + e1y * dfdy + e1z * dfdz

                                    acc += wρ * dpsi * ρ * kdotgrad
                                end
                            end
                        end
                        G_local[ix] = acc
                    end

                    HG_local = compute_hilbert_transform(x_centers, G_local)
                    @inbounds G_dir[imu, iphi, :] .= G_local
                    @inbounds HG_dir[imu, iphi, :] .= HG_local
                end
            end

            # Return directional lookup tables G(μ, φ, x_parallel), HG(μ, φ, x_parallel).
            return G_dir, HG_dir, x_centers, 0.0
        end
    end
end

"""
    compute_dielectric_response!(dielectric_sq, k_nodes, v_parallel,
                                 G, HG, r, I_integral, dim=2)

Compute dielectric-response magnitude squared at each k-sample using precomputed
profiles `G`, `HG` on a uniform lookup grid `r`.

# Arguments
- `dielectric_sq`: Output array, updated in-place
- `k_nodes`: k magnitudes
- `v_parallel`: Parallel velocity samples corresponding to `k_nodes`
- `G`, `HG`: Precomputed profile arrays on `r`
- `r`: Uniform lookup grid used for interpolation
- `I_integral`: Normalization integral used in radial mode
- `dim`: Dimension (`2` or `3`)

# Returns
- The updated `dielectric_sq` array
"""
function compute_dielectric_response!(dielectric_sq::AbstractArray, k_nodes::AbstractArray,
                                      v_parallel::AbstractArray, G::AbstractVector,
                                      HG::AbstractVector, r::AbstractVector,
                                      I_integral::Float64, dim::Int=2)
    @assert dim in (2, 3) "dim must be 2 or 3 (got=$(dim))"
    S = (dim == 2) ? (2.0 * π) : (4.0 * π)  # S_{d-1}: 2π for d=2, 4π for d=3

    # Precompute grid parameters (assumes uniform spacing)
    r_first = r[1]
    r_last = r[end]
    len_r = length(r)
    r0 = r_first
    if len_r > 1
        dr = r[2] - r[1]
        inv_dr = 1.0 / dr
    else
        dr = 1.0
        inv_dr = 1.0
    end

    @inbounds for i in eachindex(k_nodes)
        k_val = k_nodes[i]
        v_par = v_parallel[i]

        if k_val < 1e-15
            dielectric_sq[i] = 1.0
            continue
        end

        if v_par <= r_first
            G_interp = G[1]
            HG_interp = HG[1]
        elseif v_par >= r_last
            G_interp = G[len_r]
            HG_interp = HG[len_r]
        else
            idx_G = floor(Int, (v_par - r0) * inv_dr) + 1
            idx_G = max(2, min(idx_G, len_r))
            r1 = r0 + (idx_G - 2) * dr
            t = (v_par - r1) * inv_dr
            G_interp = G[idx_G - 1] * (1.0 - t) + G[idx_G] * t
            HG_interp = HG[idx_G - 1] * (1.0 - t) + HG[idx_G] * t
        end

        V_sq = 1.0 / (k_val^2 + 1e-15)
        I_real = π * v_par * HG_interp - S * I_integral
        I_imag = π * v_par * G_interp

        eps_real = 1.0 + V_sq * I_real
        eps_imag = V_sq * I_imag

        dielectric_sq[i] = max(eps_real^2 + eps_imag^2, 1e-10)
    end

    return dielectric_sq
end

"""
    convolve_hat_at_point(r::AbstractVector, G::AbstractVector, v::Float64, width::Float64,
                          r0::Float64, dr::Float64)

Return discrete convolution of `G(r)` with the triangular `hat` kernel of half-width `width`
evaluated at `v`. Assumes `r` is uniformly spaced with spacing `dr` and origin `r0`.
"""
function convolve_hat_at_point(r::AbstractVector, G::AbstractVector, v::Float64, width::Float64,
                               r0::Float64, dr::Float64)
    if width <= 0.0
        # Degenerate: no smoothing
        # Find nearest index
        idx = clamp(floor(Int, (v - r0) / dr) + 1, 1, length(r))
        return G[idx]
    end

    # Determine index range overlapping kernel support
    j_start = clamp(floor(Int, (v - width - r0) / dr) + 1, 1, length(r))
    j_end = clamp(ceil(Int, (v + width - r0) / dr) + 1, 1, length(r))

    acc = 0.0
    @inbounds for j in j_start:j_end
        x = v - r[j]
        # hat kernel: max(1 - |x|/width, 0) / width
        h = (abs(x) < width) ? (1.0 - abs(x) / width) / width : 0.0
        acc += G[j] * h
    end

    # Discrete integral approximation: multiply by grid spacing
    return acc * dr
end

@inline function interpolate_profile_line(line::AbstractVector{Float64}, v_par::Float64,
                                          r0::Float64, dr::Float64, inv_dr::Float64, len_r::Int,
                                          r_first::Float64, r_last::Float64)
    if v_par <= r_first
        return line[1]
    elseif v_par >= r_last
        return line[len_r]
    else
        idx = floor(Int, (v_par - r0) * inv_dr) + 1
        idx = max(2, min(idx, len_r))
        r1 = r0 + (idx - 2) * dr
        t = (v_par - r1) * inv_dr
        return line[idx - 1] * (1.0 - t) + line[idx] * t
    end
end

# Direct-index variants that avoid @view + SubArray construction overhead.
@inline function interp_row(G::AbstractMatrix{Float64}, row::Int, v_par::Float64,
                             r0::Float64, dr::Float64, inv_dr::Float64, len_r::Int,
                             r_first::Float64, r_last::Float64)
    if v_par <= r_first
        return @inbounds G[row, 1]
    elseif v_par >= r_last
        return @inbounds G[row, len_r]
    else
        idx = clamp(floor(Int, (v_par - r0) * inv_dr) + 1, 2, len_r)
        v_left = r0 + (idx - 2) * dr
        t = (v_par - v_left) * inv_dr
        @inbounds g_left = G[row, idx - 1]
        @inbounds g_right = G[row, idx]
        # Original equivalent form:
        # return g_left * (1.0 - t) + g_right * t
        return muladd(t, g_right - g_left, g_left)
    end
end

@inline function interpolate_directional_profile(G::AbstractArray{Float64,3}, im::Int, ip::Int, v_par::Float64,
                                                 r0::Float64, dr::Float64, inv_dr::Float64, len_r::Int,
                                                 r_first::Float64, r_last::Float64)
    if v_par <= r_first
        return @inbounds G[im, ip, 1]
    elseif v_par >= r_last
        return @inbounds G[im, ip, len_r]
    else
        idx = clamp(floor(Int, (v_par - r0) * inv_dr) + 1, 2, len_r)
        v_left = r0 + (idx - 2) * dr
        t = (v_par - v_left) * inv_dr
        @inbounds g_left = G[im, ip, idx - 1]
        @inbounds g_right = G[im, ip, idx]
        # Original equivalent form:
        # return g_left * (1.0 - t) + g_right * t
        return muladd(t, g_right - g_left, g_left)
    end
end

# Compute angular bin weights from a precomputed (non-negative) lab-frame angle.
const _INV_2PI = 1.0 / (2π)
@inline function angle_to_periodic_weights(phi_lab::Float64, nang::Int)
    t = (phi_lab * _INV_2PI - floor(phi_lab * _INV_2PI)) * nang
    i0 = floor(Int, t) + 1
    α  = t - floor(t)
    i0 = max(1, min(i0, nang))
    i1 = (i0 == nang) ? 1 : (i0 + 1)
    return i0, i1, α
end

@inline function theta_periodic_weights(ckx::Float64, cky::Float64, nang::Int)
    θ = atan(cky, ckx)
    θ = θ < 0.0 ? (θ + 2π) : θ
    t = θ / (2π) * nang
    i0 = floor(Int, t) + 1
    α = t - floor(t)
    i0 = max(1, min(i0, nang))
    i1 = (i0 == nang) ? 1 : (i0 + 1)
    return i0, i1, α
end

@inline function mu_phi_bilinear_weights(ckx::Float64, cky::Float64, ckz::Float64,
                                         nmu::Int, nphi::Int)
    kn = sqrt(ckx * ckx + cky * cky + ckz * ckz) + 1e-30
    μ = max(-1.0, min(1.0, ckz / kn))
    tμ = (μ + 1.0) * 0.5 * (nmu - 1)
    tμ = max(0.0, min(tμ, nmu - 1))
    im0 = max(1, min(floor(Int, tμ) + 1, nmu - 1))
    im1 = im0 + 1
    αμ = tμ - floor(tμ)

    θ = atan(cky, ckx)
    θ = θ < 0.0 ? (θ + 2π) : θ
    tφ = θ / (2π) * nphi
    ip0 = floor(Int, tφ) + 1
    αφ = tφ - floor(tφ)
    ip0 = max(1, min(ip0, nphi))
    ip1 = (ip0 == nphi) ? 1 : (ip0 + 1)
    return im0, im1, αμ, ip0, ip1, αφ
end

@inline function orthonormal_basis_from_e1(e1x::Float64, e1y::Float64, e1z::Float64)
    if e1z < -0.999999999
        e2x, e2y, e2z = 0.0, -1.0, 0.0
        e3x, e3y, e3z = -1.0, 0.0, 0.0
    else
        a = 1.0 / (1.0 + e1z)
        b = -e1x * e1y * a
        e2x = 1.0 - e1x * e1x * a
        e2y = b
        e2z = -e1x
        e3x = b
        e3y = 1.0 - e1y * e1y * a
        e3z = -e1y
    end
    return e2x, e2y, e2z, e3x, e3y, e3z
end

"""
    compute_lenard_balescu_kernel!(r, vx_i, vy_i, vx_j, vy_j,
                                   G, HG, I_integral,
                                   r0, dr, inv_dr, len_r,
                                   k_parallel, k_perp, norm_k, approx,
                                   phi_p_11, phi_p_22, phi_p_12;
                                   static_alpha=NaN)

Fused kernel: computes rotated k-vectors, dielectric response, and kernel
tensor in a single pass over the k-space grid.

When `static_alpha` is not `NaN`, uses the static dielectric
ε_static(k) = 1 + static_alpha * V̂(k) instead of the dynamic G/HG interpolation.

Returns the symmetric 2×2 tensor entries `(Kxx, Kxy, Kyy)` for the 2D
Lenard-Balescu path.
"""
function compute_lenard_balescu_kernel!(r::Float64, vx_i::Float64, vy_i::Float64, vx_j::Float64, vy_j::Float64,
                                       G::AbstractArray, HG::AbstractArray, I_integral::Float64,
                                       r0::Float64, dr::Float64, inv_dr::Float64, len_r::Int,
                                       k_parallel::AbstractMatrix, k_perp::AbstractMatrix,
                                       norm_k::AbstractMatrix, approx::AbstractMatrix,
                                       phi_p_11::AbstractMatrix, phi_p_22::AbstractMatrix, phi_p_12::AbstractMatrix,
                                       k_angle_local::AbstractMatrix;
                                       static_alpha::Float64=NaN,
                                       constant_fourier::Bool=false)
    if r < 1e-10
        return 0.0, 0.0, 0.0
    end

    vx_rel = vx_i - vx_j
    vy_rel = vy_i - vy_j
    v_rel_mag = sqrt(vx_rel^2 + vy_rel^2)
    if v_rel_mag < 1e-10
        return 0.0, 0.0, 0.0
    end

    cos_theta = vx_rel / v_rel_mag
    sin_theta = vy_rel / v_rel_mag
    cos_sq = cos_theta * cos_theta
    sin_sq = sin_theta * sin_theta
    sin_cos = sin_theta * cos_theta

    theta_vrel = atan(vy_rel, vx_rel)

    S_const = 2.0 * π
    r_first = r0
    r_last = r0 + (len_r - 1) * dr

    kernel_xx = 0.0
    kernel_xy = 0.0
    kernel_yy = 0.0

    @inbounds for idx in eachindex(norm_k)
        k_norm = norm_k[idx]
        if k_norm < 1e-15
            continue
        end

        kp = k_parallel[idx]
        kperp = k_perp[idx]
        ckx = cos_theta * kp - sin_theta * kperp
        cky = sin_theta * kp + cos_theta * kperp

        v_par = (ckx * vx_i + cky * vy_i) / k_norm

        V_sq = constant_fourier ? 1.0 : 1.0 / (k_norm * k_norm + 1e-15)
        local denom::Float64
        if isnan(static_alpha)
            # Dynamic dielectric: interpolate G/HG → ε(k, v∥)
            if ndims(G) == 1
                G_interp = interpolate_profile_line(G, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG_interp = interpolate_profile_line(HG, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
            else
                nang = size(G, 1)
                ith0, ith1, αθ = angle_to_periodic_weights(k_angle_local[idx] + theta_vrel, nang)

                G0 = interp_row(G, ith0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                G1 = interp_row(G, ith1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG0 = interp_row(HG, ith0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG1 = interp_row(HG, ith1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)

                G_interp = muladd(αθ, G1 - G0, G0)
                HG_interp = muladd(αθ, HG1 - HG0, HG0)
            end

            if ndims(G) == 1
                I_real = π * v_par * HG_interp - S_const * I_integral
                I_imag = π * v_par * G_interp
            else
                I_real = π * HG_interp
                I_imag = π * G_interp
            end

            eps_real = 1.0 + V_sq * I_real
            eps_imag = V_sq * I_imag
            denom = max(eps_real * eps_real + eps_imag * eps_imag, 1e-10)
        else
            # Static dielectric: ε_static(k) = 1 + α V̂(k),  V̂(k) = 1/|k|²  (Coulomb)
            eps_s = 1.0 + static_alpha * V_sq
            denom = max(eps_s * eps_s, 1e-10)
        end

        weight = approx[idx]
        p11 = phi_p_11[idx]
        p22 = phi_p_22[idx]
        p12 = phi_p_12[idx]

        phi_k_11 = p11 * cos_sq + p22 * sin_sq - 2.0 * p12 * sin_cos
        phi_k_22 = p11 * sin_sq + p22 * cos_sq + 2.0 * p12 * sin_cos
        phi_k_12 = (p11 - p22) * sin_cos + p12 * (cos_sq - sin_sq)

        kernel_xx += (phi_k_11 / denom) * weight
        kernel_xy += (phi_k_12 / denom) * weight
        kernel_yy += (phi_k_22 / denom) * weight
    end

    inv_v_rel = 1.0 / v_rel_mag
    return kernel_xx * inv_v_rel, kernel_xy * inv_v_rel, kernel_yy * inv_v_rel
end

"""
    compute_lenard_balescu_kernel_3d!(...)

Compute the 3×3 Lenard-Balescu kernel tensor in 3D using precomputed k-space
samples in a local frame aligned with `v_i - v_j`.

Returns the 9 tensor entries in row-major order.
"""
function compute_lenard_balescu_kernel_3d!(r::Float64,
                                           vx_i::Float64, vy_i::Float64, vz_i::Float64,
                                           vx_j::Float64, vy_j::Float64, vz_j::Float64,
                                           G::AbstractArray, HG::AbstractArray, I_integral::Float64,
                                           r0::Float64, dr::Float64, inv_dr::Float64, len_r::Int,
                                           k_parallel::AbstractVector,
                                           k_perp2::AbstractVector,
                                           k_perp3::AbstractVector,
                                           norm_k::AbstractVector,
                                           approx::AbstractVector,
                                           inv_norm_k4::AbstractVector;
                                           static_alpha::Float64=NaN,
                                           constant_fourier::Bool=false)
    # Guard near-singular relative distances.
    if r < 1e-10
        return 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0
    end

    vx_rel = vx_i - vx_j
    vy_rel = vy_i - vy_j
    vz_rel = vz_i - vz_j
    v_rel_mag = sqrt(vx_rel^2 + vy_rel^2 + vz_rel^2)
    if v_rel_mag < 1e-10
        return 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0
    end

    # Build local orthonormal frame with e1 aligned to v_i - v_j.
    inv_v_rel = 1.0 / v_rel_mag
    e1x = vx_rel * inv_v_rel
    e1y = vy_rel * inv_v_rel
    e1z = vz_rel * inv_v_rel
    e2x, e2y, e2z, e3x, e3y, e3z = orthonormal_basis_from_e1(e1x, e1y, e1z)

    S_const = 4.0 * π  # S_2 = 4π
    r_first = r0
    r_last = r0 + (len_r - 1) * dr

    k11 = 0.0; k12 = 0.0; k13 = 0.0
    k22 = 0.0; k23 = 0.0; k33 = 0.0

    # Main k-space quadrature loop: dielectric response + tensor accumulation.
    @inbounds for idx in eachindex(norm_k)
        kn = norm_k[idx]
        if kn < 1e-15
            continue
        end

        kp = k_parallel[idx]
        k2 = k_perp2[idx]
        k3 = k_perp3[idx]

        # Map local k to Cartesian frame
        ckx = kp * e1x + k2 * e2x + k3 * e3x
        cky = kp * e1y + k2 * e2y + k3 * e3y
        ckz = kp * e1z + k2 * e2z + k3 * e3z

        # Project particle velocity onto current k-direction.
        v_par = (ckx * vx_i + cky * vy_i + ckz * vz_i) / kn

        # Build |ε(k, v_parallel)|^2 from interpolated G/HG values.
        V_sq = constant_fourier ? 1.0 : 1.0 / (kn * kn + 1e-15)
        local denom::Float64
        if isnan(static_alpha)
            # Dynamic dielectric: interpolate G/HG → ε(k, v∥)
            # Radial mode is 1D; generic mode is bilinear in (μ, φ) + linear in v_parallel.
            if ndims(G) == 1
                G_interp = interpolate_profile_line(G, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG_interp = interpolate_profile_line(HG, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
            else
                nmu = size(G, 1)
                nphi = size(G, 2)
                im0, im1, αμ, ip0, ip1, αφ = mu_phi_bilinear_weights(ckx, cky, ckz, nmu, nphi)

                G00 = interpolate_directional_profile(G, im0, ip0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                G01 = interpolate_directional_profile(G, im0, ip1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                G10 = interpolate_directional_profile(G, im1, ip0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                G11 = interpolate_directional_profile(G, im1, ip1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)

                HG00 = interpolate_directional_profile(HG, im0, ip0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG01 = interpolate_directional_profile(HG, im0, ip1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG10 = interpolate_directional_profile(HG, im1, ip0, v_par, r0, dr, inv_dr, len_r, r_first, r_last)
                HG11 = interpolate_directional_profile(HG, im1, ip1, v_par, r0, dr, inv_dr, len_r, r_first, r_last)

                G0 = muladd(αφ, G01 - G00, G00)
                G1 = muladd(αφ, G11 - G10, G10)
                G_interp = muladd(αμ, G1 - G0, G0)

                HG0 = muladd(αφ, HG01 - HG00, HG00)
                HG1 = muladd(αφ, HG11 - HG10, HG10)
                HG_interp = muladd(αμ, HG1 - HG0, HG0)
            end

            if ndims(G) == 1
                I_real = π * v_par * HG_interp - S_const * I_integral
                I_imag = π * v_par * G_interp
            else
                I_real = π * HG_interp
                I_imag = π * G_interp
            end

            eps_real = 1.0 + V_sq * I_real
            eps_imag = V_sq * I_imag
            denom = max(eps_real * eps_real + eps_imag * eps_imag, 1e-10)
        else
            # Static dielectric: ε_static(k) = 1 + α V̂(k),  V̂(k) = 1/|k|²  (Coulomb)
            eps_s = 1.0 + static_alpha * V_sq
            denom = max(eps_s * eps_s, 1e-10)
        end

        w = approx[idx]
        invk4 = inv_norm_k4[idx]

        # Geometric projector Phi = (k k^T) / |k|^4.
        p11 = ckx * ckx * invk4
        p12 = ckx * cky * invk4
        p13 = ckx * ckz * invk4
        p22 = cky * cky * invk4
        p23 = cky * ckz * invk4
        p33 = ckz * ckz * invk4

        # Weighted contribution to the symmetric 3×3 kernel tensor.
        fac = w / denom
        k11 += p11 * fac
        k12 += p12 * fac
        k13 += p13 * fac
        k22 += p22 * fac
        k23 += p23 * fac
        k33 += p33 * fac
    end

    # Final 1/|v_i-v_j| factor from operator definition.
    return k11 * inv_v_rel, k12 * inv_v_rel, k13 * inv_v_rel,
           k12 * inv_v_rel, k22 * inv_v_rel, k23 * inv_v_rel,
           k13 * inv_v_rel, k23 * inv_v_rel, k33 * inv_v_rel
end
