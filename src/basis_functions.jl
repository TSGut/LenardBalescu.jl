"""
    BasisKernel(order::Int, e::Float64)

Basis function descriptor for particle-in-cell (PIC) velocity mesh coupling.

Supports multiple basis function orders:
- `0`: Gaussian kernel
- `1`: Linear B-spline
- `2`: Quadratic B-spline
- `3`: Cubic B-spline

Each basis function `ψ` and its gradient `∇ψ` are used to map between particles and the velocity grid.

# Arguments
- `order`: Kernel order (0=Gaussian, 1=linear, 2=quadratic, 3=cubic)
- `e`: Kernel scale parameter (Gaussian width parameter; dilation scale for B-splines)
"""
struct BasisKernel
    order::Int
    e::Float64
end

"""
    safe_e(k::BasisKernel)

Lower-bound the kernel scale parameter to avoid numerical issues when `e` is too small.
"""
safe_e(k::BasisKernel) = max(k.e, 1e-4)

"""
    G1(x::Float64)

Linear B-spline (hat) function: triangular support, linear falloff (order=1).
"""
G1(x::Float64) = (abs(x) <= 1.0) ? (1.0 - abs(x)) : 0.0

"""
    dG1(x::Float64)

Derivative of the linear B-spline (hat) function.
"""
function dG1(x::Float64)
    if x > 0.0 && x <= 1.0
        return -1.0
    elseif x < 0.0 && x >= -1.0
        return 1.0
    end
    return 0.0
end


"""
    B2(x::Float64)

Quadratic B-spline basis function (order=2).
"""
function B2(x::Float64)
    ax = abs(x)
    if ax < 0.5
        return 0.75 - x * x
    elseif ax < 1.5
        t = 1.5 - ax
        return 0.5 * t * t
    end
    return 0.0
end

"""
    dB2(x::Float64)

Derivative of the quadratic B-spline basis function.
"""
function dB2(x::Float64)
    ax = abs(x)
    if ax < 0.5
        return -2.0 * x
    elseif ax < 1.5
        s = (x >= 0.0) ? 1.0 : -1.0
        t = 1.5 - ax
        return -t * s
    end
    return 0.0
end


"""
    B3(x::Float64)

Cubic B-spline basis function (order=3).
"""
function B3(x::Float64)
    ax = abs(x)
    if ax < 1.0
        return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0
    elseif ax < 2.0
        t = 2.0 - ax
        return (t * t * t) / 6.0
    end
    return 0.0
end

"""
    dB3(x::Float64)

Derivative of the cubic B-spline basis function.
"""
function dB3(x::Float64)
    ax = abs(x)
    s = (x >= 0.0) ? 1.0 : -1.0
    if ax < 1.0
        return (-12.0 * x + 9.0 * ax * x) / 6.0
    elseif ax < 2.0
        t = 2.0 - ax
        return -0.5 * t * t * s
    end
    return 0.0
end

# ========== Generic interface functions ==========

function psi(k::BasisKernel, qx::Float64, qy::Float64)
    if k.order == 0
        ee = safe_e(k)
        r2 = qx * qx + qy * qy
        pref = 1.0 / (2.0 * pi * ee)
        return pref * exp(-r2 / (2.0 * ee))
    elseif k.order == 1
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        return (G1(x) * G1(y)) / (ee * ee)
    elseif k.order == 2
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        return (B2(x) * B2(y)) / (ee * ee)
    else
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        return (B3(x) * B3(y)) / (ee * ee)
    end
end

function psi(k::BasisKernel, qx::Float64, qy::Float64, qz::Float64)
    if k.order == 0
        ee = safe_e(k)
        r2 = qx * qx + qy * qy + qz * qz
        pref = 1.0 / ((2.0 * pi * ee)^(3.0 / 2.0))
        return pref * exp(-r2 / (2.0 * ee))
    elseif k.order == 1
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        z = qz / ee
        return (G1(x) * G1(y) * G1(z)) / (ee * ee * ee)
    elseif k.order == 2
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        z = qz / ee
        return (B2(x) * B2(y) * B2(z)) / (ee * ee * ee)
    else
        ee = safe_e(k)
        x = qx / ee
        y = qy / ee
        z = qz / ee
        return (B3(x) * B3(y) * B3(z)) / (ee * ee * ee)
    end
end

function grad(k::BasisKernel, qx::Float64, qy::Float64)
    if k.order == 0
        ee = safe_e(k)
        r2 = qx * qx + qy * qy
        pref = 1.0 / (2.0 * pi * ee)
        p = pref * exp(-r2 / (2.0 * ee))
        gx = -(qx / ee) * p
        gy = -(qy / ee) * p
        return gx, gy
    end
    ee = safe_e(k)
    x = qx / ee
    y = qy / ee
    if k.order == 1
        gx1 = dG1(x)
        gy1 = dG1(y)
        gx = (gx1 * G1(y)) / (ee * ee * ee)
        gy = (G1(x) * gy1) / (ee * ee * ee)
        return gx, gy
    elseif k.order == 2
        dbx = dB2(x)
        dby = dB2(y)
        gx = (dbx * B2(y)) / (ee * ee * ee)
        gy = (B2(x) * dby) / (ee * ee * ee)
        return gx, gy
    else
        dbx = dB3(x)
        dby = dB3(y)
        gx = (dbx * B3(y)) / (ee * ee * ee)
        gy = (B3(x) * dby) / (ee * ee * ee)
        return gx, gy
    end
end

function grad(k::BasisKernel, qx::Float64, qy::Float64, qz::Float64)
    if k.order == 0
        ee = safe_e(k)
        p = psi(k, qx, qy, qz)
        gx = -(qx / ee) * p
        gy = -(qy / ee) * p
        gz = -(qz / ee) * p
        return gx, gy, gz
    end

    ee = safe_e(k)
    x = qx / ee
    y = qy / ee
    z = qz / ee

    if k.order == 1
        gx = (dG1(x) * G1(y) * G1(z)) / (ee^4)
        gy = (G1(x) * dG1(y) * G1(z)) / (ee^4)
        gz = (G1(x) * G1(y) * dG1(z)) / (ee^4)
        return gx, gy, gz
    elseif k.order == 2
        gx = (dB2(x) * B2(y) * B2(z)) / (ee^4)
        gy = (B2(x) * dB2(y) * B2(z)) / (ee^4)
        gz = (B2(x) * B2(y) * dB2(z)) / (ee^4)
        return gx, gy, gz
    else
        gx = (dB3(x) * B3(y) * B3(z)) / (ee^4)
        gy = (B3(x) * dB3(y) * B3(z)) / (ee^4)
        gz = (B3(x) * B3(y) * dB3(z)) / (ee^4)
        return gx, gy, gz
    end
end

function neighbor_radius_cells(k::BasisKernel, cell_h::Float64; gaussian_cutoff_sigma::Float64=3.0)
    """Compute support radius in grid cells for efficient neighbor search"""
    ee = safe_e(k)
    if k.order == 0
        # Gaussian: truncate at configurable σ multiple
        cutoff = max(gaussian_cutoff_sigma, 0.0) * sqrt(ee)
        rc = Int(ceil(cutoff / cell_h))
        return max(0, rc)
    elseif k.order == 1
        # Linear: support is [-e, +e]
        return max(1, Int(ceil(ee / cell_h)))
    elseif k.order == 2
        # Quadratic: support is [-1.5e, +1.5e]
        return max(1, Int(ceil(1.5 * ee / cell_h)))
    else
        # Cubic: support is [-2e, +2e]
        return max(1, Int(ceil(2.0 * ee / cell_h)))
    end
end

"""
    fit_bspline(r_centers, f_binned)

Fit a cubic B-spline interpolation to sampled 1D data.

Returns spline metadata and callable interpolation/derivative functions with flat extrapolation.
"""
function fit_bspline(r_centers::AbstractVector, f_binned::AbstractVector)
    n = length(r_centers)
    if n < 4
        error("Need at least 4 points for cubic spline interpolation")
    end
    
    r_min = minimum(r_centers)
    r_max = maximum(r_centers)

    itp = interpolate(f_binned, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, r_centers)

    etp = extrapolate(sitp, Flat())
    
    function f_interp(r::Real)
        return etp(r)
    end
    
    function f_deriv(r::Real)
        return Interpolations.gradient(etp, r)[1]
    end
    
    coeffs = (x=r_centers, y=f_binned, itp=sitp)
    return coeffs, r_min, r_max, f_interp, f_deriv
end

"""
    gauss_legendre_quadrature(n::Int, a::Float64, b::Float64)

Compute Gauss-Legendre quadrature nodes and weights on interval [a, b].

# Arguments
- `n`: Number of quadrature points
- `a`, `b`: Integration interval endpoints

# Returns
- `nodes::Vector`: Quadrature nodes
- `weights::Vector`: Quadrature weights
"""
function gauss_legendre_quadrature(n::Int, a::Float64, b::Float64)
    # Golub-Welsch algorithm: nodes and weights from eigendecomposition of
    # the symmetric tridiagonal Jacobi matrix for Legendre polynomials.
    # Diagonal is zero; off-diagonal entries are sqrt(k^2 / (4k^2 - 1)).
    dv = zeros(n)
    ev = [sqrt(i * i / (4.0 * i * i - 1.0)) for i in 1:(n-1)]

    J = SymTridiagonal(dv, ev)
    E = eigen(J)
    perm = sortperm(E.values)
    nodes_unit = E.values[perm]

    # Weights: w_i = scal * (first component of i-th eigenvector)^2
    # scal = integral of weight function over [-1,1] = 2 for Legendre
    weights_unit = [2.0 * E.vectors[1, p]^2 for p in perm]

    # Transform from [-1, 1] to [a, b]
    nodes = (a + b) / 2.0 .+ (b - a) / 2.0 .* nodes_unit
    weights = (b - a) / 2.0 .* weights_unit

    return nodes, weights
end

"""
    hat(x::AbstractVector, eps::Float64)

Triangular hat function used to approximate delta in k_parallel.
"""
function hat(x::AbstractVector, eps::Float64)
    return max.(1.0 .- abs.(x) ./ eps, 0.0) ./ eps
end