"""
    Params

Simulation parameters for Lenard-Balescu or Landau kinetic equation simulations.

# Fields
- `gam`, `beta`, `alpha`, `c`, `temp`, `m`: Kernel and physical parameters. `alpha` is the static-dielectric screening parameter (α); set via `par.alpha` and used when `static_dielectric=true` in `run_simulation`/`Collision`.
- `Lv`, `Nv`, `Npc`, `N`: Velocity grid parameters.
- `T`, `dt`, `ntot`: Time stepping parameters.
- `batchsize`: RBM target batch size (`0` disables batching; default full interaction).
    In RBM mode, particles are repartitioned each step into balanced batches. For
    `2 <= batchsize < N`, this can still reduce cost even when `batchsize` is close
    to `N` (for example, it may produce two roughly equal batches and reduce pair
    work from about `N^2` to about `N^2/2`).
- `spline_order`, `use_cell_list`: Spline and collision parameters.
- `k_perp_scale`: Characteristic k scale for sinh-mapped k_perp quadrature (default 1.0 = thermal scale).
- `epsi`, `cell_h`: Derived parameters (set by `update_kernel_params!`).
"""
Base.@kwdef mutable struct Params
    # Kernel parameters
    gam::Float64 = 0.0
    beta::Float64 = 1/10
    alpha::Float64 = 1.0   # Static-dielectric screening parameter (α): ε_static(k) = 1 + α V̂(k)
    c::Float64 = 0.5
    temp::Float64 = 1.0
    m::Float64 = 1.0

    # Velocity grid parameters
    Lv::Float64 = 6.0
    Nv::Int = 30
    Npc::Int = 2
    N::Int = 0
    
    # Simulation dimension: currently supported values are 2 or 3
    dim::Int = 2

    # Interaction potential for Lenard-Balescu kernel: :coulomb (V̂(k)=1/|k|²) or :constant_fourier (V̂(k)=1)
    potential::Symbol = :coulomb

    # K-space integration parameters for Lenard-Balescu kernel
    k_perp_order::Int = 30            # Number of Gauss-Legendre quadrature points (k_perp)
    k_perp_max::Float64 = 50.0        # Maximum k_perp value
    k_parallel_width::Float64 = 0.02  # Hat-delta width for k_parallel
    k_parallel_order::Int = 3         # Number of points for k_parallel (delta approximation)

    # Sinh-mapped GL quadrature: k = k_perp_scale*sinh(t*asinh(k_perp_max/k_perp_scale)).
    k_perp_scale::Float64 = 1.0

    # Angular quadrature points for non-radial dielectric
    nang::Int = 100

    # 3D generic (non-radial) dielectric mesh controls
    # If <= 0, fallback to auto sizing based on Nv/nang.
    dielectric_rho_points_3d::Int = 0
    dielectric_psi_points_3d::Int = 0

    # Time stepping parameters
    T::Float64 = 3.0
    dt::Float64 = 0.01
    ntot::Int = 0
    
    # RBM target batch size: 0/1/full disable batching; otherwise a balanced
    # repartition is applied each step, so near-N targets can still reduce cost since it will re-distribute to 2 batches.
    batchsize::Int = 0

    # Spline and collision parameters
    spline_order::Int = 3
    gaussian_cutoff_sigma::Float64 = 3.0 # Does nothing if spline_order > 0, but controls Gaussian kernel cutoff radius in units of σ when spline_order=0
    use_cell_list::Bool = true

    # Derived parameters (updated by update_kernel_params!)
    epsi::Float64 = 0.0
    cell_h::Float64 = 0.0
end

function is_constant_fourier(potential::Symbol)
    potential == :constant_fourier && return true
    potential == :coulomb && return false
    throw(ArgumentError("Unknown interaction potential: $potential"))
end

"""
    update_kernel_params!(par::Params)

Update the derived kernel parameters (`cell_h` and `epsi`) in a `Params` struct based on the current grid and spline settings.

# Arguments
- `par`: The `Params` struct to update (modified in-place).

# Returns
- The updated `Params` struct (same object, for convenience).
"""
function update_kernel_params!(par::Params)
    par.cell_h = 2.0 * par.Lv / par.Nv
    if par.spline_order == 0
        par.epsi = 0.64 * par.cell_h^1.98
    else
        par.epsi = par.cell_h
    end
    return par
end

"""
    SetParam()

Create and initialize a Params struct with default simulation parameters for Lenard-Balescu or Landau kinetic equation simulations.

This function sets up the default grid, time, and kernel parameters, computes derived quantities, and ensures consistency between fields.

# Returns
- A fully initialized Params struct with all fields set to default values.
"""
function SetParam()
    par = Params()

    par.N = par.Nv ^ par.dim * par.Npc
    par.ntot = Int(ceil(par.T / par.dt))

    update_kernel_params!(par)
    par.use_cell_list = true
    return par
end
