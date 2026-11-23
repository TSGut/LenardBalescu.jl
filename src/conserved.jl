"""
    compute_conserved_quantities(vstack, wg, par)

Compute total mass, momentum (as a vector), and energy from particle velocities and weights.

# Arguments
- `vstack`: `N×dim` matrix of particle velocities (`dim = par.dim`, supports 2 or 3)
- `wg`: Vector of particle weights
- `par`: Simulation parameters (must have field `m` for particle mass)

# Returns
Named tuple with fields:
- `mass`: Total mass (sum of weights)
- `momentum`: Total momentum vector (`2` or `3` components depending on `par.dim`)
- `energy`: Total kinetic energy (sum of weighted `0.5*m*|v|^2`)
"""
function compute_conserved_quantities(vstack::AbstractMatrix{Float64}, wg::AbstractVector{Float64}, par::Params)
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3"
    mass = sum(wg)
    if par.dim == 2
        vx = vstack[:, 1]
        vy = vstack[:, 2]
        momentum = [sum(wg .* vx), sum(wg .* vy)]
        energy = sum(wg .* 0.5 .* par.m .* (vx.^2 .+ vy.^2))
    else # par.dim == 3
        vx = vstack[:, 1]
        vy = vstack[:, 2]
        vz = vstack[:, 3]
        momentum = [sum(wg .* vx), sum(wg .* vy), sum(wg .* vz)]
        energy = sum(wg .* 0.5 .* par.m .* (vx.^2 .+ vy.^2 .+ vz.^2))
    end
    return (mass=mass, momentum=momentum, energy=energy)
end

"""
    compute_entropy(f_grid, dV; dim=2)

Compute entropy from a distribution f_grid and grid spacing dV.

# Arguments
- `f_grid`: Smoothed distribution values on the velocity grid
- `dV`: Grid spacing in one velocity dimension
- `dim`: Velocity-space dimension used for volume scaling (`dV^dim`)

# Returns
- Approximated entropy value `S = ∫ f log(f) dV`.
"""
function compute_entropy(f_grid::AbstractVector, dV::Float64; dim::Int=2)
    small = eps(Float64)
    return dV^dim * sum(f_grid .* log.(max.(f_grid, small)))
end