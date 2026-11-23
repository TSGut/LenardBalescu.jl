"""
    component_grid_views(Vgrid::AbstractArray{Float64}, par::Params)

Return a tuple of component-wise views of `Vgrid` along its last dimension.

For `par.dim == 2`, returns `(Vx, Vy)`; for `par.dim == 3`, returns `(Vx, Vy, Vz)`.
"""
@inline function component_grid_views(Vgrid::AbstractArray{Float64}, par::Params)
    @assert ndims(Vgrid) == par.dim + 1 "Vgrid must have par.dim + 1 dimensions"
    @assert size(Vgrid, ndims(Vgrid)) == par.dim "Last Vgrid dimension must equal par.dim"
    return ntuple(d -> selectdim(Vgrid, ndims(Vgrid), d), par.dim)
end

"""
    build_velocity_grid(par::Params)

Construct a 2D or 3D velocity grid for the simulation.

# Arguments
- `par::Params`: Simulation parameters, must contain `Lv` (velocity limit),
  `Nv` (number of grid cells per dimension), and `dim` (`2` or `3`).

# Returns
- `V`: Vector of velocity grid edges
- `dV`: Grid cell width
- `Vc`: Vector of velocity grid cell centers
- `Vgrid`: Tensor grid of velocity-cell centres, with component index in the last dimension
"""
function build_velocity_grid(par::Params)
    @assert par.dim in (2, 3) "Params.dim must be 2 or 3 (got=$(par.dim))"
    V = collect(range(-par.Lv, stop=par.Lv, length=par.Nv + 1))
    dV = V[2] - V[1]
    Vc = V[1:end-1] .+ dV / 2.0

    if par.dim == 2
        Vgrid = zeros(Float64, par.Nv, par.Nv, 2)
        for j in 1:par.Nv
            for i in 1:par.Nv
                Vgrid[i, j, 1] = Vc[j]
                Vgrid[i, j, 2] = Vc[i]
            end
        end
        return V, dV, Vc, Vgrid
    else
        Vgrid = zeros(Float64, par.Nv, par.Nv, par.Nv, 3)
        for k in 1:par.Nv
            for j in 1:par.Nv
                for i in 1:par.Nv
                    Vgrid[i, j, k, 1] = Vc[j]
                    Vgrid[i, j, k, 2] = Vc[i]
                    Vgrid[i, j, k, 3] = Vc[k]
                end
            end
        end
        return V, dV, Vc, Vgrid
    end
end

"""
    L1_error(a::AbstractVector{Float64}, b::AbstractVector{Float64}, dV::Float64; dim::Int=2)

Computes the L1 error (sum of absolute differences) between two distributions on a 2D or 3D grid.

# Arguments
- `a`, `b`: Vectors of distribution values (must be same length)
- `dV`: Grid cell width
- `dim`: Velocity-space dimension used in volume scaling (`dV^dim`, default `2`)

# Returns
- L1 error as a Float64
"""
function L1_error(a::AbstractVector{Float64}, b::AbstractVector{Float64}, dV::Float64; dim::Int=2)
    return sum(abs.(a .- b)) * dV^dim
end

"""
    L2_error(a::AbstractVector{Float64}, b::AbstractVector{Float64}, dV::Float64; dim::Int=2)

Computes the L2 error norm between two distributions on a 2D or 3D grid.

# Arguments
- `a`, `b`: Vectors of distribution values (must be same length)
- `dV`: Grid cell width
- `dim`: Velocity-space dimension used in volume scaling (`dV^dim`, default `2`)

# Returns
- L2 error as a Float64
"""
function L2_error(a::AbstractVector{Float64}, b::AbstractVector{Float64}, dV::Float64; dim::Int=2)
    return sqrt(sum((a .- b) .^ 2) * dV^dim)
end
