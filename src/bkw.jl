"""
    BKW_vector(t, T_tot, C_BKW, B_BKW, Vgrid)

Bobylev-Krook-Wu (BKW) exact solution for the homogeneous Boltzmann equation on a 2D or 3D velocity grid.

# Arguments
- `t`: Time
- `T_tot`: Total temperature
- `C_BKW`: Amplitude parameter
- `B_BKW`: Decay rate
- `Vgrid`: tensor grid of velocity-cell centres, with components stored in the last dimension

# Returns
- `M`-vector of distribution-function values
"""
function BKW_vector(t::Float64, T_tot::Float64, C_BKW::Float64, B_BKW::Float64,
                    Vgrid::AbstractArray{Float64})
    grid_dim = size(Vgrid, ndims(Vgrid))
    @assert grid_dim in (2, 3) "BKW_vector requires a 2D or 3D tensor velocity grid"
    @assert ndims(Vgrid) == grid_dim + 1 "BKW_vector requires a tensor velocity grid with components in the last dimension"

    Kval = T_tot * (1.0 - C_BKW * exp(-B_BKW * 2.0 * t))
    m = 1.0
    grid_views = ntuple(d -> selectdim(Vgrid, ndims(Vgrid), d), grid_dim)
    r2 = Vector{Float64}(undef, length(grid_views[1]))
    @inbounds for i in eachindex(grid_views[1])
        r2_i = 0.0
        for d in 1:grid_dim
            r2_i += grid_views[d][i]^2
        end
        r2[i] = r2_i
    end

    denom = (2.0 * pi * Kval) ^ (grid_dim / 2.0)
    c1 = ((grid_dim + 2.0) * Kval - grid_dim * T_tot) / (2.0 * Kval)
    c2 = m * (T_tot - Kval) / (2.0 * Kval ^ 2)
    result = (m / denom) .* exp.(-m .* r2 ./ (2.0 * Kval)) .* (c1 .+ c2 .* r2)
    return result
end
