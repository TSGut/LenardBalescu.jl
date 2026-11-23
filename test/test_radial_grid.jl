using Test
using LenardBalescu

@testset "Mesh-width radial shells" begin
    for dim in (2, 3), Nv in (5, 8, 13, 30)
        par = SetParam()
        par.dim = dim
        par.Nv = Nv
        update_kernel_params!(par)
        _, h, Vc, Vgrid = build_velocity_grid(par)
        edges = LenardBalescu.radial_bin_edges(par)
        n = length(edges) - 1
        @test all(isapprox.(diff(edges), h))
        @test last(edges) >= par.Lv * sqrt(dim)
        @test edges[end-1] < par.Lv * sqrt(dim)

        grids = LenardBalescu.component_grid_views(Vgrid, par)
        radii = vec(sqrt.(sum(g .^ 2 for g in grids)))
        values = exp.(-radii .^ 2)
        r, f = compute_radial_binned_f(values, Vgrid, par)
        @test length(r) == 2n
        @test all(isapprox.(diff(r), h))
        @test f ≈ reverse(f)
        for q in 0:n-1
            members = findall(x -> edges[q+1] <= x < edges[q+2], radii)
            expected = isempty(members) ? 0.0 : sum(values[members]) / length(members)
            @test f[n+q+1] ≈ expected atol=1e-14
        end

        r0, dr, inv_dr, len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=true)
        @test r0 ≈ first(r)
        @test len == 2length(r) - 1
        @test r0 + (len-1)*dr ≈ last(r)
        @test inv_dr ≈ 1/dr
        if dim == 2
            r_compat, f_compat = compute_radial_binned_f(values, Vc, par)
            @test r_compat == r
            @test f_compat ≈ f
        end

        old_edges = LenardBalescu.radial_bin_edges(par; n_bins=15)
        @test length(old_edges) == 16
        @test last(old_edges) ≈ par.Lv * sqrt(dim)
        r_old, f_old = compute_radial_binned_f(values, Vgrid, par; n_bins=15)
        @test length(r_old) == length(f_old) == 30
    end
end
