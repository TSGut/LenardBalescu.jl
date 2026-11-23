# Install ProfileView in the active environment before running this script:
# julia --project=. -e 'using Pkg; Pkg.add("ProfileView")'
using LenardBalescu
using Random
using Profile

function short_lb_profile_3d(; ntot=10, seed=1,
                             radial_symmetry=false,
                             out_profile=joinpath(@__DIR__, "output", "benchmark", "profile_lenard_balescu_3d_maxwell.txt"))
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 3
    par.Nv = 9
    par.Npc = 1
    par.N = par.Nv^par.dim * par.Npc
    par.dt = 0.01
    par.T = ntot * par.dt
    par.ntot = ntot
    par.k_perp_order = 24
    par.k_perp_max = 20.0
    par.k_parallel_width = 0.05
    par.k_parallel_order = 3
    update_kernel_params!(par)

    outdir = joinpath(@__DIR__, "output", "lenard_balescu_3d_profile")

    run_simulation(; par=par, rng=rng,
        use_lenard_balescu=true,
        radial_symmetry=radial_symmetry,
        outdir=outdir,
        step_callback=nothing,
        verbose=0)

    Profile.clear()
    Profile.init()

    Profile.@profile run_simulation(; par=par, rng=rng,
        use_lenard_balescu=true,
        radial_symmetry=radial_symmetry,
        outdir=outdir,
        step_callback=nothing,
        verbose=0)

    mkpath(dirname(out_profile))
    open(out_profile, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, maxdepth=20)
    end

    println("Profile saved to: $(out_profile)")
    println("Top profile entries (flat):")
    Profile.print(; format=:flat, sortedby=:count, maxdepth=20)

    return nothing
end

short_lb_profile_3d(ntot=10, seed=1)

# Display the profile interactively.
using ProfileView
ProfileView.view()
