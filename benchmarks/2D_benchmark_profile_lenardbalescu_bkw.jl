# For the interactive viewer, first install ProfileView in the active environment:
# julia --project=. -e 'using Pkg; Pkg.add("ProfileView")'
using LenardBalescu
using Random
using Profile

function short_lb_profile(; ntot=20, seed=1,
                          out_profile=joinpath(@__DIR__, "output", "benchmark", "profile_lenard_balescu_bkw.txt"))
    rng = MersenneTwister(seed)
    par = SetParam()
    par.dim = 2
    par.Nv = 15
    par.Npc = 2
    par.N = par.Nv * par.Nv * par.Npc
    par.dt = 0.01
    par.T = ntot * par.dt
    par.ntot = ntot
    update_kernel_params!(par)

    # Warm up to reduce compilation noise in the profile.
    run_simulation(; par=par, rng=rng, use_lenard_balescu=true,
        outdir=joinpath(@__DIR__, "output", "lenard_balescu_bkw"),
        step_callback=nothing, verbose=0)

    Profile.clear()
    Profile.init()

    Profile.@profile run_simulation(; par=par, rng=rng, use_lenard_balescu=true,
        outdir=joinpath(@__DIR__, "output", "lenard_balescu_bkw"),
        step_callback=nothing, verbose=0)

    mkpath(dirname(out_profile))
    open(out_profile, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, maxdepth=20)
    end

    println("Profile saved to: $(out_profile)")
    println("Top profile entries (flat):")
    Profile.print(; format=:flat, sortedby=:count, maxdepth=20)

    return nothing
end

# Run a short profile by default.
short_lb_profile(ntot=40, seed=1)

# If you have ProfileView.jl installed, you can also view the profile interactively:
try 
    @eval import ProfileView
catch
    @warn "ProfileView.jl not available; skipping interactive profile view"
else
    println("Opening interactive profile view with ProfileView.jl...")
    ProfileView.view()
end
