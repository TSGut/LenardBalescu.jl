"""
    3D_benchmark_threading.jl

Automatically tests all thread counts (1 to system max) and plots timing results
for 3D Lenard-Balescu collision timings.
"""

using Statistics, Random
using Plots, Measures

project_root = normpath(joinpath(@__DIR__, ".."))

if haskey(ENV, "BENCHMARK_THREAD_COUNT")
    include(normpath(joinpath(@__DIR__, "..", "src", "LenardBalescu.jl")))
    using .LenardBalescu

    n_threads = parse(Int, ENV["BENCHMARK_THREAD_COUNT"])
    results = Dict{Int,Float64}()

    for N_test in [80, 160, 320, 640, 1280, 2560]
        par = SetParam()
        par.dim = 3
        par.Nv = 9
        par.k_perp_order = 24
        par.k_perp_max = 20.0
        par.k_parallel_width = 0.05
        par.k_parallel_order = 3
        update_kernel_params!(par)

        _, _, _, Vgrid = build_velocity_grid(par)
        dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=false)

        times_lb = Float64[]
        for _ in 1:3
            rng = MersenneTwister(42)
            vstack = rand(rng, N_test, 3) .* (2 * par.Lv) .- par.Lv
            wg = fill(1.0 / N_test, N_test)
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                radial_symmetry=false,
                verbose=0,
                dielectric_r0,
                dielectric_dr,
                dielectric_inv_dr,
                dielectric_len)
            push!(times_lb, elapsed[])
        end
        results[N_test] = median(times_lb)
    end

    for N in [80, 160, 320, 640, 1280, 2560]
        println("RESULT:$n_threads:$N:$(results[N])")
    end

else
    max_threads_system = Sys.CPU_THREADS
    max_threads_bench = tryparse(Int, get(ENV, "BENCHMARK_MAX_THREADS", ""))
    max_threads = max_threads_bench !== nothing ? max_threads_bench : max_threads_system

    println("\n3D THREADING PERFORMANCE BENCHMARK")
    println("System CPU threads: $max_threads_system")
    println("Testing thread counts: 1 to $max_threads\n")

    all_results = Dict{Int,Dict{Int,Float64}}()

    for n_threads in 1:max_threads
        print("Benchmarking with $n_threads thread(s)... ")
        flush(stdout)

        script_path = joinpath(project_root, "benchmarks", "3D_benchmark_threading.jl")
        julia_exe = Base.julia_cmd().exec[1]
        cmd = Cmd([julia_exe, "--project=$project_root", "-t", "$n_threads", "-e", "ENV[\"BENCHMARK_THREAD_COUNT\"] = \"$n_threads\"; include(\"$script_path\")"])
        output = read(cmd, String)

        all_results[n_threads] = Dict{Int,Float64}()
        for line in split(output, "\n")
            if startswith(line, "RESULT:")
                parts = split(line, ":")
                if length(parts) == 4
                    n_test = parse(Int, parts[3])
                    time_val = parse(Float64, parts[4])
                    all_results[n_threads][n_test] = time_val
                end
            end
        end

        println("✓")
    end

    println("\n3D LENARD-BALESCU COLLISION TIME RESULTS\n")

    local header = "N particles"
    for t in 1:max_threads
        header *= " | $t-thread (s)"
    end
    println(header)
    println("-" ^ (length(header) + 10))

    for N in [80, 160, 320, 640, 1280, 2560]
        row = "$N"
        for t in 1:max_threads
            time_val = all_results[t][N]
            row *= " | $(lpad(round(time_val, digits=4), 12))"
        end
        println(row)
    end

    println("\nSPEEDUP ANALYSIS (vs 1 thread)\n")

    local header = "N particles"
    for t in 1:max_threads
        header *= " | $(t)x speedup"
    end
    println(header)
    println("-" ^ (length(header) + 10))

    for N in [80, 160, 320, 640, 1280, 2560]
        t1_time = all_results[1][N]
        row = "$N"
        for t in 1:max_threads
            t_time = all_results[t][N]
            speedup = t1_time / t_time
            efficiency = (speedup / t) * 100
            row *= " | $(lpad(round(speedup, digits=2), 3))x ($(lpad(round(efficiency, digits=0), 3))%)"
        end
        println(row)
    end

    println("\nGenerating 3D threading complexity plot...")

    N_values = [80, 160, 320, 640, 1280, 2560]
    p = plot(xscale=:log10, yscale=:log10,
        xlabel="N (particles)", ylabel="Time (s)",
        title="3D LB Threading: Complexity",
        grid=false, legend=:topleft, size=(900, 500), widen=true,
        bottom_margin=5mm, left_margin=5mm)

    for t in 1:max_threads
        times_t = [all_results[t][N] for N in N_values]
        logN_sub = log.(float.(N_values[2:end]))
        logT_sub = log.(times_t[2:end])
        slope = cov(logN_sub, logT_sub) / var(logN_sub)
        suffix = t > 1 ? "s" : ""

        plot!(p, N_values, times_t;
            marker=:circle, markersize=6,
            label="$(t) thread$(suffix) (slope=$(round(slope, digits=2)))",
            linewidth=2)
    end

    ref_n = N_values
    ref_n1 = all_results[1][N_values[1]] * (ref_n ./ ref_n[1])
    ref_n2 = all_results[1][N_values[1]] * (ref_n ./ ref_n[1]).^2
    plot!(p, ref_n, ref_n1; linestyle=:dot, linewidth=1.5, label="O(N)", color=:black, alpha=0.7)
    plot!(p, ref_n, ref_n2; linestyle=:dash, linewidth=1.5, label="O(N²)", color=:black, alpha=0.7)

    benchmark_dir = joinpath(@__DIR__, "output", "benchmark")
    mkpath(benchmark_dir)
    output_file = joinpath(benchmark_dir, "benchmark_threading_3d.png")
    savefig(p, output_file)

    println("3D threading benchmark complete. Plot saved to: benchmarks/output/benchmark/benchmark_threading_3d.png\n")
end
