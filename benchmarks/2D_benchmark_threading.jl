"""
    benchmark_threading.jl

Automatically tests all thread counts (1 to system max) and plots timing results.
"""

using Statistics, Random
using Plots, Measures

# Get the project root
project_root = normpath(joinpath(@__DIR__, ".."))

# Check if we're the controller or a worker process
if haskey(ENV, "BENCHMARK_THREAD_COUNT")
    # WORKER MODE: Run the actual benchmark for a single thread count
    include(normpath(joinpath(@__DIR__, "..", "src", "LenardBalescu.jl")))
    using .LenardBalescu
    
    n_threads = parse(Int, ENV["BENCHMARK_THREAD_COUNT"])
    
    results = Dict()
    
    for N_test in [100, 200, 400, 800]
        par = SetParam()
        par.dim = 2
        par.Nv = 15
        update_kernel_params!(par)
        
        _, dV, Vc, Vgrid = build_velocity_grid(par)
        dielectric_r0, dielectric_dr, dielectric_inv_dr, dielectric_len = precompute_dielectric_lookup_grid(par, Vgrid; radial_symmetry=false)
        vstack, wg = initialize_particles_stratified_BKW(N_test, dV, Vgrid, par, MersenneTwister(42))
        
        # Run 3 trials and take median
        times_lb = []
        for trial in 1:3
            elapsed = Ref(0.0)
            Collision(vstack, par, wg, elapsed;
                Vgrid=Vgrid,
                use_lenard_balescu=true,
                verbose=0,
                dielectric_r0,
                dielectric_dr,
                dielectric_inv_dr,
                dielectric_len)
            push!(times_lb, elapsed[])
        end
        median_time = median(times_lb)
        results[N_test] = median_time
    end
    
    # Output results in a parseable format
    for N in [100, 200, 400, 800]
        println("RESULT:$n_threads:$N:$(results[N])")
    end
    
else
    # CONTROLLER MODE: Spawn workers for different thread counts
    
    # Detect max threads
    max_threads_system = Sys.CPU_THREADS
    max_threads_bench = tryparse(Int, get(ENV, "BENCHMARK_MAX_THREADS", ""))
    max_threads = max_threads_bench !== nothing ? max_threads_bench : max_threads_system
    
    println("\nTHREADING PERFORMANCE BENCHMARK")
    println("System CPU threads: $max_threads_system")
    println("Testing thread counts: 1 to $max_threads\n")
    
    # Collect results from all thread counts
    all_results = Dict()
    
    for n_threads in 1:max_threads
        print("Benchmarking with $n_threads thread(s)... ")
        flush(stdout)
        
        # Spawn Julia process with specific thread count
        script_path = joinpath(project_root, "benchmarks", "2D_benchmark_threading.jl")
        julia_exe = Base.julia_cmd().exec[1]
        cmd = Cmd([julia_exe, "--project=$project_root", "-t", "$n_threads", "-e", "ENV[\"BENCHMARK_THREAD_COUNT\"]=\"$n_threads\"; include(\"$script_path\")"])
        output = read(cmd, String)
        all_results[n_threads] = Dict()
        
        # Parse results
        for line in split(output, "\n")
            if startswith(line, "RESULT:")
                parts = split(line, ":")
                if length(parts) == 4
                    n_t = parse(Int, parts[2])
                    n_test = parse(Int, parts[3])
                    time_val = parse(Float64, parts[4])
                    all_results[n_threads][n_test] = time_val
                end
            end
        end
        
        println("✓")
    end
    
    # Display results table
    println("\nLENARD-BALESCU COLLISION TIME RESULTS\n")
    
    # Header
    local header = "N particles"
    for t in 1:max_threads
        header *= " | $t-thread (s)"
    end
    println(header)
    println("-" ^ (length(header) + 10))
    
    # Data rows
    for N in [100, 200, 400, 800]
        row = "$N"
        for t in 1:max_threads
            time_val = all_results[t][N]
            row *= " | $(lpad(round(time_val, digits=4), 12))"
        end
        println(row)
    end
    
    # Speedup analysis
    println("\nSPEEDUP ANALYSIS (vs 1 thread)\n")
    
    local header = "N particles"
    for t in 1:max_threads
        header *= " | $(t)x speedup"
    end
    println(header)
    println("-" ^ (length(header) + 10))
    
    for N in [100, 200, 400, 800]
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
    
    println("\nGenerating complexity plot...")
    
    # Complexity plot: log-log scaling for each thread count
    N_values = [100, 200, 400, 800]
    p3 = plot(xscale=:log10, yscale=:log10,
        xlabel="N (particles)", ylabel="Time (s)",
        title="Computational Complexity",
        grid=false, legend=:topleft, size=(900, 500), widen=true, bottom_margin=5mm, left_margin=5mm)
    
    # Calculate slopes for each thread count
    logN = log.(float.(N_values))
    for t in 1:max_threads
        times_t = [all_results[t][N] for N in N_values]
        # Exclude the first point for slope estimation
        logN_sub = log.(float.(N_values[2:end]))
        logT_sub = log.(times_t[2:end])
        slope = cov(logN_sub, logT_sub) / var(logN_sub)

        plot!(p3, N_values, times_t;
            marker=:circle, markersize=6,
            label="$(t) thread$(t > 1 ? "s" : "") (slope=$(round(slope, digits=2)))", linewidth=2)
    end
    
    # Add reference lines for O(N) and O(N²)
    ref_n = N_values
    ref_n1 = all_results[1][N_values[1]] * (ref_n ./ ref_n[1])
    ref_n2 = all_results[1][N_values[1]] * (ref_n ./ ref_n[1]).^2
    
    plot!(p3, ref_n, ref_n1; linestyle=:dot, linewidth=1.5, label="O(N)", color=:black, alpha=0.7)
    plot!(p3, ref_n, ref_n2; linestyle=:dash, linewidth=1.5, label="O(N²)", color=:black, alpha=0.7)
        
    # Save and display
    benchmark_dir = joinpath(@__DIR__, "output", "benchmark")
    mkpath(benchmark_dir)
    output_file = joinpath(benchmark_dir, "benchmark_threading.png")
    savefig(p3, output_file)
    println("Benchmark complete. Plot saved to: benchmarks/output/benchmark/benchmark_threading.png\n")
end