"""
    write_vector_csv(fname::AbstractString, v::AbstractVector{<:Real}; dir::AbstractString="../data")

Write a vector of real numbers to a CSV file.

# Arguments
- `fname`: Name of the CSV file to write.
- `v`: Vector of real numbers to write.
- `dir`: Directory to save the file (default: "../data").

# Returns
- The full path to the written CSV file as a string.
"""
function write_vector_csv(fname::AbstractString, v::AbstractVector{<:Real}; dir::AbstractString="../data")
    fullpath = joinpath(dir, fname)
    mkpath(dir)
    open(fullpath, "w") do io
        for i in eachindex(v)
            println(io, float(v[i]))
        end
    end
    return fullpath
end

"""
    read_vector_csv(path::AbstractString)

Read a vector of Float64 values from a CSV file.

# Arguments
- `path`: Path to the CSV file.

# Returns
- A vector of Float64 values read from the file.
"""
function read_vector_csv(path::AbstractString)
    data = readdlm(path, ',', Float64)
    if ndims(data) == 0
        return Float64[]
    elseif ndims(data) == 1
        return Vector{Float64}(data)
    else
        return vec(data)
    end
end
