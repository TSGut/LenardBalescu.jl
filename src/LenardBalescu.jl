"""
    LenardBalescu

A Julia package for Lenard-Balescu (and Landau) collision operator physics simulations.
"""
module LenardBalescu

using LinearAlgebra
using SpecialFunctions
using DSP
using Interpolations
using Random, Statistics
using DelimitedFiles

export Params, SetParam, update_kernel_params!, BasisKernel, BKW_vector,
       initialize_particles_stratified_BKW, initialize_particles_stratified_BKW_elliptical,
       initialize_particles_stratified_maxwellian,
       initialize_particles_maxwellian, initialize_particles_stratified_bumptail,
     initialize_particles_torus_3d,
     initialize_particles_spherical_shell_3d,
       initialize_particles_stratified_double_ring,
       f_recon, Collision, getA_Landau,
       build_velocity_grid, L1_error, L2_error, write_vector_csv,
       read_vector_csv, run_simulation,
    compute_radial_binned_f, fit_bspline,
     compute_conserved_quantities,
     precompute_dielectric_lookup_grid

include("params.jl")
include("conserved.jl")
include("basis_functions.jl")
include("io.jl")
include("grid.jl")
include("f_reconstruction.jl")
include("kernel.jl")
include("bkw.jl")
include("collision.jl")
include("init_particles.jl")
include("simulate.jl")

end
