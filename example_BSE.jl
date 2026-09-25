#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using DataFrames
using Base.Threads
using PyPlot

include("units.jl")
using .Units

include("TB_hBN.jl")
using .hBN2D

include("lattice.jl")
using .LatticeTools

include("TB_tools.jl")
using .TB_tools

include("bz_sampling.jl")
using .BZ_sampling

include("Dipoles.jl")
include("Linear_response.jl")
include("BSE.jl")
include("BSE_dipoles.jl")
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# * * * DIPOLES * * * #
#
# dipoles are calculated using dH/dh
#

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

lattice=set_Lattice(2,[a_1,a_2])

n_k1=24
n_k2=24

#
# Gauge for the tight-binding is "lattice" gauge
#
# Step for finite differences in k-space
#
dk=0.001

# For Linear reponse only
freqs_range  =[0.0/ha2ev, 20.0/ha2ev] # eV
eta          =0.15/ha2ev
freqs_nsteps =400
E_vec        = [0.0,1.0] # electric field direction

k_grid=generate_unif_grid(n_k1, n_k2, lattice)
# 
# Solve TB on a regular grid
#
TB_sol=Solve_TB_on_grid(k_grid,Hamiltonian)
# 
# Dipoles
#
Dip_h,∇H_w=Build_Dipole(k_grid,lattice,TB_sol,orbitals,Hamiltonian,dk)

freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
# IP dielectric response using existing Linear_response module
eps_2_ip = Linear_response(TB_sol, Dip_h, freqs, E_vec, eta)

#
#
#
## Physical Parameters for Monolayer hBN for electron-hole interaction
r0     = 33.5  # Screening length in Angstroms
eps_bg = 1.0   # 1.0 for suspended vacuum, 2.45 for SiO2 substrate


# 3. Solve BSE directly on top of `tb_sol` and `k_grid`[cite: 2]
exciton_energies, exciton_wavefunctions = solve_bse(TB_sol, k_grid, lattice, r0, eps_bg)

println("Lowest Exciton Energy[1]: ", exciton_energies[1]*ha2ev, " eV")
println("Lowest Exciton Energy[2]: ", exciton_energies[2]*ha2ev, " eV")
println("Lowest Exciton Energy[3]: ", exciton_energies[3]*ha2ev, " eV")
println("Lowest Exciton Energy[4]: ", exciton_energies[4]*ha2ev, " eV")

D_exciton, f_osc = Build_Exciton_Dipoles(Dip_h, exciton_envelopes, k_grid, lattice)


# 4. Compute \varepsilon_2(\omega) along x-polarization [1, 0]
freqs_ha = LinRange(0.0 / ha2ev, 8.0 / ha2ev, 400)[cite: 5, 6]
eta_ha   = eta # 0.15 / ha2ev[cite: 5]

eps_2 = Build_Dielectric_Function(
    D_exciton,
    exciton_energies,
    freqs_ha,
    lattice,
    k_grid;
    eta=eta_ha,
    pol_dir=[1.0, 0.0]
)

# Convert outputs for printing/plotting
freqs_ev = freqs_ha .* ha2ev[cite: 6]
