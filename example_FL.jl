include("FloquetTB.jl")

include("units.jl")
using .Units

include("TB_tools.jl")
using .TB_tools

include("TB_hBN.jl")
using .hBN2D

include("lattice.jl")
using .LatticeTools

include("bz_sampling.jl")
using .BZ_sampling

# Assumed to already exist in your main script:
# k_grid, Hamiltonian, TB_sol, lattice, orbitals, dk

include("Dipoles.jl")
include("Linear_response.jl")

# ----------------------------
# Linear-response energy range
# ----------------------------
Emin = 0.0
Emax = 4.0
nE   = 400
energies = collect(range(Emin, Emax, length=nE))

# Probe field polarization for the linear-response calculation
E_probe = [1.0, 0.0]

# Broadening
η = 0.05
nv = 1

lattice=set_Lattice(2,[a_1,a_2])

n_k1=36
n_k2=36
k_grid=generate_unif_grid(n_k1, n_k2, lattice)
# 
# Solve TB on a regular grid
#
TB_sol=Solve_TB_on_grid(k_grid,Hamiltonian)

println("Building dipole matrix elements...")
# Step for finite differences in k-space
dk=0.001
Dip_h, ∇H_w = Build_Dipole(k_grid, lattice, TB_sol, orbitals, Hamiltonian, dk)

println("Computing linear-response spectrum on $(length(energies)) energies...")
χ_lr = Linear_response(k_grid, TB_sol, Dip_h, energies, E_probe, η, nv)

# Absorption-like spectrum in the usual convention
absorption_lr = -imag.(χ_lr)

println("Linear-response spectrum computed.")
println("Peak position at E = ", energies[argmax(absorption_lr)], " eV")

# -------------------------------------------------
# Optional Floquet check in the weak-field regime
# -------------------------------------------------
# Use a very small drive to stay close to linear response
E0_field = 1e-4 .* E_probe
N_floq = 1

# Example: evaluate Floquet on a coarse subset of the same energy window
step = max(1, nE ÷ 20)
sample_energies = energies[1:step:end]
floquet_gap = zeros(length(sample_energies))

println("Running a weak-field Floquet check on a coarse energy subset...")
for (i, ωL) in enumerate(sample_energies)
    qe, _ = solve_floquet_on_grid(k_grid, Hamiltonian, E0_field, ωL; N_floq=N_floq, N_time=128)

    # Simple proxy: minimum splitting between the two central quasi-energy branches
    mid = size(qe, 1) ÷ 2
    floquet_gap[i] = minimum(qe[mid+1, :] .- qe[mid, :])
end

println("Floquet check done.")
println("You can now compare:")
println("  - absorption_lr vs energies")
println("  - floquet_gap vs sample_energies")
