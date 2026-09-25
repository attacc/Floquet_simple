using .FloquetTB

# 1. Define laser drive parameters
E0_field = [0.05, 0.0]  # Electric field amplitude vector (x-polarized drive)
omega_L  = 1.5          # Laser photon energy / frequency (eV)
N_floq   = 2            # Number of Floquet photon sidebands (-2, -1, 0, +1, +2)

# 2. Solve Floquet system on k_grid
quasi_energies, floquet_states = solve_floquet_on_grid(k_grid, Hamiltonian, E0_field, omega_L; N_floq=N_floq)

# 3. Print Quasi-Energy Band Gap at K-point (or minimum gap)
band_gaps = quasi_energies[N_floq*2 + 1, :] .- quasi_energies[N_floq*2, :]
println("Floquet-renormalized minimum quasi-energy gap: ", minimum(band_gaps), " eV")
