using LinearAlgebra
using Base.Threads
using .LatticeTools: Lattice, wrap_k_to_BZ  # Import wrap_k_to_BZ from your LatticeTools module

"""
    rytova_keldysh_au(q_au, r0_au, eps_bg, q0_cutoff_au)

Calculates the 2D Rytova-Keldysh screened Coulomb potential in Atomic Units (Hartree * Bohr).
- `q_au`: Momentum transfer in 1/Bohr
- `r0_au`: Screening length in Bohr (33.5 Angstroms = 63.306 Bohr)
- `eps_bg`: Background dielectric constant
- `q0_cutoff_au`: Cutoff parameter in 1/Bohr
"""
function rytova_keldysh_au(q_au::Float64, r0_au::Float64, eps_bg::Float64, q0_cutoff_au::Float64)
    # e^2 = 1.0 in atomic units (Hartree * Bohr)
    q_eff = max(q_au, q0_cutoff_au)
    return (2.0 * pi) / (eps_bg * q_eff * (1.0 + r0_au * q_eff))
end

"""
    solve_bse(tb_sol, k_grid, lattice, r0_ang, eps_bg)

Solves the Bethe-Salpeter Equation using native Atomic Units (Hartree / Bohr).
Input `r0_ang` is provided in Angstroms and converted internally to Bohr.
Returns exciton energies in Hartrees.
"""
function solve_bse(tb_sol, k_grid, lattice, r0_ang::Float64, eps_bg::Float64)
    Nk = k_grid.nk
    H_BSE = zeros(ComplexF64, Nk, Nk)
    
    # 1. Convert physical screening length r0 from Angstroms to Bohr (1 Angstrom = 1.8897261 Bohr)
    r0_au = r0_ang * 1.889726125
    
    # Extract valence and conduction energies (already in Hartrees)
    E_v = tb_sol.eigenval[1, :]
    E_c = tb_sol.eigenval[2, :]
    
    # Reciprocal lattice vectors (in 1/Bohr)
    b1 = lattice.rvectors[1]
    b2 = lattice.rvectors[2]
    
    # Compute 2D Brillouin Zone area in atomic units (1/Bohr^2)
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])
    
    # Area weight per k-point (d^2k integration weight)
    area_per_k = BZ_area / Nk
    #cut off at q=0
    q0_au = sqrt(area_per_k / pi)

    #Analytic cell-averaged W(q=0): Integral of 2pi / (eps * q * (1 + r0*q)) * q dq dphi
    W_q0 = (2.0 * pi / eps_bg) * (2.0 * pi / area_per_k) * (log(1.0 + r0_au * q0_au) / r0_au)
    
    println("Building BSE matrix:")
    
    Threads.@threads for ik in ProgressBar(1:Nk)
        k_i = k_grid.kpt[:, ik]
        
        u_v_i = tb_sol.eigenvec[:, 1, ik]
        u_c_i = tb_sol.eigenvec[:, 2, ik]
        
        for jk in 1:Nk
            u_v_j = tb_sol.eigenvec[:, 1, jk]
            u_c_j = tb_sol.eigenvec[:, 2, jk]
            # Sublattice overlaps
            overlap_c = dot(u_c_i, u_c_j)
            overlap_v = dot(u_v_j, u_v_i)
            overlap = overlap_c * conj(overlap_v)

            if ik == jk
		# Diagonal kinetic term + q=0 averaged kernel
                K_d = -W_q0 * abs2(dot(u_c_i, u_c_i) * dot(u_v_i, u_v_i)) 
                H_BSE[ik, ik] = (E_c[ik] - E_v[ik]) + K_d * (area_per_k / (4.0 * pi^2))
            else
                k_j = k_grid.kpt[:, jk]
                
                # Momentum transfer in 1/Bohr
                dk = k_i .- k_j
                q_vec = wrap_k_to_BZ(dk, lattice)
                q_au = norm(q_vec)
                
                # Screened potential in Hartrees
                W_q = rytova_keldysh_au(q_au, r0_au, eps_bg, q0_au)
                
                # Direct interaction kernel: d^2k / (2*pi)^2 * W(q)
                K_d = -W_q * overlap
                
                # Scale kernel by (BZ_area / Nk) / (2*pi)^2
                H_BSE[ik, jk] = K_d * (area_per_k / (4.0 * pi^2))
            end
        end
    end
    
    # Diagonalize BSE matrix
    energies, envelopes = eigen(H_BSE)
    
    idx = sortperm(real.(energies))
    return real.(energies[idx]), envelopes[:, idx]
end
