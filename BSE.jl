using LinearAlgebra
using Base.Threads

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
    area_per_kpoint = BZ_area / Nk
    q0_cutoff_au = sqrt(area_per_kpoint / pi)
    
    println("Building BSE matrix in Atomic Units:")
    
    for ik in 1:Nk
        k_i = k_grid.kpt[:, ik]
        
        u_v_i = tb_sol.eigenvec[:, 1, ik]
        u_c_i = tb_sol.eigenvec[:, 2, ik]
        
        for jk in 1:Nk
            if ik == jk
                # Diagonal kinetic energy: E_c(k) - E_v(k) [Hartrees]
                H_BSE[ik, ik] = E_c[ik] - E_v[ik]
            else
                k_j = k_grid.kpt[:, jk]
                
                # Momentum transfer in 1/Bohr
                dk = k_i .- k_j
                q_au = norm(dk)
                
                # Screened potential in Hartrees
                W_q = rytova_keldysh_au(q_au, r0_au, eps_bg, q0_cutoff_au)
                
                u_v_j = tb_sol.eigenvec[:, 1, jk]
                u_c_j = tb_sol.eigenvec[:, 2, jk]
                
                # Sublattice overlaps
                overlap_c = dot(u_c_i, u_c_j)
                overlap_v = dot(u_v_j, u_v_i)
                
                # Direct interaction kernel: d^2k / (2*pi)^2 * W(q)
                K_d = -W_q * overlap_c * overlap_v
                
                # Scale kernel by (BZ_area / Nk) / (2*pi)^2
                H_BSE[ik, jk] = K_d * (area_per_kpoint / (4.0 * pi^2))
            end
        end
    end
    
    # Diagonalize BSE matrix
    energies, envelopes = eigen(H_BSE)
    
    idx = sortperm(real.(energies))
    return real.(energies[idx]), envelopes[:, idx]
end
