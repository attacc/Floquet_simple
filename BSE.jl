using LinearAlgebra

"""
    rytova_keldysh(q, r0, eps_bg, q0_cutoff)

Calculates the 2D Rytova-Keldysh screened Coulomb potential in momentum space.
"""
function rytova_keldysh(q::Float64, r0::Float64, eps_bg::Float64, q0_cutoff::Float64)
    # e^2 in eV * Angstroms
    e2 = 14.3996 
    
    # Regularize q=0 divergence
    q_eff = max(q, q0_cutoff)
    
    return (2.0 * pi * e2) / (eps_bg * q_eff * (1.0 + r0 * q_eff))
end

"""
    solve_bse(tb_sol, k_grid, lattice, r0, eps_bg)

Solves the Bethe-Salpeter Equation using the output structure from TB_tools.jl.
"""
function solve_bse(tb_sol, k_grid, lattice, r0::Float64, eps_bg::Float64)
    Nk = k_grid.nk
    H_BSE = zeros(ComplexF64, Nk, Nk)
    
    # Extract valence and conduction energies from TB_Solution
    # Assuming standard 2-band model (Index 1 = Valence, Index 2 = Conduction)
    E_v = tb_sol.eigenval[1, :]
    E_c = tb_sol.eigenval[2, :]
    
    # Calculate reciprocal unit cell area to estimate q0 cutoff
    # Area = |b1 x b2|
    b1 = lattice.bvectors[:, 1]
    b2 = lattice.bvectors[:, 2]
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])
    
    area_per_kpoint = BZ_area / Nk
    q0_cutoff = sqrt(area_per_kpoint / pi)
    
    for i in 1:Nk
        # Get k-vector for state i
        k_i = k_grid.kpt[:, i]
        
        # Single-particle eigenvectors for point i
        u_v_i = tb_sol.eigenvec[:, 1, i]
        u_c_i = tb_sol.eigenvec[:, 2, i]
        
        for j in 1:Nk
            if i == j
                # Diagonal transition energy: E_c(k) - E_v(k)
                H_BSE[i, i] = E_c[i] - E_v[i]
            else
                k_j = k_grid.kpt[:, j]
                
                # Momentum transfer vector dk
                dk = k_i .- k_j
                q = norm(dk)
                
                # Direct and exchange interactions
                W_q = rytova_keldysh(q, r0, eps_bg, q0_cutoff)
                V_q = rytova_keldysh(q, 0.0, 1.0, q0_cutoff)
                
                # Fetch single-particle eigenvectors for point j
                u_v_j = tb_sol.eigenvec[:, 1, j]
                u_c_j = tb_sol.eigenvec[:, 2, j]
                
                # Sublattice overlaps
                overlap_c = dot(u_c_i, u_c_j) # u_c(k)* . u_c(k')
                overlap_v = dot(u_v_j, u_v_i) # u_v(k')* . u_v(k)
                
                overlap_x1 = dot(u_v_i, u_c_i)
                overlap_x2 = dot(u_c_j, u_v_j)
                
                # Kernel components
                K_d = -W_q * overlap_c * overlap_v
                K_x = 2.0 * V_q * overlap_x1 * overlap_x2
                
                H_BSE[i, j] = (K_d + K_x) / Nk
            end
        end
    end
    
    # Diagonalize BSE matrix
    energies, envelopes = eigen(H_BSE)
    
    # Sort exciton energies ascending
    idx = sortperm(real.(energies))
    return real.(energies[idx]), envelopes[:, idx]
end
