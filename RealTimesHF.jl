module RealTimesHF

using LinearAlgebra
using ProgressBars

"""
    rytova_keldysh(q, r0, eps_bg, q0_cutoff)
"""
function rytova_keldysh(q::Float64, r0::Float64, eps_bg::Float64, q0_cutoff::Float64)
    e2 = 14.3996 # eV * Angstrom
    q_eff = max(q, q0_cutoff)
    return (2.0 * pi * e2) / (eps_bg * q_eff * (1.0 + r0 * q_eff))
end

"""
    compute_dipole_moments(tb_sol, k_grid, lattice)

Computes the dipole matrix elements D_cv(k) = <u_c| i grad_k |u_v> in band basis.
Uses finite-difference momentum derivatives from Grad_H or analytical dH/dk.
"""
function compute_dipole_moments(tb_sol, k_grid, dH_dk_list)
    Nk = k_grid.nk
    D_cv = zeros(ComplexF64, 2, Nk) # 2 space dimensions (x, y)
    
    for ik in 1:Nk
        E_v = tb_sol.eigenval[1, ik]
        E_c = tb_sol.eigenval[2, ik]
        gap = E_c - E_v
        
        u_v = tb_sol.eigenvec[:, 1, ik]
        u_c = tb_sol.eigenvec[:, 2, ik]
        
        for dir in 1:2
            # Velocity operator matrix element <u_c | dH/dk | u_v>
            dH_matrix = dH_dk_list[ik][:, :, dir]
            v_cv = dot(u_c, dH_matrix * u_v)
            
            # Dipole matrix element D_cv = i * v_cv / gap
            if abs(gap) > 1e-6
                D_cv[dir, ik] = im * v_cv / gap
            end
        end
    end
    return D_cv
end

"""
    build_sHF_Hamiltonian(rho, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)

Constructs the time-dependent screened Hartree-Fock Hamiltonian matrix in band basis at each k-point:
H_sHF(k, t) = H_0(k) + Sigma_sex[rho(t)] - D . E(t)
"""
function build_sHF_Hamiltonian(rho, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)
    Nk = k_grid.nk
    H_sHF = [zeros(ComplexF64, 2, 2) for _ in 1:Nk]
    
    # 1. Non-interacting band energies H_0
    for ik in 1:Nk
        H_sHF[ik][1, 1] = tb_sol.eigenval[1, ik]
        H_sHF[ik][2, 2] = tb_sol.eigenval[2, ik]
        
        # Dipole coupling: - D . E(t)
        dipole_coupling = -(D_cv[1, ik] * E_field[1] + D_cv[2, ik] * E_field[2])
        H_sHF[ik][1, 2] = conj(dipole_coupling)
        H_sHF[ik][2, 1] = dipole_coupling
    end
    
    # 2. Add Screened Exchange Self-Energy (SEX) from Rytova-Keldysh potential
    for i in 1:Nk
        k_i = k_grid.kpt[:, i]
        u_v_i = tb_sol.eigenvec[:, 1, i]
        u_c_i = tb_sol.eigenvec[:, 2, i]
        
        for j in 1:Nk
            if i != j
                k_j = k_grid.kpt[:, j]
                q = norm(k_i .- k_j)
                W_q = rytova_keldysh(q, r0, eps_bg, q0_cutoff)
                
                u_v_j = tb_sol.eigenvec[:, 1, j]
                u_c_j = tb_sol.eigenvec[:, 2, j]
                
                # Single-particle overlaps
                S_vv = dot(u_v_i, u_v_j)
                S_vc = dot(u_v_i, u_c_j)
                S_cv = dot(u_c_i, u_v_j)
                S_cc = dot(u_c_i, u_c_j)
                
                # Update off-diagonal interband density matrix elements (screened exchange)
                # Sigma_sex_cv = - W_q/Nk * rho_cv(j) * (overlaps)
                Sigma_cv = - (W_q / Nk) * rho[j][2, 1] * S_cc * conj(S_vv)
                H_sHF[i][2, 1] += Sigma_cv
                H_sHF[i][1, 2] += conj(Sigma_cv)
            end
        end
    end
    
    return H_sHF
end

"""
    rhodot(rho, H_sHF)

Computes d/dt rho(t) = -i/hbar [H_sHF, rho]
"""
function rhodot(rho, H_sHF, Nk)
    drho = [zeros(ComplexF64, 2, 2) for _ in 1:Nk]
    for ik in 1:Nk
        # Commutator [H, rho] = H*rho - rho*H
        comm = H_sHF[ik] * rho[ik] - rho[ik] * H_sHF[ik]
        drho[ik] = -im * comm
    end
    return drho
end

"""
    propagate_real_time(tb_sol, k_grid, lattice, dH_dk_list, r0, eps_bg, dt, Nt, E0_pulse)

Propagates the density matrix in real-time under a delta electric field pulse using RK4.
"""
function propagate_real_time(tb_sol, k_grid, lattice, dH_dk_list, r0::Float64, eps_bg::Float64, dt::Float64, Nt::Int, E0_pulse::Vector{Float64})
    Nk = k_grid.nk
    
    # Reciprocal cell setup for q0 cutoff
    b1 = lattice.bvectors[:, 1]
    b2 = lattice.bvectors[:, 2]
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])
    q0_cutoff = sqrt(BZ_area / (pi * Nk))
    
    # Precompute dipole matrix elements
    D_cv = compute_dipole_moments(tb_sol, k_grid, dH_dk_list)
    
    # Initialize ground state density matrix: filled valence band (1,1)=1, empty conduction band (2,2)=0
    rho = [ComplexF64[1.0 0.0; 0.0 0.0] for _ in 1:Nk]
    
    # Storage for time-dependent macroscopic dipole polarization P(t)
    P_t = zeros(Float64, 2, Nt)
    time_axis = [(t - 1) * dt for t in 1:Nt]
    
    println("Starting Real-Time Screened Hartree-Fock Propagation ($Nt steps)...")
    
    for t in ProgressBar(1:Nt)
        current_time = time_axis[t]
        
        # Delta pulse applied at t = 0 (first time step)
        E_field = (t == 1) ? E0_pulse ./ dt : [0.0, 0.0]
        
        # Compute current macroscopic polarization P(t) = (1/Nk) Sum_k Tr(D * rho(k))
        P_x, P_y = 0.0, 0.0
        for ik in 1:Nk
            P_x += 2.0 * real(D_cv[1, ik] * rho[ik][1, 2])
            P_y += 2.0 * real(D_cv[2, ik] * rho[ik][1, 2])
        end
        P_t[1, t] = P_x / Nk
        P_t[2, t] = P_y / Nk
        
        # RK4 Time Step
        # k1
        H1 = build_sHF_Hamiltonian(rho, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)
        k1 = rhodot(rho, H1, Nk)
        
        # k2
        rho_k2 = [rho[ik] + 0.5 * dt * k1[ik] for ik in 1:Nk]
        H2 = build_sHF_Hamiltonian(rho_k2, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)
        k2 = rhodot(rho_k2, H2, Nk)
        
        # k3
        rho_k3 = [rho[ik] + 0.5 * dt * k2[ik] for ik in 1:Nk]
        H3 = build_sHF_Hamiltonian(rho_k3, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)
        k3 = rhodot(rho_k3, H3, Nk)
        
        # k4
        rho_k4 = [rho[ik] + dt * k3[ik] for ik in 1:Nk]
        H4 = build_sHF_Hamiltonian(rho_k4, tb_sol, k_grid, lattice, r0, eps_bg, q0_cutoff, E_field, D_cv)
        k4 = rhodot(rho_k4, H4, Nk)
        
        # Update density matrix
        for ik in 1:Nk
            rho[ik] += (dt / 6.0) * (k1[ik] + 2.0 * k2[ik] + 2.0 * k3[ik] + k4[ik])
        end
    end
    
    return time_axis, P_t
end

end # module
