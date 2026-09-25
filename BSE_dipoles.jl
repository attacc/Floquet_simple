"""
    Build_Exciton_Dipoles(Dip_h, exciton_envelopes, k_grid, lattice)

Calculates the excitonic transition dipoles and oscillator strengths from the single-particle
dipoles `Dip_h` and the BSE eigenvectors `exciton_envelopes`.

# Arguments
- `Dip_h`: Array(h_dim, h_dim, s_dim, Nk) of single-particle dipoles from Build_Dipole[cite: 8]
- `exciton_envelopes`: Matrix(Nk, N_excitons) containing BSE eigenvectors A^S(k) as columns
- `k_grid`: K-point sampling structure[cite: 7]
- `lattice`: Lattice structure containing space dimension `dim`[cite: 3, 7]

# Returns
- `D_exciton`: Array(s_dim, N_excitons) containing complex exciton dipoles [D_x, D_y]
- `f_osc`: Vector(N_excitons) containing total oscillator strengths |D|^2
"""
function Build_Exciton_Dipoles(Dip_h, exciton_envelopes, k_grid, lattice)
    Nk = k_grid.nk
    s_dim = lattice.dim
    N_excitons = size(exciton_envelopes, 2)
    
    # D_exciton[direction, exciton_index]
    D_exciton = zeros(ComplexF64, s_dim, N_excitons)
    f_osc     = zeros(Float64, N_excitons)
    
    # Band indices for 2-band model: Valence = 1, Conduction = 2
    v_idx = 1
    c_idx = 2
    
    println("Calculating Exciton Dipoles:")
    
    for S in 1:N_excitons
        # Extract envelope A_S(k) for state S
        A_S = exciton_envelopes[:, S]
        
        for id in 1:s_dim
            # D_S^\alpha = (1 / \sqrt{Nk}) * \sum_k A^S(k) * Dip_h[v, c, id, k]
            dip_sum = 0.0 + 0.0im
            for ik in 1:Nk
                dip_sum += A_S[ik] * Dip_h[v_idx, c_idx, id, ik]
            end
            D_exciton[id, S] = dip_sum / sqrt(Nk)
        end
        
        # Total oscillator strength |D_S|^2
        f_osc[S] = sum(abs2.(D_exciton[:, S]))
    end
    
    return D_exciton, f_osc
end

"""
    Build_Dielectric_Function(D_exciton, exciton_energies, freqs, lattice, k_grid; eta=0.005, pol_dir=[1.0, 0.0])

Calculates the 2D macroscopic dielectric function eps_2(omega) from exciton dipoles.

# Arguments
- `D_exciton`: Matrix(s_dim, N_excitons) of exciton dipoles
- `exciton_energies`: Vector of exciton energies Omega_S (in Hartrees)
- `freqs`: Range/Vector of photon energies hbar omega (in Hartrees)
- `lattice`: Lattice structure containing reciprocal vectors[cite: 3]
- `k_grid`: K-point grid structure[cite: 7]
- `eta`: Broadening parameter in Hartrees
- `pol_dir`: Polarization vector [E_x, E_y]
"""
function Build_Dielectric_Function(
    D_exciton, 
    exciton_energies, 
    freqs, 
    lattice, 
    k_grid; 
    eta::Float64=0.005, 
    pol_dir::Vector{Float64}=[1.0, 0.0]
)
    N_excitons = length(exciton_energies)
    N_freqs = length(freqs)
    eps_2 = zeros(Float64, N_freqs)
    
    # Normalize polarization direction vector
    e_pol = pol_dir / norm(pol_dir)
    
    # Reciprocal unit cell area A_BZ in 1/Bohr^2
    b1 = lattice.rvectors[1][cite: 3]
    b2 = lattice.rvectors[2][cite: 3]
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])
    
    # 2D normalization prefactor: 16 * \pi^2 / A_BZ
    prefactor = (16.0 * pi^2) / BZ_area
    
    for (iw, w) in enumerate(freqs)
        val = 0.0
        for S in 1:N_excitons
            Omega_S = exciton_energies[S]
            
            # Project exciton dipole onto polarization vector
            D_proj = dot(e_pol, D_exciton[:, S])
            
            # Lorentzian broadening delta_eta(w - Omega_S)
            lorentzian = (1.0 / pi) * (eta / ((w - Omega_S)^2 + eta^2))
            
            val += abs2(D_proj) * lorentzian
        end
        eps_2[iw] = prefactor * val
    end
    
    return eps_2
end
