module FloquetTB

using LinearAlgebra
using ProgressBars

export compute_fourier_components, build_floquet_matrix, solve_floquet_on_grid, floquet_linear_response

"""
    compute_fourier_components(Hamiltonian, k_vec, A0, omega_L; N_fourier=2, N_time=128)

Computes the Fourier components H_m(k) of the Peierls-substituted Hamiltonian H_0(k + A(t)),
with A(t) = -A0 sin(omega_L t)  (E(t) = -dA/dt = E0 cos(omega_L t), A0 = E0/omega_L).
Convention: H(t) = sum_m H_m exp(-i m omega_L t),  H_m = (1/T) int_0^T H(t) exp(+i m omega_L t) dt.
"""
function compute_fourier_components(Hamiltonian, k_vec::Vector{Float64}, A0::Vector{Float64}, omega_L::Float64; N_fourier::Int=2, N_time::Int=128)
    omega_L > 0.0 || error("omega_L must be positive (A0 = E0/omega_L and the period 2*pi/omega_L are undefined for omega_L <= 0)")
    t_vals = range(0, stop=2.0*pi/omega_L, length=N_time+1)[1:end-1]

    H0 = Hamiltonian(k_vec)
    h_dim = size(H0, 1)

    H_m = Dict{Int, Matrix{ComplexF64}}()
    for m in -N_fourier:N_fourier
        H_m[m] = zeros(ComplexF64, h_dim, h_dim)
    end

    for t in t_vals
        A_t = -A0 .* sin(omega_L * t)
        k_driven = k_vec .+ A_t
        H_t = ComplexF64.(Hamiltonian(k_driven))

        for m in -N_fourier:N_fourier
            H_m[m] .+= H_t .* exp(1im * m * omega_L * t) ./ N_time
        end
    end

    return H_m
end

"""
    build_floquet_matrix(H_m, h_dim, omega_L, N_floq)

Assembles the block Floquet-Bloch Hamiltonian of dimension (2*N_floq + 1)*h_dim,
(H_F)_{mn} = H_{m-n} - m*omega_L*delta_{mn}  (H_m needed for |m| <= 2*N_floq).
"""
function build_floquet_matrix(H_m::Dict{Int, Matrix{ComplexF64}}, h_dim::Int, omega_L::Float64, N_floq::Int)
    num_blocks = 2 * N_floq + 1
    total_dim = num_blocks * h_dim
    H_F = zeros(ComplexF64, total_dim, total_dim)
    I_h = Matrix{ComplexF64}(I, h_dim, h_dim)

    for m_idx in 1:num_blocks
        m = m_idx - N_floq - 1

        for n_idx in 1:num_blocks
            n = n_idx - N_floq - 1
            diff = m - n

            r_range = (m_idx - 1) * h_dim + 1 : m_idx * h_dim
            c_range = (n_idx - 1) * h_dim + 1 : n_idx * h_dim

            if m == n
                # with H(t) = sum_m H_m exp(-i m w t) the Floquet matrix is  H_{m-n} - m w delta_{mn}
                H_F[r_range, c_range] = H_m[0] .- (m * omega_L) .* I_h
            elseif haskey(H_m, diff)
                H_F[r_range, c_range] = H_m[diff]
            end
        end
    end

    return H_F
end

"""
    solve_floquet_on_grid(k_grid, Hamiltonian, E0_field, omega_L; N_floq=2, N_time=128)

Solves the Floquet quasi-energies and dressed states over the entire k_grid.
"""
function solve_floquet_on_grid(k_grid, Hamiltonian, E0_field::Vector{Float64}, omega_L::Float64; N_floq::Int=2, N_time::Int=128)
    Nk = k_grid.nk
    sample_k = k_grid.kpt[:, 1]
    h_dim = size(Hamiltonian(sample_k), 1)

    A0 = E0_field ./ omega_L

    num_blocks = 2 * N_floq + 1
    total_dim = num_blocks * h_dim

    quasi_energies = zeros(Float64, total_dim, Nk)
    floquet_states = zeros(ComplexF64, total_dim, total_dim, Nk)

    println("Solving Floquet-Bloch system on k-grid ($Nk k-points, $N_floq photon blocks)...")

    for ik in ProgressBar(1:Nk)
        k_vec = k_grid.kpt[:, ik]

        H_m = compute_fourier_components(Hamiltonian, k_vec, A0, omega_L; N_fourier=2*N_floq, N_time=N_time)
        H_F = build_floquet_matrix(H_m, h_dim, omega_L, N_floq)

        data = eigen(Hermitian(H_F))

        quasi_energies[:, ik] = real.(data.values)
        floquet_states[:, :, ik] = data.vectors
    end

    return quasi_energies, floquet_states
end

"""
    floquet_linear_response(Hamiltonian, k_grid, TB_sol, lattice, E0_field, omega_L, eta;
                            Dip_h=nothing, nv=1, N_floq=1, N_time=32)

Linear response (dynamical polarisability) of the Floquet problem, directly comparable with
`Linear_response`. For a weak monochromatic field E(t) = E0 cos(omega_L t) the quasi-energy of the
Floquet state connected to the valence band (v, m=0) is shifted by the optical Stark shift

    delta eps_v(k) = -(E0^2/4) sum_c |E.D_vc|^2 [ 1/(Delta - w - i eta) + 1/(Delta + w - i eta) ]

so that   chi_F(w) = -(4/E0^2) (1/Nk) sum_k sum_v [eps_v^F(k) - E_v(k)]   (x 16 pi/Omega_BZ)

reproduces the polarisability: Im chi_F is the absorption (Linear_response keeps only the resonant
term; Re chi_F also contains the anti-resonant one). The broadening eta is introduced as a damping
-i*eta on the conduction bands in every photon block (non-Hermitian Floquet matrix), so eta > 0 is required.
The field must be weak: |E0 . D| << eta (e.g. E0 ~ 1e-5 a.u.).

Two couplings (same result for eta -> 0, different phenomenological damping off resonance):
- `Dip_h === nothing` : velocity gauge, Peierls substitution H(k + A(t)) (the Floquet code of this module);
                        Im chi_F agrees with Linear_response near resonances, off resonance it differs by
                        ~(Delta/omega)^2 because the damping is not gauge invariant.
- `Dip_h` given       : length gauge, H(t) = diag(E_k) + E0 cos(wt) E.D_k in the band basis with the dipoles
                        of `Build_Dipole`; reproduces Linear_response (plus its anti-resonant term).

The result has the same normalisation as `Linear_response` (BSE eps2 normalisation 16*pi/Omega_BZ).
"""
function floquet_linear_response(Hamiltonian, k_grid, TB_sol, lattice, E0_field::Vector{Float64},
                                 omega_L::Float64, eta::Float64;
                                 Dip_h=nothing, nv::Int=1, N_floq::Int=1, N_time::Int=32)
    omega_L > 0.0 || error("omega_L must be positive")
    eta > 0.0     || error("eta must be > 0 (damping needed to reach the resonances)")
    Nk    = k_grid.nk
    h_dim = TB_sol.h_dim
    E0    = norm(E0_field)
    A0    = E0_field ./ omega_L
    nblk  = 2 * N_floq + 1

    # normalisation compatible with Linear_response / BSE
    b1 = lattice.rvectors[1]
    b2 = lattice.rvectors[2]
    BZ_area   = abs(b1[1] * b2[2] - b1[2] * b2[1])
    prefactor = 16.0 * pi / BZ_area

    shifts = zeros(ComplexF64, Nk)

    Threads.@threads for ik in 1:Nk
        E_k = TB_sol.eigenval[:, ik]
        U   = TB_sol.eigenvec[:, :, ik]

        # Fourier components in the band basis of the undriven Hamiltonian
        H_m = Dict{Int, Matrix{ComplexF64}}()
        if Dip_h === nothing
            Hm_orb = compute_fourier_components(Hamiltonian, k_grid.kpt[:, ik], A0, omega_L;
                                                N_fourier=2*N_floq, N_time=N_time)
            for (m, Hmat) in Hm_orb
                H_m[m] = U' * Hmat * U
            end
        else
            for m in -2*N_floq:2*N_floq
                H_m[m] = zeros(ComplexF64, h_dim, h_dim)
            end
            H_m[0] = Matrix{ComplexF64}(Diagonal(E_k))
            Dk = zeros(ComplexF64, h_dim, h_dim)
            for id in 1:length(E0_field)
                Dk .+= Dip_h[:, :, id, ik] .* E0_field[id]
            end
            H_m[1]  = Dk ./ 2        # cos(wt) = (e^{iwt} + e^{-iwt})/2
            H_m[-1] = Dk ./ 2
        end

        HF = build_floquet_matrix(H_m, h_dim, omega_L, N_floq)

        # damping on the conduction bands, in every photon block
        for blk in 1:nblk, j in (nv + 1):h_dim
            i = (blk - 1) * h_dim + j
            HF[i, i] -= 1im * eta
        end

        F = eigen(HF)                # non-Hermitian: complex quasi-energies
        norms = vec(sum(abs2.(F.vectors), dims=1))

        s = 0.0im
        for iv in 1:nv
            idx0 = N_floq * h_dim + iv                 # component (valence iv, photon block m=0)
            wgt  = abs2.(F.vectors[idx0, :]) ./ norms
            j    = argmax(wgt)                         # Floquet state connected to (v, m=0)
            s   += F.values[j] - E_k[iv]
        end
        shifts[ik] = s
    end

    chi = -4.0 / E0^2 * sum(shifts) / Nk
    return prefactor * chi
end

end # module
