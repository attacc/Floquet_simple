module FloquetTB

using LinearAlgebra
using ProgressBars

"""
    compute_fourier_components(Hamiltonian, k_vec, A0, omega_L; N_fourier=2, N_time=128)

Computes the Fourier components H_m(k) of the Peierls-substituted Hamiltonian H_0(k + A(t)).
"""
function compute_fourier_components(Hamiltonian, k_vec::Vector{Float64}, A0::Vector{Float64}, omega_L::Float64; N_fourier::Int=2, N_time::Int=128)
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

Assembles the block Floquet-Bloch Hamiltonian of dimension (2*N_floq + 1)*h_dim.
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
                H_F[r_range, c_range] = H_m[0] .+ (m * omega_L) .* I_h
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

        H_m = compute_fourier_components(Hamiltonian, k_vec, A0, omega_L; N_fourier=N_floq, N_time=N_time)
        H_F = build_floquet_matrix(H_m, h_dim, omega_L, N_floq)

        data = eigen(Hermitian(H_F))

        quasi_energies[:, ik] = real.(data.values)
        floquet_states[:, :, ik] = data.vectors
    end

    return quasi_energies, floquet_states
end

end # module
