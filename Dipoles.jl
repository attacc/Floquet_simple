#
# Dipole matrix elements
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads

#
# Dipoles are calculated using dH/dk
#
function Build_Dipole(k_grid, lattice, TB_sol, orbitals, Hamiltonian, dk)

    println("Delta-k for derivatives : $dk ")
    println("Building Dipoles using dH/dk:")

    h_dim = TB_sol.h_dim
    s_dim = Int(lattice.dim)

    # I consider only off-diagonal elements of the Dipole
    mask = trues(h_dim, h_dim)
    for i in 1:h_dim
        mask[i, i] = false
    end

    Dip_h = zeros(ComplexF64, h_dim, h_dim, s_dim, k_grid.nk)
    ∇H_w  = zeros(ComplexF64, h_dim, h_dim, s_dim, k_grid.nk)

    Threads.@threads for ik in 1:k_grid.nk
        ∇H_w[:, :, :, ik] = Grad_H(ik, k_grid, lattice, TB_sol; Hamiltonian=Hamiltonian, deltaK=dk)

        for id in 1:s_dim
            Dip_h[:, :, id, ik] = WH_rotate(∇H_w[:, :, id, ik], TB_sol.eigenvec[:, :, ik])
            Dip_h[:, :, id, ik] .*= mask
        end

        for i in 1:h_dim
            for j in i+1:h_dim
                Dip_h[i, j, :, ik] .= 1im .* Dip_h[i, j, :, ik] ./ (TB_sol.eigenval[j, ik] - TB_sol.eigenval[i, ik])
                Dip_h[j, i, :, ik] .= conj.(Dip_h[i, j, :, ik])
            end
        end
    end

    return Dip_h, ∇H_w
end
