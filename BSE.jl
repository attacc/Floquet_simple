using LinearAlgebra
using Base.Threads
using .LatticeTools: Lattice, min_image_shifts
using .hBN2D:rytova_keldysh_kernel

"""
    bse_index(iv, ic, ik, nc, Nk)

Position of the electron-hole pair (valence `iv`, conduction `ic`, k-point `ik`) in the BSE basis.
`iv` and `ic` are positions in the lists `vbands`, `cbands` of `solve_bse` (1-based), `nc` = length(cbands).
The k index runs fastest:  index = ik + Nk*((ic-1) + nc*(iv-1)).  For one valence and one conduction band
the index is simply ik (layout of the original two-band code).
"""
@inline bse_index(iv::Int, ic::Int, ik::Int, nc::Int, Nk::Int) = ik + Nk * ((ic - 1) + nc * (iv - 1))

"""
    solve_bse(tb_sol, k_grid, lattice, orbitals, r0_ang, eps_bg; nv=h_dim÷2, vbands=1:nv, cbands=nv+1:h_dim)

Solves the Bethe-Salpeter Equation (Tamm-Dancoff, Q = 0, direct screened term) in atomic units (Hartree / Bohr),
for any number of valence and conduction bands. The basis is |v c k> = c^+_{ck} c_{vk}|0>, v in `vbands`,
c in `cbands`:

    H_{vck,v'c'k'} = (E_c(k) - E_v(k)) delta_{vv'} delta_{cc'} delta_{kk'}
                     - W(q_kk') < u_c(k) | u_c'(k') > < u_v'(k') | u_v(k) > / (Nk A_cell)

(umklapp-averaged overlaps for k != k', cell-averaged W(q=0) for k = k').
Input `r0_ang` is in Angstrom (converted to Bohr). Returns the exciton energies (Hartree, ascending) and the
envelopes A^S as columns of a (nv*nc*Nk) x (nv*nc*Nk) matrix, ordered as in `bse_index`.
With the default for a two-band model (nv=1) the result is identical to the previous two-band code.
Note: `Build_Exciton_Dipoles` (BSE_dipoles.jl) still assumes one valence and one conduction band.
"""
function solve_bse(tb_sol, k_grid, lattice, orbitals, r0_ang::Float64, eps_bg::Float64;
                   nv::Int=size(tb_sol.eigenval, 1) ÷ 2,
                   vbands=1:nv, cbands=(nv + 1):size(tb_sol.eigenval, 1))
    Nk  = k_grid.nk
    nvb = length(vbands)
    ncb = length(cbands)
    (nvb >= 1 && ncb >= 1) || error("solve_bse: at least one valence and one conduction band are required")
    Nt  = nvb * ncb * Nk
    H_BSE = zeros(ComplexF64, Nt, Nt)

    # 1. Convert physical screening length r0 from Angstroms to Bohr (1 Angstrom = 1.8897261 Bohr)
    r0_au = r0_ang * 1.889726125

    E = tb_sol.eigenval        # E[n, k] (Hartree)
    U = tb_sol.eigenvec        # U[a, n, k]

    # Reciprocal lattice vectors (in 1/Bohr)
    b1 = lattice.rvectors[1]
    b2 = lattice.rvectors[2]

    # Compute 2D Brillouin Zone area in atomic units (1/Bohr^2)
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])

    # Area weight per k-point (d^2k integration weight)
    area_per_k = BZ_area / Nk
    pref = area_per_k / (4.0 * pi^2)
    #cut off at q=0
    q0_au = sqrt(area_per_k / pi)

    #Analytic cell-averaged W(q=0): Integral of 2pi / (eps * q * (1 + r0*q)) * q dq dphi
    W_q0 = (2.0 * pi / eps_bg) * (2.0 * pi / area_per_k) * (log(1.0 + r0_au * q0_au) / r0_au)

    tau  = orbitals.tau
    norb = length(tau)

    println("Building BSE matrix (", nvb, " valence x ", ncb, " conduction bands, dimension ", Nt, "):")

    Threads.@threads for ik in ProgressBar(1:Nk)
        k_i = k_grid.kpt[:, ik]
        Oc  = zeros(ComplexF64, ncb, ncb)      # Oc[b, b2] = <u_c(k)| ph |u_c2(k')>
        Ov  = zeros(ComplexF64, nvb, nvb)      # Ov[a2, a] = <ph u_v2(k')| u_v(k)>
        acc = zeros(ComplexF64, ncb, ncb, nvb, nvb)

        for jk in 1:Nk
            if ik == jk
                W    = W_q0                                         # q=0 averaged kernel
                phs  = [ones(ComplexF64, norb)]
            else
                k_j = k_grid.kpt[:, jk]
                # Momentum transfer in 1/Bohr
                dk = k_i .- k_j
                q_au, Gs = min_image_shifts(dk, lattice)
                W   = rytova_keldysh_kernel(q_au, r0_au, eps_bg, q0_au)
                phs = [[cis(-dot(-G, tau[a])) for a in 1:norb] for G in Gs]   # k_j' = k_j + Gp, Gp = -G
            end

            # umklapp-averaged products of conduction and valence overlaps
            acc .= 0.0
            for ph in phs
                for b in 1:ncb, b2 in 1:ncb
                    Oc[b, b2] = dot(U[:, cbands[b], ik], ph .* U[:, cbands[b2], jk])
                end
                for a2 in 1:nvb, a in 1:nvb
                    Ov[a2, a] = dot(ph .* U[:, vbands[a2], jk], U[:, vbands[a], ik])
                end
                for a in 1:nvb, a2 in 1:nvb, b in 1:ncb, b2 in 1:ncb
                    acc[b, b2, a, a2] += Oc[b, b2] * Ov[a2, a]
                end
            end
            acc ./= length(phs)

            for a in 1:nvb, b in 1:ncb, a2 in 1:nvb, b2 in 1:ncb
                row = bse_index(a, b, ik, ncb, Nk)
                col = bse_index(a2, b2, jk, ncb, Nk)
                val = -W * acc[b, b2, a, a2] * pref
                if ik == jk && a == a2 && b == b2
                    # Diagonal kinetic term  (for k=k', same bands: overlap = 1)
                    val = (E[cbands[b], ik] - E[vbands[a], ik]) - W_q0 * abs2(acc[b, b2, a, a2]) * pref
                end
                H_BSE[row, col] = val
            end
        end
    end

    # Diagonalize BSE matrix
    energies, envelopes = eigen(H_BSE)

    idx = sortperm(real.(energies))
    return real.(energies[idx]), envelopes[:, idx]
end
