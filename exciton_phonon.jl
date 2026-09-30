#
# Exciton-phonon coupling from the finite-Q BSE and phonon-induced lifetime
# of the Q=0 exciton.
#
# Exciton |S,Q> = sum_k A^S_k(Q) c^+_{c,k+Q} c_{v,k} |GS>
#
# Exciton-phonon matrix element (initial |S,0>, final |S',Q>, phonon nu with momentum Q):
#
#   G_{S'S}^nu(Q) = 1/sqrt(2 omega) * sum_k conj(A^{S'}_k(Q)) *
#                   [ A^S_k * g_cc(k,Q)  -  A^S_{k+Q} * g_vv(k,Q) * S_c(k) ]
#
#   g_mn(k,Q) = <u_m(k+Q)| M(k,Q,nu) |u_n(k)>         (M = model.vertex)
#   electron term : c(k) -> c(k+Q)          hole term : v(k+Q) -> v(k)  (minus sign, fermion ordering)
#   S_c(k)        : overlap of the spectator conduction state (k+Q) between the Q=0 basis and the
#                   Q basis -> fixes the arbitrary phases of the numerically computed eigenvectors.
#   k+Q is folded back onto the grid: k+Q = k_j + G, u(k_j+G)_a = exp(-iG.tau_a) u(k_j)_a.
#
# Decay rate of the exciton S at Q=0 (Fermi golden rule, N_Q points of the Q grid):
#
#   Gamma_S = 2 pi / N_Q  sum_{Q,nu,S'} |G_{S'S}^nu(Q)|^2 *
#             [ (n+1) delta(E_S - E_S'(Q) - w) + n delta(E_S - E_S'(Q) + w) ]
#
# (emission of a phonon -Q / absorption of a phonon +Q; hbar*Gamma = linewidth, tau = 1/Gamma).
#
# Requires BSE_finite_q.jl (patched: returns UC), LatticeTools, and an e-ph model
# (NamedTuple with nmodes, modes, vertex; see EPH_hBN.jl).
#

using LinearAlgebra
using Base.Threads
using .LatticeTools: min_image_shifts

const KB_HA_PER_K = 3.166811563e-6      # Boltzmann constant [Hartree/K]

"""
    ExcPhCoupling

Exciton-phonon couplings on the Q grid (all energies in Hartree).
- `Q`     : Q vectors (Cartesian, first BZ)
- `E_i`   : energies of the n_init excitons at Q=0
- `E_f`   : n_final x nQ exciton energies at Q
- `omega` : nmodes x nQ phonon frequencies (modes sorted by energy at each Q)
- `G`     : n_final x n_init x nmodes x nQ couplings G_{S'S}^nu(Q) (includes 1/sqrt(2 omega))
"""
struct ExcPhCoupling
    Q::Vector{Vector{Float64}}
    E_i::Vector{Float64}
    E_f::Matrix{Float64}
    omega::Matrix{Float64}
    G::Array{ComplexF64,4}
end

"""
    exciton_phonon_coupling(kernel, tb_sol, k_grid, lattice, orbitals, Hamiltonian, model;
                            q_step=1, n_init=4, n_final=8, exchange=false, omega_min=1e-7)

Computes G_{S'S}^nu(Q) on the Q grid Q = (i1/n1) b1 + (i2/n2) b2, with i = 0, q_step, 2 q_step, ...
(so the Q grid is the k grid, or a coarser one if q_step>1; n_k must be divisible by q_step).
Each Q is folded into the first BZ. Phonons with omega < omega_min (acoustic at Gamma) are skipped.

`n_init` / `n_final` should not cut a degenerate manifold (e.g. 4 = two doublets at Q=0).
"""
function exciton_phonon_coupling(kernel, tb_sol, k_grid, lattice, orbitals, Hamiltonian, model;
                                 q_step::Int=1, n_init::Int=4, n_final::Int=8,
                                 exchange::Bool=false, omega_min::Float64=1e-7,
                                 v_band::Int=1, c_band::Int=2, verbose::Bool=true)
    Nk  = k_grid.nk
    n1  = k_grid.nk_dir[1]
    n2  = k_grid.nk_dir[2]
    (n1 % q_step == 0 && n2 % q_step == 0) || error("n_k must be divisible by q_step")
    dim = Int(lattice.dim)
    b1  = lattice.rvectors[1]
    b2  = lattice.rvectors[2]
    tau = orbitals.tau
    norb = length(tau)

    UV   = tb_sol.eigenvec[:, v_band, :]     # norb x Nk
    UCtb = tb_sol.eigenvec[:, c_band, :]

    # ---- initial excitons at Q = 0 ----
    res0 = solve_bse_finite_q(kernel, tb_sol, k_grid, Hamiltonian, zeros(Float64, dim);
                              nstates=n_init, exchange=exchange, return_vectors=true,
                              v_band=v_band, c_band=c_band)
    E_i = res0.energies
    Ai  = res0.vectors                        # Nk x n_init

    # ---- Q grid, folded into the first BZ ----
    Qs = Vector{Vector{Float64}}()
    for i1 in 0:q_step:(n1 - 1), i2 in 0:q_step:(n2 - 1)
        Qc = b1 .* (i1 / n1) .+ b2 .* (i2 / n2)
        _, Gs = min_image_shifts(Qc, lattice)
        push!(Qs, Qc .+ Gs[1])
    end
    nQ = length(Qs)
    nm = model.nmodes

    E_f   = zeros(Float64, n_final, nQ)
    omega = zeros(Float64, nm, nQ)
    G     = zeros(ComplexF64, n_final, n_init, nm, nQ)

    gcc = zeros(ComplexF64, Nk)
    gvv = zeros(ComplexF64, Nk)
    idx = zeros(Int, Nk)
    ph  = zeros(ComplexF64, norb, Nk)

    for (iq, Q) in enumerate(Qs)
        verbose && println("Exciton-phonon coupling: Q-point ", iq, " / ", nQ,
                           "   Q = ", round.(Q, digits=4))

        res = solve_bse_finite_q(kernel, tb_sol, k_grid, Hamiltonian, Q;
                                 nstates=n_final, exchange=exchange, return_vectors=true,
                                 v_band=v_band, c_band=c_band)
        E_f[:, iq] = res.energies
        Af  = res.vectors                     # Nk x n_final
        UCn = res.UC                          # conduction vectors at k+Q (Q basis)

        om, epsm = model.modes(Q)
        omega[:, iq] = om

        # k+Q = k_j + G  ->  index j and umklapp phases
        for ik in 1:Nk
            kq = k_grid.kpt[:, ik] .+ Q
            x  = lattice.b_mat_inv * kq
            y1 = x[1] * n1
            y2 = x[2] * n2
            if abs(y1 - round(y1)) > 1e-5 || abs(y2 - round(y2)) > 1e-5
                error("k+Q is not on the k grid")
            end
            j1 = mod(round(Int, y1), n1)
            j2 = mod(round(Int, y2), n2)
            jk = k_grid.ik_map[j1 + 1, j2 + 1, 1]
            idx[ik] = jk
            Gv = kq .- k_grid.kpt[:, jk]
            for a in 1:norb
                ph[a, ik] = cis(-dot(Gv, tau[a]))
            end
        end

        UVQ = ph .* UV[:, idx]                # valence at k+Q  (same state as the Q=0 basis)
        UCi = ph .* UCtb[:, idx]              # conduction at k+Q (Q=0 basis phase)
        Sc  = vec(sum(conj.(UCn) .* UCi, dims=1))

        Ai_shift = Ai[idx, :]

        for nu in 1:nm
            om[nu] < omega_min && continue
            epsv = epsm[:, nu]
            Threads.@threads for ik in 1:Nk
                Mk = model.vertex(k_grid.kpt[:, ik], Q, epsv)
                gcc[ik] = dot(UCn[:, ik], Mk * UCtb[:, ik])
                gvv[ik] = dot(UVQ[:, ik], Mk * UV[:, ik])
            end
            Tm = (gcc .* Ai) .- ((gvv .* Sc) .* Ai_shift)
            G[:, :, nu, iq] = (1.0 / sqrt(2.0 * om[nu])) .* (Af' * Tm)
        end
    end

    return ExcPhCoupling(Qs, E_i, E_f, omega, G)
end

"""
    exciton_phonon_linewidth(cpl; T=0.0, sigma_ev=0.010, deg_tol=1e-6)

Phonon-induced decay rate of the Q=0 excitons (Hartree; hbar*Gamma = linewidth, tau = 1/Gamma).
`sigma_ev` : Gaussian broadening of the energy-conserving delta (converge it together with the Q grid).

Returns a NamedTuple:
- `gamma`      : total rate for each initial exciton
- `gamma_avg`  : rate averaged over degenerate initial states (basis independent; use this one)
- `gamma_mode` : n_init x nmodes contributions of each phonon branch
- `gamma_emis`, `gamma_abs` : emission / absorption parts
- `tau_fs`     : lifetime 1/gamma_avg in fs (Inf if no decay channel)
"""
function exciton_phonon_linewidth(cpl::ExcPhCoupling; T::Real=0.0, sigma_ev::Real=0.010,
                                  deg_tol::Real=1e-6)
    sigma = sigma_ev / ha2ev
    ni = length(cpl.E_i)
    nf, _, nm, nq = size(cpl.G)
    gauss(x) = exp(-x^2 / (2 * sigma^2)) / (sigma * sqrt(2 * pi))
    kT = KB_HA_PER_K * T

    gmode = zeros(Float64, ni, nm)
    gem   = zeros(Float64, ni)
    gab   = zeros(Float64, ni)

    for iq in 1:nq, nu in 1:nm
        w = cpl.omega[nu, iq]
        w <= 0.0 && continue
        nb = T > 0 ? 1.0 / expm1(w / kT) : 0.0
        for S in 1:ni, Sp in 1:nf
            g2 = abs2(cpl.G[Sp, S, nu, iq])
            dE = cpl.E_i[S] - cpl.E_f[Sp, iq]
            em = (nb + 1.0) * gauss(dE - w)
            ab = nb * gauss(dE + w)
            gmode[S, nu] += g2 * (em + ab)
            gem[S] += g2 * em
            gab[S] += g2 * ab
        end
    end

    pref = 2 * pi / nq
    gmode .*= pref
    gem   .*= pref
    gab   .*= pref
    gamma = vec(sum(gmode, dims=2))

    # average over degenerate initial states
    gamma_avg = copy(gamma)
    start = 1
    for S in 1:ni
        if S == ni || abs(cpl.E_i[S + 1] - cpl.E_i[S]) > deg_tol
            gamma_avg[start:S] .= sum(gamma[start:S]) / (S - start + 1)
            start = S + 1
        end
    end

    tau_fs = [g > 0 ? 1.0 / g / fs2aut : Inf for g in gamma_avg]

    return (gamma=gamma, gamma_avg=gamma_avg, gamma_mode=gmode,
            gamma_emis=gem, gamma_abs=gab, tau_fs=tau_fs)
end
