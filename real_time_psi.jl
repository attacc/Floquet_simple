#
# Real-time propagation of the occupied single-particle wavefunctions |psi_{k,i}(t)>, i = 1..nv,
# for a generic number of bands (h_dim) and of valence bands (nv), with the screened-exchange
# (BSE) interaction and dephasing. Same equations as before, written with matrices and sums over
# valence and conduction bands instead of the explicit 2x2 algebra (less efficient, but general).
#
# ---------------------------------------------------------------------------------------------
# Band basis of H(k): bands 1..nv valence (occupied), nv+1..h_dim conduction.
#
#   Psi_k(t)  : h_dim x nv matrix, column i = |psi_{k,i}>,   Psi_k(0) = first nv unit vectors
#   rho_k     = Psi_k Psi_k^dagger = sum_i |psi_{k,i}><psi_{k,i}|,      rho0 = diag(1,..,1,0,..,0)
#
#   i d|psi_{k,i}>/dt = H_k(t) |psi_{k,i}>  -  i eta P_c |psi_{k,i}>     (P_c: projector on the conduction bands)
#
#   H_k(t) = diag(E_n(k)) + E(t).D_k + Sigma_k[ rho - rho0 ]
#
#   Static screened exchange, local in the orbital (atomic-gauge) basis, drho^orb = U drho U^dagger:
#       Sigma^orb_ab(k) = sum_k' M^{ab}_{kk'} drho^orb_ab(k'),        Sigma = U^dagger Sigma^orb U
#       M^{ab}_{kk'}    = V_{kk'} (1/|G|) sum_G phi^G_a conj(phi^G_b)  (umklapp phases; M^{aa} = V)
#   For two orbitals M^{11} = M^{22} = V, M^{12} = C, M^{21} = conj(C) of rt_build_interaction.
#
#   tda=true : only the conduction-valence block rho_cv is a source, and Sigma is restricted to its
#              cv block and the hermitian conjugate (vc block); Sigma_vv = Sigma_cc = 0.
#   tda=false: full drho = rho - rho0.
#
#   D_k : dipoles of Build_Dipole (diagonal discarded); with interband_only=true (default) only the
#         valence-conduction blocks are kept (for two bands this is all of them).
#
#   Polarisation per cell:  P_alpha(t) = -(1/Nk) sum_k Re Tr[ rho_k D^alpha_k ]
#   Delta kick:             Psi_k(0+) = exp(-i kappa.D_k) Psi_k(0)
#
# For h_dim = 2 and nv = 1 the results are identical to the previous 2x2 implementation.
# ---------------------------------------------------------------------------------------------

using LinearAlgebra
using Base.Threads
include("real_time.jl")

"""
    rt_build_interaction_general(kernel::BSEKernel)

Orbital kernel for any number of orbitals: returns `(M = M,)` with
`M[k, k', a, b] = V_{kk'} (1/|G|) sum_G phi^G_a conj(phi^G_b)` (and `W_diag` on the diagonal k = k'),
built from the output of `build_bse_kernel`. For two orbitals it contains the same numbers as
`rt_build_interaction` (V on a = b, C on (1,2), conj(C) on (2,1)).
"""
function rt_build_interaction_general(kernel)
    Nk   = kernel.nk
    n1   = kernel.n1
    n2   = kernel.n2
    norb = kernel.norb
    M = zeros(ComplexF64, Nk, Nk, norb, norb)
    Threads.@threads for ik in 1:Nk
        for jk in 1:Nk
            if ik == jk
                for a in 1:norb, b in 1:norb
                    M[ik, jk, a, b] = kernel.W_diag
                end
            else
                sx  = kernel.ikx[ik] - kernel.ikx[jk] + n1
                sy  = kernel.iky[ik] - kernel.iky[jk] + n2
                v   = kernel.V[sx, sy]
                phs = kernel.phases[sx, sy]
                for a in 1:norb, b in 1:norb
                    F = 0.0im
                    for ph in phs
                        F += ph[a] * conj(ph[b])
                    end
                    M[ik, jk, a, b] = v * F / length(phs)
                end
            end
        end
    end
    return (M=M,)
end

"""
    rt_orbital_kernel(interaction)

Returns the array `M[k, k', a, b]` from either `rt_build_interaction_general` (field `M`) or the
two-orbital `rt_build_interaction` (fields `V`, `C`).
"""
function rt_orbital_kernel(interaction)
    hasproperty(interaction, :M) && return interaction.M
    V  = interaction.V
    C  = interaction.C
    Nk = size(V, 1)
    M  = zeros(ComplexF64, Nk, Nk, 2, 2)
    M[:, :, 1, 1] = V
    M[:, :, 2, 2] = V
    M[:, :, 1, 2] = C
    M[:, :, 2, 1] = conj.(C)
    return M
end

"""
    rt_propagate_psi(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                     tmax, dt, eta, nv=1, tda=false, interband_only=true, nsave=1, verbose=true)

Propagates the occupied wavefunctions with RK4 (atomic units), any number of bands.

- `interaction`   : `nothing` (IPA), `rt_build_interaction(kernel)` (two orbitals) or
                    `rt_build_interaction_general(kernel)` (any number of orbitals)
- `kick`          : delta kick E(t) = kick*delta(t) (one component per direction of `Dip_h`)
- `field`         : function t -> Vector, applied field
- `eta`           : damping of the conduction components (Hartree)
- `nv`            : number of valence (occupied) bands
- `tda`           : Tamm-Dancoff approximation for the self-energy
- `interband_only`: keep only the valence-conduction dipoles

Returns `(times, P, Efield)` as `rt_propagate`.
"""
function rt_propagate_psi(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                          tmax::Real, dt::Real, eta::Real, nv::Int=1, tda::Bool=false,
                          interband_only::Bool=true, nsave::Int=1, verbose::Bool=true)
    h_dim = size(TB_sol.eigenval, 1)
    Nk    = size(TB_sol.eigenval, 2)
    ndir  = size(Dip_h, 3)
    1 <= nv < h_dim || error("rt_propagate_psi: need 1 <= nv < number of bands ($h_dim)")
    E     = TB_sol.eigenval                     # E[n, k]
    U     = TB_sol.eigenvec                     # U[a, n, k] (orbital a, band n)
    norb  = size(U, 1)
    iv    = 1:nv                                # valence bands
    ic    = (nv + 1):h_dim                      # conduction bands

    # dipoles (diagonal discarded; optionally valence-conduction blocks only)
    D = zeros(ComplexF64, h_dim, h_dim, ndir, Nk)
    for ik in 1:Nk, id in 1:ndir, n in 1:h_dim, m in 1:h_dim
        n == m && continue
        if !interband_only || ((n <= nv) != (m <= nv))
            D[n, m, id, ik] = Dip_h[n, m, id, ik]
        end
    end

    M = interaction === nothing ? nothing : rt_orbital_kernel(interaction)
    if M !== nothing
        (size(M, 1) == Nk && size(M, 3) == norb) || error("interaction kernel does not match the k-grid / orbitals")
    end

    nsteps = ceil(Int, tmax / dt)

    # initial condition: the nv valence bands are occupied
    Psi = [Matrix{ComplexF64}(I, h_dim, nv) for _ in 1:Nk]

    # delta kick: Psi(0+) = exp(-i kappa.D) Psi(0)
    if kick !== nothing
        length(kick) == ndir || error("kick must have $ndir components")
        for ik in 1:Nk
            X = zeros(ComplexF64, h_dim, h_dim)
            for id in 1:ndir
                X .+= D[:, :, id, ik] .* kick[id]
            end
            F  = eigen(Hermitian(X))
            Uk = F.vectors * Diagonal(cis.(-F.values)) * F.vectors'
            Psi[ik] = Uk * Psi[ik]
        end
    end

    rho0 = zeros(ComplexF64, h_dim, h_dim)
    for i in iv
        rho0[i, i] = 1.0
    end

    # buffers
    Sg   = zeros(ComplexF64, h_dim, h_dim, Nk)
    Ro   = zeros(ComplexF64, Nk, norb, norb)    # induced density matrix in the orbital basis
    So   = zeros(ComplexF64, Nk, norb, norb)    # self-energy in the orbital basis
    k1   = [zeros(ComplexF64, h_dim, nv) for _ in 1:Nk]
    k2   = [zeros(ComplexF64, h_dim, nv) for _ in 1:Nk]
    k3   = [zeros(ComplexF64, h_dim, nv) for _ in 1:Nk]
    k4   = [zeros(ComplexF64, h_dim, nv) for _ in 1:Nk]
    Psit = [zeros(ComplexF64, h_dim, nv) for _ in 1:Nk]

    Efun(t) = field === nothing ? zeros(Float64, ndir) : Float64.(field(t))

    # ---------------- induced screened-exchange self-energy (band basis) ----------------
    function selfenergy!(Sg, Psi)
        for ik in 1:Nk
            rho = Psi[ik] * Psi[ik]'
            if tda
                S = zeros(ComplexF64, h_dim, h_dim)
                S[ic, iv] = rho[ic, iv]                 # only the conduction-valence coherences
            else
                S = rho .- rho0
            end
            Uk = U[:, :, ik]
            Ro[ik, :, :] = Uk * S * Uk'
        end
        for a in 1:norb, b in 1:norb
            So[:, a, b] = view(M, :, :, a, b) * Ro[:, a, b]
        end
        for ik in 1:Nk
            Uk = U[:, :, ik]
            s  = Uk' * So[ik, :, :] * Uk
            if tda
                t = zeros(ComplexF64, h_dim, h_dim)
                t[ic, iv] = s[ic, iv]
                t[iv, ic] = s[ic, iv]'
                s = t
            end
            Sg[:, :, ik] = s
        end
    end

    # ---------------- Schroedinger RHS:  d Psi/dt = -i H Psi - eta P_c Psi ----------------
    function rhs!(dPsi, Psi_in, Et)
        interaction === nothing || selfenergy!(Sg, Psi_in)
        Threads.@threads for ik in 1:Nk
            H = Matrix{ComplexF64}(Diagonal(E[:, ik]))
            for id in 1:ndir
                H .+= Et[id] .* view(D, :, :, id, ik)
            end
            interaction === nothing || (H .+= view(Sg, :, :, ik))
            d = -1im .* (H * Psi_in[ik])
            d[ic, :] .-= eta .* Psi_in[ik][ic, :]
            dPsi[ik] = d
        end
    end

    # ---------------- time loop (RK4) ----------------
    nsaved = div(nsteps, nsave) + 1
    times  = zeros(Float64, nsaved)
    P      = zeros(Float64, ndir, nsaved)
    Eout   = zeros(Float64, ndir, nsaved)
    isave  = 0

    for n in 0:(nsteps - 1)
        t = n * dt
        if n % nsave == 0
            isave += 1
            times[isave] = t
            Eout[:, isave] = Efun(t)
            for id in 1:ndir
                s = 0.0
                for ik in 1:Nk
                    s += real(tr(Psi[ik]' * view(D, :, :, id, ik) * Psi[ik]))   # Tr[rho D]
                end
                P[id, isave] = -s / Nk
            end
        end

        rhs!(k1, Psi, Efun(t))
        for ik in 1:Nk; Psit[ik] = Psi[ik] + 0.5 * dt * k1[ik]; end

        rhs!(k2, Psit, Efun(t + 0.5 * dt))
        for ik in 1:Nk; Psit[ik] = Psi[ik] + 0.5 * dt * k2[ik]; end

        rhs!(k3, Psit, Efun(t + 0.5 * dt))
        for ik in 1:Nk; Psit[ik] = Psi[ik] + dt * k3[ik]; end

        rhs!(k4, Psit, Efun(t + dt))
        for ik in 1:Nk
            Psi[ik] += (dt / 6.0) * (k1[ik] + 2.0 * k2[ik] + 2.0 * k3[ik] + k4[ik])
        end

        if verbose && n % max(1, div(nsteps, 10)) == 0
            println("  real-time wave function step ", n, " / ", nsteps)
        end
    end
    return (times=times[1:isave], P=P[:, 1:isave], Efield=Eout[:, 1:isave])
end

"""
    rt_spectrum_psi(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=[1.0,0.0], kappa=1e-4,
                    tmax, dt, eta, nv=1, tda=false, interband_only=true, damping=0.0)

Delta kick along `pol`, propagation with `rt_propagate_psi`, and chi(w) (same normalisation as `rt_spectrum`).
"""
function rt_spectrum_psi(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=[1.0, 0.0],
                         kappa::Real=1e-4, tmax::Real, dt::Real, eta::Real, nv::Int=1, tda::Bool=false,
                         interband_only::Bool=true, damping::Real=0.0, verbose::Bool=true)
    p = Float64.(pol) ./ norm(pol)
    res = rt_propagate_psi(TB_sol, Dip_h; interaction=interaction, kick=kappa .* p,
                           tmax=tmax, dt=dt, eta=eta, nv=nv, tda=tda, interband_only=interband_only,
                           verbose=verbose)
    Pp = vec(sum(res.P .* p, dims=1))
    chi = rt_susceptibility(res.times, Pp, freqs, lattice; kick=kappa, damping=damping)
    return chi, res.times, Pp
end
