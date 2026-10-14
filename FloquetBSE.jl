#
# Floquet-Bethe-Salpeter equation for a tight-binding model with any number of bands
# (length gauge, density matrix)
#
# ---------------------------------------------------------------------------------------------
# Band basis of H(k): bands 1..nv valence (occupied), nv+1..h_dim conduction.  Same equation of motion
# as real_time.jl, written for h_dim x h_dim density matrices:
#
#   d rho_k/dt = -i [ H_k(t), rho_k ] - Gamma(rho_k)
#   H_k(t)     = diag(E_n(k)) + E(t).D_k + Sigma_k[ rho - rho0 ]          (static screened exchange)
#   rho0       = diag(1,..,1,0,..,0)   (nv ones)
#   Gamma(rho) : eta * rho_nm  (n != m, dephasing of all coherences, T2 = 1/eta)
#                gamma1 * (rho_nn - rho0_nn)  (population relaxation, T1 = 1/gamma1)
#
#   Sigma^orb_ab(k) = sum_k' M^{ab}_{kk'} drho^orb_ab(k'),  drho^orb = U drho U^dagger,  Sigma = U^dagger Sigma^orb U
#   M^{ab}_{kk'}    = V_{kk'} (1/|G|) sum_G phi^G_a conj(phi^G_b)
#   (rt_build_interaction_general of real_time_psi.jl; for two orbitals rt_build_interaction (V, C) is accepted:
#    M^{11} = M^{22} = V, M^{12} = C, M^{21} = conj(C))
#
# for a monochromatic field  E(t) = E0 e cos(w t), looking for the time-periodic steady state
#
#   rho_k(t) = sum_{n=-N..N} rho_{k,n} exp(-i n w t),     rho_{k,-n} = rho_{k,n}^dagger
#
#   [ -i n w + i[H_0,.] + Gamma ] rho_n + i sum_{m != 0} [ H_m, rho_{n-m} ] = delta_{n0} gamma1 rho0
#
# with H_{+-1} = (E0/2) e.D + Sigma_{+-1}, H_m = Sigma_m otherwise, Sigma_m = Sigma[rho_m - delta_{m0} rho0].
#
# 1) floquet_bse_linear : first order in E0. Only the valence-conduction coherences change. Writing
#    rho_cv(k,t) = a_p e^{-iwt} + b_p e^{+iwt} for every pair p = (v,c,k):
#
#       | Delta - w - i eta + A        B                 | | a    |      | d    |
#       |      B^*            Delta + w + i eta + A^*    | | b^*  |  = - | d^*  |      (E0/2 = 1)
#
#    A_{p,p'} = dSigma_cv(k)/drho_c'v'(k')  = multi-band BSE kernel of BSE.jl (resonant block)
#    B_{p,p'} = dSigma_cv(k)/drho_v'c'(k')  = resonant-antiresonant coupling,   d_p = e.D_cv(k)
#    pair index (same as bse_index of BSE.jl):  p = k + Nk*((c-nv-1) + nc*(v-1))
#    chi(w) = (16 pi/Omega_BZ) P_1/(E0/2),  P_1 = -(1/Nk) sum_p [ a_p d_p^* + b_p^* d_p ]
#
#      tda=true,  antires=false : Im chi = eps2 of the TDA BSE (solve_bse)
#      tda=true,  antires=true  : = rt_spectrum(...; tda=true)       (two bands)
#      tda=false, antires=true  : = rt_spectrum(...; tda=false)      (two bands, full BSE)
#    Only the dephasing eta enters: populations are second order, gamma1 is not needed.
#
# 2) floquet_bse_steady_state : any E0 (non-perturbative). For a given Sigma the problem is linear and
#    k-diagonal (one h_dim^2 (2N+1) linear system per k, solved exactly: Rabi physics, Autler-Townes, harmonics);
#    the self-consistency on Sigma is solved with a Jacobian-free Newton-GMRES method, with a
#    continuation (ramp) in the field amplitude E0 to follow the physical branch.
#    gamma1 > 0 is REQUIRED: without population relaxation the n=0 populations are undetermined
#    (singular Floquet-Liouville matrix), i.e. there is no steady state without a bath.
#    The plain fixed-point iteration Sigma -> Sigma[rho[Sigma]] does not converge near bound excitons
#    (spectral radius of (Delta-w-i eta)^{-1} K is ~1 there), hence Newton.
#
# For h_dim = 2, nv = 1 the results are identical to the previous two-band implementation.
#
# Requirements: TB_sol from Solve_TB_on_grid, Dip_h from Build_Dipole, and the interaction from
#               rt_build_interaction (real_time.jl, two orbitals) or rt_build_interaction_general
#               (real_time_psi.jl, any number of orbitals) applied to build_bse_kernel (BSE_finite_q.jl).
# ---------------------------------------------------------------------------------------------
#
using LinearAlgebra
using Base.Threads

"""
    fl_orbital_kernel(interaction)

Array `M[k, k', a, b]` of the orbital screened-exchange kernel, from either `rt_build_interaction_general`
(field `M`) or the two-orbital `rt_build_interaction` (fields `V`, `C`).
"""
function fl_orbital_kernel(interaction)
    hasproperty(interaction, :M) && return Array{ComplexF64,4}(interaction.M)
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

# k-range of the valence/conduction pair (iv, ic) in the pair basis (iv = 1..nv, ic = 1..nc)
fl_pair_range(iv::Int, ic::Int, nc::Int, Nk::Int) = (Nk * ((ic - 1) + nc * (iv - 1)) + 1):(Nk * ((ic - 1) + nc * (iv - 1)) + Nk)

"""
    fl_bse_blocks(TB_sol, interaction; nv=h_dim÷2)

Pair-basis blocks of the linearised screened-exchange self-energy (dimension nv*nc*Nk):
`A[p, p'] = dSigma_cv(k)/drho_c'v'(k')` (the TDA BSE kernel of `BSE.jl`) and
`B[p, p'] = dSigma_cv(k)/drho_v'c'(k')` (resonant-antiresonant coupling), with p = (v, c, k) ordered as in `bse_index`.
"""
function fl_bse_blocks(TB_sol, interaction; nv::Int=size(TB_sol.eigenval, 1) ÷ 2)
    U = TB_sol.eigenvec
    norb, h, Nk = size(U)
    nc = h - nv
    (1 <= nv < h) || error("fl_bse_blocks: need 1 <= nv < number of bands ($h)")
    M  = fl_orbital_kernel(interaction)
    (size(M, 1) == Nk && size(M, 3) == norb) || error("interaction kernel does not match the k-grid / orbitals")
    Np = nv * nc * Nk
    A  = zeros(ComplexF64, Np, Np)
    B  = zeros(ComplexF64, Np, Np)
    for iv in 1:nv, ic in 1:nc, iv2 in 1:nv, ic2 in 1:nc
        v, c, v2, c2 = iv, nv + ic, iv2, nv + ic2
        r  = fl_pair_range(iv, ic, nc, Nk)
        cl = fl_pair_range(iv2, ic2, nc, Nk)
        for a in 1:norb, b in 1:norb
            l   = [conj(U[a, c, k]) * U[b, v, k] for k in 1:Nk]
            rA  = [U[a, c2, k] * conj(U[b, v2, k]) for k in 1:Nk]
            rB  = [U[a, v2, k] * conj(U[b, c2, k]) for k in 1:Nk]
            Mab = view(M, :, :, a, b)
            A[r, cl] .+= l .* Mab .* transpose(rA)
            B[r, cl] .+= l .* Mab .* transpose(rB)
        end
    end
    return A, B
end

fl_prefactor(lattice) = 16.0 * pi / abs(lattice.rvectors[1][1] * lattice.rvectors[2][2] -
                                         lattice.rvectors[1][2] * lattice.rvectors[2][1])

"""
    floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; interaction=nothing, pol=[1.0,0.0],
                       nv=h_dim÷2, tda=false, antires=true)

Linear Floquet response with the electron-hole interaction (= BSE by inversion, see header), any number of
valence (`nv`) and conduction bands. `interaction=nothing` gives the independent-particle result
(= `Linear_response(...; antires=antires)`). Returns chi(w) with the normalisation of `Linear_response`
(Im chi = eps2 of the BSE).
"""
function floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta::Real; interaction=nothing,
                            pol=[1.0, 0.0], nv::Int=size(TB_sol.eigenval, 1) ÷ 2,
                            tda::Bool=false, antires::Bool=true)
    h  = size(TB_sol.eigenval, 1)
    Nk = size(TB_sol.eigenval, 2)
    (1 <= nv < h) || error("floquet_bse_linear: need 1 <= nv < number of bands ($h)")
    nc = h - nv
    Np = nv * nc * Nk
    p  = Float64.(pol) ./ norm(pol)

    # pair energies and dipoles d_p = e.D_cv(k), ordered as fl_pair_range
    Delta = zeros(Float64, Np)
    d     = zeros(ComplexF64, Np)
    for iv in 1:nv, ic in 1:nc
        c = nv + ic
        r = fl_pair_range(iv, ic, nc, Nk)
        for (j, ik) in enumerate(1:Nk)
            Delta[r[j]] = TB_sol.eigenval[c, ik] - TB_sol.eigenval[iv, ik]
            d[r[j]]     = sum(p[id] * Dip_h[c, iv, id, ik] for id in eachindex(p))
        end
    end

    if interaction === nothing
        A = zeros(ComplexF64, Np, Np)
        B = zeros(ComplexF64, Np, Np)
    else
        A, B = fl_bse_blocks(TB_sol, interaction; nv=nv)
        tda && (B .= 0.0)
    end
    prefactor = fl_prefactor(lattice)
    chi = zeros(ComplexF64, length(freqs))

    if antires
        M   = zeros(ComplexF64, 2 * Np, 2 * Np)
        rhs = vcat(-d, -conj.(d))
        r1  = 1:Np
        r2  = (Np + 1):(2 * Np)
        for (iw, w) in enumerate(freqs)
            M[r1, r1] .= A
            M[r1, r2] .= B
            M[r2, r1] .= conj.(B)
            M[r2, r2] .= conj.(A)
            for q in 1:Np
                M[q, q]           += Delta[q] - w - 1im * eta
                M[Np + q, Np + q] += Delta[q] + w + 1im * eta
            end
            x  = M \ rhs
            P1 = -(sum(x[r1] .* conj.(d)) + sum(x[r2] .* d)) / Nk
            chi[iw] = prefactor * P1
        end
    else
        M = zeros(ComplexF64, Np, Np)
        for (iw, w) in enumerate(freqs)
            M .= A
            for q in 1:Np
                M[q, q] += Delta[q] - w - 1im * eta
            end
            a  = M \ (-d)
            chi[iw] = prefactor * (-sum(a .* conj.(d)) / Nk)
        end
    end
    return chi
end

# ============================== non-perturbative steady state ==============================

# workspace shared by the routines below
struct FLBSEWork
    Nk::Int
    h::Int                       # number of bands
    nv::Int                      # number of valence bands
    norb::Int                    # number of orbitals
    N::Int                       # photon harmonics -N..N
    nb::Int                      # 2N+1
    omega::Float64
    E::Matrix{Float64}           # band energies (h, Nk)
    U::Array{ComplexF64,3}       # TB eigenvectors U[a, n, k]
    Dp::Array{ComplexF64,3}      # e.D (unit polarisation) (h, h, Nk)
    M::Array{ComplexF64,4}       # orbital kernel M[k, k', a, b] (empty if no interaction)
    interacting::Bool
    gam::Vector{Float64}         # Gamma on vec(rho) (column-major): gamma1 on populations, eta on coherences
    src::Vector{ComplexF64}      # gamma1 * rho0 in the n = 0 block
end

function FLBSEWork(TB_sol, Dip_h, interaction, pol, omega, eta, gamma1, N, nv, interband_only)
    h  = size(TB_sol.eigenval, 1)
    Nk = size(TB_sol.eigenval, 2)
    U  = Array{ComplexF64,3}(TB_sol.eigenvec)
    norb = size(U, 1)
    (1 <= nv < h) || error("need 1 <= nv < number of bands ($h)")
    p  = Float64.(pol) ./ norm(pol)
    Dp = zeros(ComplexF64, h, h, Nk)
    for ik in 1:Nk, id in eachindex(p), n in 1:h, m in 1:h
        n == m && continue
        if !interband_only || ((n <= nv) != (m <= nv))
            Dp[n, m, ik] += p[id] * Dip_h[n, m, id, ik]
        end
    end
    nb  = 2 * N + 1
    gam = [n == m ? Float64(gamma1) : Float64(eta) for m in 1:h for n in 1:h]   # n runs fastest
    src = zeros(ComplexF64, h * h * nb)
    for i in 1:nv
        src[h * h * N + (i - 1) * h + i] = gamma1            # element (i,i) of the n = 0 block
    end
    if interaction === nothing
        M = zeros(ComplexF64, 0, 0, 0, 0)
    else
        M = fl_orbital_kernel(interaction)
        (size(M, 1) == Nk && size(M, 3) == norb) || error("interaction kernel does not match the k-grid / orbitals")
    end
    return FLBSEWork(Nk, h, nv, norb, N, nb, Float64(omega), Matrix{Float64}(TB_sol.eigenval), U, Dp, M,
                     interaction !== nothing, gam, src)
end

"""
    fl_sigma(ws, dR)

Band-basis static screened-exchange self-energy of the (h,h,Nk) matrix `dR` (complex-linear in dR,
same expression as the self-energy of `rt_propagate` / `rt_propagate_psi` with tda=false).
"""
function fl_sigma(ws::FLBSEWork, dR::AbstractArray{ComplexF64,3})
    Nk, norb, h = ws.Nk, ws.norb, ws.h
    Ro = zeros(ComplexF64, Nk, norb, norb)
    for k in 1:Nk
        Uk = view(ws.U, :, :, k)
        Ro[k, :, :] = Uk * dR[:, :, k] * Uk'
    end
    So = zeros(ComplexF64, Nk, norb, norb)
    for a in 1:norb, b in 1:norb
        So[:, a, b] = view(ws.M, :, :, a, b) * Ro[:, a, b]
    end
    Sg = zeros(ComplexF64, h, h, Nk)
    for k in 1:Nk
        Uk = view(ws.U, :, :, k)
        Sg[:, :, k] = Uk' * So[k, :, :] * Uk
    end
    return Sg
end

# G(R): self-energy of every harmonic, Sigma_n = Sigma[R_n - delta_{n0} rho0]   (R and result: (h,h,nb,Nk))
function fl_G(ws::FLBSEWork, R::Array{ComplexF64,4}; subtract_rho0::Bool=true)
    Sg = zeros(ComplexF64, ws.h, ws.h, ws.nb, ws.Nk)
    ws.interacting || return Sg
    for i in 1:ws.nb
        dR = R[:, :, i, :]
        if subtract_rho0 && i == ws.N + 1
            for v in 1:ws.nv
                dR[v, v, :] .-= 1.0
            end
        end
        Sg[:, :, i, :] = fl_sigma(ws, dR)
    end
    return Sg
end

"""
    fl_liouvillian(Hm, N, omega, gam)

Floquet-Liouville matrix (dimension h^2 (2N+1)) of one k-point acting on vec(rho_n) (column-major,
blocks n = -N..N):  -i n w rho_n + i sum_m [H_m, rho_{n-m}] + Gamma(rho_n).
`Hm[:, :, m+N+1]` is the m-th harmonic of the Hamiltonian (h x h), m = -N..N; `gam` is Gamma on vec(rho).
"""
function fl_liouvillian(Hm::Array{ComplexF64,3}, N::Int, omega::Real, gam::Vector{Float64})
    h   = size(Hm, 1)
    h2  = h * h
    nb  = 2 * N + 1
    Ih  = Matrix{ComplexF64}(I, h, h)
    L   = zeros(ComplexF64, h2 * nb, h2 * nb)
    for i in 1:nb
        n    = i - N - 1
        rows = (h2 * (i - 1) + 1):(h2 * i)
        for c in 1:h2
            L[rows[c], rows[c]] += -1im * n * omega + gam[c]
        end
        for j in 1:nb
            m = n - (j - N - 1)                  # rho_{n-m} sits in block j
            abs(m) <= N || continue
            H    = Hm[:, :, m + N + 1]
            cols = (h2 * (j - 1) + 1):(h2 * j)
            L[rows, cols] .+= 1im .* (kron(Ih, H) .- kron(transpose(H), Ih))
        end
    end
    return L
end

# i sum_m [dH_m, R_{n-m}] for one k-point (dH, R: (h,h,nb)); returns (h,h,nb)
function fl_commutator(dH::AbstractArray{ComplexF64,3}, R::AbstractArray{ComplexF64,3}, N::Int)
    nb  = 2 * N + 1
    out = zeros(ComplexF64, size(R, 1), size(R, 2), nb)
    for i in 1:nb
        n = i - N - 1
        for mi in 1:nb
            m = mi - N - 1
            j = n - m + N + 1
            1 <= j <= nb || continue
            H  = dH[:, :, mi]
            Rj = R[:, :, j]
            out[:, :, i] .+= 1im .* (H * Rj .- Rj * H)
        end
    end
    return out
end

# for a given Sigma (h,h,nb,Nk) and field amplitude E0: exact per-k solution, returns (R, LU factors)
function fl_solve_k(ws::FLBSEWork, Sg::Array{ComplexF64,4}, E0::Real)
    N, nb, Nk, h = ws.N, ws.nb, ws.Nk, ws.h
    R   = zeros(ComplexF64, h, h, nb, Nk)
    LT  = typeof(lu(Matrix{ComplexF64}(I, h * h * nb, h * h * nb)))
    lus = Vector{LT}(undef, Nk)
    Threads.@threads for k in 1:Nk
        Hm = Sg[:, :, :, k]                      # copy
        for n in 1:h
            Hm[n, n, N + 1] += ws.E[n, k]
        end
        if N >= 1
            Hm[:, :, N + 2] .+= (E0 / 2) .* ws.Dp[:, :, k]
            Hm[:, :, N]     .+= (E0 / 2) .* ws.Dp[:, :, k]
        end
        F = lu(fl_liouvillian(Hm, N, ws.omega, ws.gam))
        lus[k] = F
        R[:, :, :, k] = reshape(F \ ws.src, h, h, nb)
    end
    return R, lus
end

# Jacobian of G with respect to Sigma (at fixed E0) applied to dS:  Sigma[dR],  dR = -L^{-1} i[dS, R]
function fl_JG(ws::FLBSEWork, R::Array{ComplexF64,4}, lus, dS::Array{ComplexF64,4})
    dR = zeros(ComplexF64, ws.h, ws.h, ws.nb, ws.Nk)
    Threads.@threads for k in 1:ws.Nk
        rhs = -vec(fl_commutator(view(dS, :, :, :, k), view(R, :, :, :, k), ws.N))
        dR[:, :, :, k] = reshape(lus[k] \ rhs, ws.h, ws.h, ws.nb)
    end
    return fl_G(ws, dR; subtract_rho0=false)
end

"""
    fl_gmres(Aop, b; tol=1e-8, m=40, maxrestart=20)

Restarted GMRES (modified Gram-Schmidt, least-squares on the Hessenberg matrix) for Aop(x) = b.
Returns (x, number of matrix-vector products).
"""
function fl_gmres(Aop, b::Vector{ComplexF64}; tol::Real=1e-8, m::Int=40, maxrestart::Int=20)
    n   = length(b)
    x   = zeros(ComplexF64, n)
    bn  = norm(b)
    nmv = 0
    bn == 0 && return x, nmv
    for rs in 1:maxrestart
        r = b .- Aop(x); nmv += 1
        beta = norm(r)
        beta <= tol * bn && break
        Q = zeros(ComplexF64, n, m + 1)
        H = zeros(ComplexF64, m + 1, m)
        Q[:, 1] = r ./ beta
        y = ComplexF64[]
        jlast = 0
        for j in 1:m
            v = Aop(Q[:, j]); nmv += 1
            for i in 1:j
                H[i, j] = dot(Q[:, i], v)
                v .-= H[i, j] .* Q[:, i]
            end
            H[j + 1, j] = norm(v)
            e1 = zeros(ComplexF64, j + 1); e1[1] = beta
            y = H[1:(j + 1), 1:j] \ e1
            res = norm(H[1:(j + 1), 1:j] * y .- e1)
            jlast = j
            breakdown = abs(H[j + 1, j]) <= 1e-14 * beta
            breakdown || (Q[:, j + 1] = v ./ H[j + 1, j])
            (res <= tol * bn || breakdown) && break
        end
        x .+= Q[:, 1:jlast] * y
    end
    return x, nmv
end

"""
    floquet_bse_steady_state(TB_sol, Dip_h, lattice, omega, E0; interaction=nothing, pol=[1.0,0.0],
                             eta, gamma1, nv=h_dim÷2, interband_only=true, N_floq=2, n_ramp=8,
                             Sigma0=nothing, tol=1e-10, maxit=30, gmres_m=40, verbose=true)

Time-periodic steady state of the density-matrix equation of `real_time.jl` (with population relaxation
`gamma1`) under E(t) = E0 e cos(omega t), for any number of bands (`nv` valence), with the screened exchange
treated self-consistently (time-dependent Hartree-Fock/SEX Floquet). `interband_only` keeps only the
valence-conduction dipoles (for two bands this is all of them). The field is ramped in `n_ramp` steps from
E0/n_ramp to E0 (continuation), unless `Sigma0` (a previous `Sigma`, same N_floq) is given, in which case a
single step is done.

Returns a NamedTuple:
- `harmonics` : -N:N
- `P`         : polarisation harmonics along e, P(t) = sum_n P_n exp(-i n w t)   (P_{-n} = conj(P_n))
- `chi1`      : (16 pi/Omega_BZ) P_1/(E0/2), first-order-like susceptibility (-> floquet_bse_linear for E0 -> 0)
- `nc`        : total conduction population per k-point (time average, summed over conduction bands)
- `R`, `Sigma`: harmonics of rho and Sigma, arrays (h,h,2N+1,Nk)
- `residual`, `converged`, `newton_it`, `n_matvec`
"""
function floquet_bse_steady_state(TB_sol, Dip_h, lattice, omega::Real, E0::Real; interaction=nothing,
                                  pol=[1.0, 0.0], eta::Real, gamma1::Real,
                                  nv::Int=size(TB_sol.eigenval, 1) ÷ 2, interband_only::Bool=true,
                                  N_floq::Int=2, n_ramp::Int=8, Sigma0=nothing, tol::Real=1e-10,
                                  maxit::Int=30, gmres_m::Int=40, verbose::Bool=true)
    gamma1 > 0 || error("gamma1 must be > 0: without population relaxation (a bath) there is no steady state")
    eta > 0    || error("eta must be > 0")
    N_floq >= 1 || error("N_floq must be >= 1")
    ws  = FLBSEWork(TB_sol, Dip_h, interaction, pol, omega, eta, gamma1, N_floq, nv, interband_only)
    shp = (ws.h, ws.h, ws.nb, ws.Nk)

    Sg = Sigma0 === nothing ? zeros(ComplexF64, shp) : Array{ComplexF64,4}(Sigma0)
    size(Sg) == shp || error("Sigma0 has the wrong size")
    Elist = (Sigma0 === nothing && interaction !== nothing) ? [E0 * s / n_ramp for s in 1:n_ramp] : [E0]

    local R, lus
    fn = 0.0; it_tot = 0; nmv_tot = 0; converged = true
    for Et in Elist
        R, lus = fl_solve_k(ws, Sg, Et)
        ws.interacting || break
        f  = fl_G(ws, R) .- Sg
        fn = norm(f)
        conv = false
        for it in 1:maxit
            if fn < tol * max(1.0, norm(Sg)) || fn < 1e-14
                conv = true
                break
            end
            it_tot += 1
            Rc, lc = R, lus
            Aop = v -> begin
                vs = reshape(v, shp)
                vec(fl_JG(ws, Rc, lc, Array{ComplexF64,4}(vs)) .- vs)
            end
            dx, nmv = fl_gmres(Aop, -vec(f); tol=1e-3 * min(1.0, fn), m=gmres_m)
            nmv_tot += nmv
            dS  = reshape(dx, shp)
            lam = 1.0
            local Sn, Rn, ln, fnew
            while true
                Sn = Sg .+ lam .* dS
                Rn, ln = fl_solve_k(ws, Sn, Et)
                fnew = fl_G(ws, Rn) .- Sn
                (norm(fnew) < (1 - 1e-4 * lam) * fn || lam < 1e-3) && break
                lam /= 2
            end
            Sg, R, lus, f, fn = Sn, Rn, ln, fnew, norm(fnew)
            verbose && println("  E0 = ", Et, "  Newton ", it, "  |F| = ", fn, "  step ", lam, "  GMRES mv ", nmv)
        end
        conv = conv || fn < tol * max(1.0, norm(Sg)) || fn < 1e-14
        if !conv
            converged = false
            verbose && println("  WARNING: Newton not converged at E0 = ", Et, " (|F| = ", fn, ")")
        end
    end

    # polarisation harmonics along e:  P_n = -(1/Nk) sum_k Tr[rho_{k,n} e.D_k]
    P = zeros(ComplexF64, ws.nb)
    for i in 1:ws.nb, k in 1:ws.Nk
        P[i] -= sum(R[a, b, i, k] * ws.Dp[b, a, k] for a in 1:ws.h, b in 1:ws.h) / ws.Nk
    end
    chi1 = fl_prefactor(lattice) * P[ws.N + 2] / (E0 / 2)
    nc   = sum(real(R[c, c, ws.N + 1, k]) for c in (ws.nv + 1):ws.h, k in 1:ws.Nk) / ws.Nk
    return (harmonics=(-N_floq):N_floq, P=P, chi1=chi1, nc=nc, R=R, Sigma=Sg,
            residual=fn, converged=converged, newton_it=it_tot, n_matvec=nmv_tot)
end
