#
# Floquet-Bethe-Salpeter equation for the two-band TB model (length gauge, density matrix)
#
# ---------------------------------------------------------------------------------------------
# Same equation of motion as real_time.jl (band basis of H(k): 1 = valence, 2 = conduction)
#
#   d rho_k/dt = -i [ H_k(t), rho_k ] - Gamma(rho_k)
#   H_k(t)     = diag(E_v,E_c) + E(t).D_k + Sigma_k[ rho - rho0 ]          (static screened exchange)
#   Gamma(rho) : eta * rho_cv, eta * rho_vc                               (dephasing, T2 = 1/eta)
#                gamma1 * (rho_vv - 1), gamma1 * rho_cc                   (population relaxation, T1 = 1/gamma1)
#
# for a monochromatic field  E(t) = E0 e cos(w t)  and looks for the time-periodic steady state
#
#   rho_k(t) = sum_{n=-N..N} rho_{k,n} exp(-i n w t),     rho_{k,-n} = rho_{k,n}^dagger
#
# (harmonic balance / Floquet-Liouville). The harmonics obey
#
#   [ -i n w + i[H_0,.] + Gamma ] rho_n + i sum_{m != 0} [ H_m, rho_{n-m} ] = delta_{n0} gamma1 rho0
#
# with H_{+-1} = (E0/2) e.D + Sigma_{+-1}, H_m = Sigma_m otherwise, and Sigma_m = Sigma[rho_m - delta_{m0} rho0].
#
# 1) floquet_bse_linear : first order in E0. Writing  rho_cv(t) = a_k e^{-iwt} + b_k e^{+iwt}  gives
#
#       | Delta - w - i eta + A        B                 | | a    |      | d    |
#       |      B^*            Delta + w + i eta + A^*    | | b^*  |  = - | d^*  |      (E0/2 = 1)
#
#    A_kk' = dSigma_cv(k)/drho_cv(k')  = BSE kernel of BSE.jl (resonant block)
#    B_kk' = dSigma_cv(k)/drho_vc(k')  = resonant-antiresonant coupling,   d_k = e.D_cv(k)
#    chi(w) = (16 pi/Omega_BZ) P_1/(E0/2),  P_1 = -(1/Nk) sum_k [ a_k d_k^* + b_k^* d_k ]
#
#    This is the BSE solved by inversion at each frequency:
#      tda=true,  antires=false : Im chi = eps2 of Build_Dielectric_Function (solve_bse, TDA)
#      tda=true,  antires=true  : = rt_spectrum(...; tda=true)
#      tda=false, antires=true  : = rt_spectrum(...; tda=false)  (full BSE)
#    Only the dephasing eta enters: populations are second order, gamma1 is not needed.
#
# 2) floquet_bse_steady_state : any E0 (non-perturbative). For a given Sigma the problem is linear and
#    k-diagonal (one 4(2N+1) linear system per k, solved exactly: Rabi physics, Autler-Townes, harmonics);
#    the self-consistency on Sigma is solved with a Jacobian-free Newton-GMRES method, with a
#    continuation (ramp) in the field amplitude E0 to follow the physical branch.
#    gamma1 > 0 is REQUIRED: without population relaxation the n=0 populations are undetermined
#    (singular Floquet-Liouville matrix), i.e. there is no steady state without a bath.
#    The plain fixed-point iteration Sigma -> Sigma[rho[Sigma]] does not converge near bound excitons
#    (spectral radius of (Delta-w-i eta)^{-1} K is ~1 there), hence Newton.
#
# Requirements: two bands, TB_sol from Solve_TB_on_grid, Dip_h from Build_Dipole,
#               rt_build_interaction (real_time.jl) applied to build_bse_kernel (BSE_finite_q.jl).
# ---------------------------------------------------------------------------------------------
#
using LinearAlgebra
using Base.Threads

"""
    fl_bse_blocks(TB_sol, interaction)

Band-basis blocks of the linearised screened-exchange self-energy (Nk x Nk):
`A[k,k'] = dSigma_cv(k)/drho_cv(k')` (the TDA BSE kernel of `BSE.jl`) and
`B[k,k'] = dSigma_cv(k)/drho_vc(k')` (resonant-antiresonant coupling).
`interaction` is the output of `rt_build_interaction`.
"""
function fl_bse_blocks(TB_sol, interaction)
    U  = TB_sol.eigenvec
    Nk = size(U, 3)
    V  = interaction.V
    C  = interaction.C
    A  = zeros(ComplexF64, Nk, Nk)
    B  = zeros(ComplexF64, Nk, Nk)
    for a in 1:2, b in 1:2
        M  = a == b ? V : (a == 1 ? C : conj.(C))      # orbital kernel M^{ab}
        l  = [conj(U[a, 2, k]) * U[b, 1, k] for k in 1:Nk]
        rA = [U[a, 2, k] * conj(U[b, 1, k]) for k in 1:Nk]
        rB = [U[a, 1, k] * conj(U[b, 2, k]) for k in 1:Nk]
        A .+= l .* M .* transpose(rA)
        B .+= l .* M .* transpose(rB)
    end
    return A, B
end

fl_prefactor(lattice) = 16.0 * pi / abs(lattice.rvectors[1][1] * lattice.rvectors[2][2] -
                                         lattice.rvectors[1][2] * lattice.rvectors[2][1])

"""
    floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; interaction=nothing, pol=[1.0,0.0],
                       tda=false, antires=true)

Linear Floquet response with the electron-hole interaction (= BSE by inversion, see header).
`interaction=nothing` gives the independent-particle result (= `Linear_response(...; antires=antires)`).
Returns chi(w) with the normalisation of `Linear_response` (Im chi = eps2 of the BSE).
"""
function floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta::Real; interaction=nothing,
                            pol=[1.0, 0.0], tda::Bool=false, antires::Bool=true)
    TB_sol.h_dim == 2 || error("floquet_bse_linear: two bands required")
    Nk    = size(TB_sol.eigenval, 2)
    Delta = TB_sol.eigenval[2, :] .- TB_sol.eigenval[1, :]
    p     = Float64.(pol) ./ norm(pol)
    d     = [sum(p[id] * Dip_h[2, 1, id, ik] for id in eachindex(p)) for ik in 1:Nk]   # e.D_cv
    if interaction === nothing
        A = zeros(ComplexF64, Nk, Nk)
        B = zeros(ComplexF64, Nk, Nk)
    else
        A, B = fl_bse_blocks(TB_sol, interaction)
        tda && (B .= 0.0)
    end
    prefactor = fl_prefactor(lattice)
    chi = zeros(ComplexF64, length(freqs))

    if antires
        M   = zeros(ComplexF64, 2 * Nk, 2 * Nk)
        rhs = vcat(-d, -conj.(d))
        r1  = 1:Nk
        r2  = (Nk + 1):(2 * Nk)
        for (iw, w) in enumerate(freqs)
            M[r1, r1] .= A
            M[r1, r2] .= B
            M[r2, r1] .= conj.(B)
            M[r2, r2] .= conj.(A)
            for k in 1:Nk
                M[k, k]           += Delta[k] - w - 1im * eta
                M[Nk + k, Nk + k] += Delta[k] + w + 1im * eta
            end
            x  = M \ rhs
            P1 = -(sum(x[r1] .* conj.(d)) + sum(x[r2] .* d)) / Nk
            chi[iw] = prefactor * P1
        end
    else
        M = zeros(ComplexF64, Nk, Nk)
        for (iw, w) in enumerate(freqs)
            M .= A
            for k in 1:Nk
                M[k, k] += Delta[k] - w - 1im * eta
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
    N::Int                       # photon harmonics -N..N
    nb::Int                      # 2N+1
    omega::Float64
    eta::Float64
    gamma1::Float64
    E::Matrix{Float64}           # band energies (2, Nk)
    U::Array{ComplexF64,3}       # TB eigenvectors U[a, n, k]
    Dp::Array{ComplexF64,3}      # e.D (unit polarisation, interband only) (2, 2, Nk)
    V::Matrix{ComplexF64}        # kernels of rt_build_interaction (or empty)
    C::Matrix{ComplexF64}
    Cc::Matrix{ComplexF64}
    interacting::Bool
    src::Vector{ComplexF64}      # gamma1 * rho0 in the n = 0 block
end

function FLBSEWork(TB_sol, Dip_h, interaction, pol, omega, eta, gamma1, N)
    Nk = size(TB_sol.eigenval, 2)
    p  = Float64.(pol) ./ norm(pol)
    Dp = zeros(ComplexF64, 2, 2, Nk)
    for ik in 1:Nk, id in eachindex(p)
        Dp[1, 2, ik] += p[id] * Dip_h[1, 2, id, ik]
        Dp[2, 1, ik] += p[id] * Dip_h[2, 1, id, ik]
    end
    nb  = 2 * N + 1
    src = zeros(ComplexF64, 4 * nb)
    src[4 * N + 1] = gamma1                  # element (1,1) of the n = 0 block (column-major vec)
    if interaction === nothing
        V = zeros(ComplexF64, 0, 0); C = V; Cc = V
    else
        V = Matrix{ComplexF64}(interaction.V); C = Matrix{ComplexF64}(interaction.C); Cc = conj.(C)
    end
    return FLBSEWork(Nk, N, nb, omega, eta, gamma1, Matrix{Float64}(TB_sol.eigenval),
                     Array{ComplexF64,3}(TB_sol.eigenvec), Dp, V, C, Cc, interaction !== nothing, src)
end

"""
    fl_sigma(ws, dR)

Band-basis static screened-exchange self-energy of the (2,2,Nk) matrix `dR` (complex-linear in dR,
same expression as `selfenergy!` of `rt_propagate` with tda=false).
"""
function fl_sigma(ws::FLBSEWork, dR::AbstractArray{ComplexF64,3})
    Nk = ws.Nk
    Ro = zeros(ComplexF64, 2, 2, Nk)
    for k in 1:Nk
        Uk = view(ws.U, :, :, k)
        Ro[:, :, k] = Uk * dR[:, :, k] * Uk'
    end
    So = zeros(ComplexF64, 2, 2, Nk)
    So[1, 1, :] = ws.V * Ro[1, 1, :]
    So[2, 2, :] = ws.V * Ro[2, 2, :]
    So[1, 2, :] = ws.C * Ro[1, 2, :]
    So[2, 1, :] = ws.Cc * Ro[2, 1, :]
    Sg = zeros(ComplexF64, 2, 2, Nk)
    for k in 1:Nk
        Uk = view(ws.U, :, :, k)
        Sg[:, :, k] = Uk' * So[:, :, k] * Uk
    end
    return Sg
end

# G(R): self-energy of every harmonic, Sigma_n = Sigma[R_n - delta_{n0} rho0]   (R and result: (2,2,nb,Nk))
function fl_G(ws::FLBSEWork, R::Array{ComplexF64,4}; subtract_rho0::Bool=true)
    Sg = zeros(ComplexF64, 2, 2, ws.nb, ws.Nk)
    ws.interacting || return Sg
    for i in 1:ws.nb
        dR = R[:, :, i, :]
        if subtract_rho0 && i == ws.N + 1
            dR[1, 1, :] .-= 1.0
        end
        Sg[:, :, i, :] = fl_sigma(ws, dR)
    end
    return Sg
end

"""
    fl_liouvillian(Hm, N, omega, eta, gamma1)

Floquet-Liouville matrix (dimension 4(2N+1)) of one k-point acting on vec(rho_n) (column-major,
blocks n = -N..N):  -i n w rho_n + i sum_m [H_m, rho_{n-m}] + Gamma(rho_n).
`Hm[:, :, m+N+1]` is the m-th harmonic of the Hamiltonian, m = -N..N.
"""
function fl_liouvillian(Hm::Array{ComplexF64,3}, N::Int, omega::Real, eta::Real, gamma1::Real)
    nb  = 2 * N + 1
    I2  = Matrix{ComplexF64}(I, 2, 2)
    gam = (gamma1, eta, eta, gamma1)             # vec order: 11, 21, 12, 22
    L   = zeros(ComplexF64, 4 * nb, 4 * nb)
    for i in 1:nb
        n    = i - N - 1
        rows = (4 * (i - 1) + 1):(4 * i)
        for c in 1:4
            L[rows[c], rows[c]] += -1im * n * omega + gam[c]
        end
        for j in 1:nb
            m = n - (j - N - 1)                  # rho_{n-m} sits in block j
            abs(m) <= N || continue
            H    = Hm[:, :, m + N + 1]
            cols = (4 * (j - 1) + 1):(4 * j)
            L[rows, cols] .+= 1im .* (kron(I2, H) .- kron(transpose(H), I2))
        end
    end
    return L
end

# i sum_m [dH_m, R_{n-m}] for one k-point (dH, R: (2,2,nb)); returns (2,2,nb)
function fl_commutator(dH::AbstractArray{ComplexF64,3}, R::AbstractArray{ComplexF64,3}, N::Int)
    nb  = 2 * N + 1
    out = zeros(ComplexF64, 2, 2, nb)
    for i in 1:nb
        n = i - N - 1
        for mi in 1:nb
            m = mi - N - 1
            j = n - m + N + 1
            1 <= j <= nb || continue
            H = dH[:, :, mi]
            Rj = R[:, :, j]
            out[:, :, i] .+= 1im .* (H * Rj .- Rj * H)
        end
    end
    return out
end

# for a given Sigma (2,2,nb,Nk) and field amplitude E0: exact per-k solution, returns (R, LU factors)
function fl_solve_k(ws::FLBSEWork, Sg::Array{ComplexF64,4}, E0::Real)
    N, nb, Nk = ws.N, ws.nb, ws.Nk
    R   = zeros(ComplexF64, 2, 2, nb, Nk)
    LT  = typeof(lu(Matrix{ComplexF64}(I, 4 * nb, 4 * nb)))
    lus = Vector{LT}(undef, Nk)
    Threads.@threads for k in 1:Nk
        Hm = Sg[:, :, :, k]                      # copy
        Hm[1, 1, N + 1] += ws.E[1, k]
        Hm[2, 2, N + 1] += ws.E[2, k]
        if N >= 1
            Hm[:, :, N + 2] .+= (E0 / 2) .* ws.Dp[:, :, k]
            Hm[:, :, N]     .+= (E0 / 2) .* ws.Dp[:, :, k]
        end
        F = lu(fl_liouvillian(Hm, N, ws.omega, ws.eta, ws.gamma1))
        lus[k] = F
        R[:, :, :, k] = reshape(F \ ws.src, 2, 2, nb)
    end
    return R, lus
end

# Jacobian of G with respect to Sigma (at fixed E0) applied to dS:  Sigma[dR],  dR = -L^{-1} i[dS, R]
function fl_JG(ws::FLBSEWork, R::Array{ComplexF64,4}, lus, dS::Array{ComplexF64,4})
    dR = zeros(ComplexF64, 2, 2, ws.nb, ws.Nk)
    Threads.@threads for k in 1:ws.Nk
        rhs = -vec(fl_commutator(view(dS, :, :, :, k), view(R, :, :, :, k), ws.N))
        dR[:, :, :, k] = reshape(lus[k] \ rhs, 2, 2, ws.nb)
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
                             eta, gamma1, N_floq=2, n_ramp=8, Sigma0=nothing, tol=1e-10,
                             maxit=30, gmres_m=40, verbose=true)

Time-periodic steady state of the density-matrix equation of `real_time.jl` (with population relaxation
`gamma1`) under E(t) = E0 e cos(omega t), with the screened exchange treated self-consistently
(time-dependent Hartree-Fock/SEX Floquet). The field is ramped in `n_ramp` steps from E0/n_ramp to E0
(continuation), unless `Sigma0` (a previous `Sigma`, same N_floq) is given, in which case a single step is done.

Returns a NamedTuple:
- `harmonics` : -N:N
- `P`         : polarisation harmonics along e, P(t) = sum_n P_n exp(-i n w t)   (P_{-n} = conj(P_n))
- `chi1`      : (16 pi/Omega_BZ) P_1/(E0/2), first-order-like susceptibility (-> floquet_bse_linear for E0 -> 0)
- `nc`        : average conduction population (time average, per k-point)
- `R`, `Sigma`: harmonics of rho and Sigma, arrays (2,2,2N+1,Nk)
- `residual`, `converged`, `newton_it`, `n_matvec`
"""
function floquet_bse_steady_state(TB_sol, Dip_h, lattice, omega::Real, E0::Real; interaction=nothing,
                                  pol=[1.0, 0.0], eta::Real, gamma1::Real, N_floq::Int=2,
                                  n_ramp::Int=8, Sigma0=nothing, tol::Real=1e-10, maxit::Int=30,
                                  gmres_m::Int=40, verbose::Bool=true)
    TB_sol.h_dim == 2 || error("floquet_bse_steady_state: two bands required")
    gamma1 > 0 || error("gamma1 must be > 0: without population relaxation (a bath) there is no steady state")
    eta > 0    || error("eta must be > 0")
    N_floq >= 1 || error("N_floq must be >= 1")
    ws  = FLBSEWork(TB_sol, Dip_h, interaction, pol, Float64(omega), Float64(eta), Float64(gamma1), N_floq)
    shp = (2, 2, ws.nb, ws.Nk)

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
        P[i] -= sum(R[a, b, i, k] * ws.Dp[b, a, k] for a in 1:2, b in 1:2) / ws.Nk
    end
    chi1 = fl_prefactor(lattice) * P[ws.N + 2] / (E0 / 2)
    nc   = sum(real(R[2, 2, ws.N + 1, k]) for k in 1:ws.Nk) / ws.Nk
    return (harmonics=(-N_floq):N_floq, P=P, chi1=chi1, nc=nc, R=R, Sigma=Sg,
            residual=fn, converged=converged, newton_it=it_tot, n_matvec=nmv_tot)
end
