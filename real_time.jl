#
# Real-time propagation of the single-particle density matrix of a two-band TB model
# (independent particles, IPA, or with the BSE electron-hole interaction) in the length gauge,
# using the dipoles of Dipoles.jl, and spectra from the Fourier transform of the polarisation.
#
# ---------------------------------------------------------------------------------------------
# Equations (atomic units, band basis of H(k): 1 = valence, 2 = conduction)
#
#   rho_k(t) : 2x2 density matrix,  rho_k(0) = diag(1,0)
#
#   i d rho_k/dt = [ H_k(t), rho_k ]  -  i eta ( rho_k - diag(rho_k) )          (dephasing of the coherences)
#
#   H_k(t) = diag(E_v(k),E_c(k))  +  E(t) . D_k  +  Sigma_k[ drho ],     drho = rho - rho(0)
#
#   D_k   : interband dipoles of Build_Dipole (Dip_h[:,:,:,ik], diagonal discarded)
#   Sigma : static screened exchange (Rytova-Keldysh W) of the induced density matrix; in the orbital
#           basis (atomic gauge, rho_orb = U rho U^dagger)
#
#       Sigma_ab(k) = sum_k' V_{kk'} F^{ab}_{kk'} drho_ab(k'),    V_{kk'} = -W(q_{kk'})/(Nk A_cell),
#       F^{ab}_{kk'} = (1/|G|) sum_l phi^(l)_a conj(phi^(l)_b)        (umklapp phases of the min. image)
#
#       V_kk = -Wbar(0)/(Nk A_cell),  F_kk = 1      (same kernel as build_bse_kernel / solve_bse)
#
#   The equilibrium self-energy is assumed to be contained in the TB bands (as in the BSE codes), so only
#   the induced part enters. The linear response of this equation is the BSE:
#     * tda=true  : only the resonant coherence rho_cv acts as a source of Sigma_cv  -> BSE in the
#                   Tamm-Dancoff approximation of BSE.jl (resonant + its complex conjugate)
#     * tda=false : all components of drho -> full BSE (resonant-antiresonant coupling included)
#
#   Polarisation per unit cell (charge -1, H_int = +E.r):   P_alpha(t) = -(1/Nk) sum_k Re Tr[ rho_k D^alpha_k ]
#
#   Spectrum:  chi(w) = P(w)/E(w) = (16 pi / Omega_BZ) * [ int P(t) e^{i w t} dt ] / E(w)
#   (same normalisation as Linear_response and as eps2 of Build_Dielectric_Function:  Im chi = eps2).
#   For a delta kick  E(t) = kappa delta(t)  one has E(w) = kappa and
#   rho(0+) = exp(-i kappa.D) rho(0) exp(+i kappa.D).
#
#   Without interaction  chi(w) = (1/Nk) sum_k |e.D_vc|^2 [ 1/(Delta-w-i eta) + 1/(Delta+w+i eta) ]
#   (Linear_response keeps only the resonant term).
# ---------------------------------------------------------------------------------------------
#
# Requirements: two bands (h_dim = 2), TB_sol from Solve_TB_on_grid, Dip_h from Build_Dipole,
#               build_bse_kernel from BSE_finite_q.jl for the interacting case.
#

using LinearAlgebra
using Base.Threads

"""
    rt_build_interaction(kernel::BSEKernel)

Dense Nk x Nk matrices of the screened-exchange kernel built from `build_bse_kernel`:
- `V` : V_{kk'} (acts on the diagonal orbital elements drho_11, drho_22)
- `C` : V_{kk'} F^{12}_{kk'} (acts on drho_12; for drho_21 use conj(C))
"""
function rt_build_interaction(kernel)
    kernel.norb == 2 || error("rt_build_interaction: two orbitals/bands required")
    Nk = kernel.nk
    n1 = kernel.n1
    n2 = kernel.n2
    V = zeros(ComplexF64, Nk, Nk)
    C = zeros(ComplexF64, Nk, Nk)
    Threads.@threads for ik in 1:Nk
        for jk in 1:Nk
            if ik == jk
                V[ik, jk] = kernel.W_diag
                C[ik, jk] = kernel.W_diag
            else
                sx  = kernel.ikx[ik] - kernel.ikx[jk] + n1
                sy  = kernel.iky[ik] - kernel.iky[jk] + n2
                v   = kernel.V[sx, sy]
                phs = kernel.phases[sx, sy]
                F = 0.0im
                for ph in phs
                    F += ph[1] * conj(ph[2])
                end
                F /= length(phs)
                V[ik, jk] = v
                C[ik, jk] = v * F
            end
        end
    end
    return (V=V, C=C)
end

# element (a,b) of  U S U^dagger  for the 2x2 band-basis matrix S=(S11,S12;S21,S22) at k-point ik
@inline function rt_orb_elem(U, ik, S11, S12, S21, S22, a, b)
    return U[a, 1, ik] * (S11 * conj(U[b, 1, ik]) + S12 * conj(U[b, 2, ik])) +
           U[a, 2, ik] * (S21 * conj(U[b, 1, ik]) + S22 * conj(U[b, 2, ik]))
end

"""
    rt_propagate(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                 tmax, dt, eta, tda=false, nsave=1, verbose=true)

Propagates the density matrix with RK4 (time step `dt`, final time `tmax`, atomic units).

- `interaction` : `nothing` (IPA) or the output of `rt_build_interaction` (BSE)
- `kick`        : vector (one component per Cartesian direction of `Dip_h`): delta kick E(t)=kick*delta(t)
- `field`       : function `t -> Vector` giving E(t) (length = number of directions), optional
- `eta`         : dephasing rate of the coherences (Hartree)
- `tda`         : with interaction, use the Tamm-Dancoff approximation
- `nsave`       : the polarisation is stored every `nsave` steps

Returns `(times, P, Efield)`: `P[alpha, n]` is the polarisation (per cell) and `Efield[alpha, n]`
the applied field (0 for a pure kick).
"""
function rt_propagate(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                      tmax::Real, dt::Real, eta::Real, tda::Bool=false, nsave::Int=1,
                      verbose::Bool=true)
    TB_sol.h_dim == 2 || error("rt_propagate: two bands required")
    Nk   = size(TB_sol.eigenval, 2)
    ndir = size(Dip_h, 3)
    Ev   = TB_sol.eigenval[1, :]
    Ec   = TB_sol.eigenval[2, :]
    U    = TB_sol.eigenvec                      # U[a, n, k]  (orbital a, band n)

    # interband dipoles only
    D = zeros(ComplexF64, 2, 2, ndir, Nk)
    for ik in 1:Nk, id in 1:ndir
        D[1, 2, id, ik] = Dip_h[1, 2, id, ik]
        D[2, 1, id, ik] = Dip_h[2, 1, id, ik]
    end

    nsteps = ceil(Int, tmax / dt)

    # ---------------- initial condition (+ delta kick) ----------------
    R = zeros(ComplexF64, 2, 2, Nk)
    R[1, 1, :] .= 1.0
    if kick !== nothing
        length(kick) == ndir || error("kick must have $ndir components")
        for ik in 1:Nk
            X = zeros(ComplexF64, 2, 2)
            for id in 1:ndir
                X .+= D[:, :, id, ik] .* kick[id]
            end
            x = abs(X[1, 2])
            Uk = x > 0 ? cos(x) .* Matrix{ComplexF64}(I, 2, 2) .- 1im * (sin(x) / x) .* X :
                         Matrix{ComplexF64}(I, 2, 2)           # exp(-iX), X^2 = |x|^2 * 1
            R[:, :, ik] = Uk * R[:, :, ik] * Uk'
        end
    end

    # ---------------- buffers ----------------
    X1 = zeros(ComplexF64, Nk, 2)
    X2 = zeros(ComplexF64, Nk, 2)
    Y1 = zeros(ComplexF64, Nk, 2)
    Y2 = zeros(ComplexF64, Nk, 2)
    Sg = zeros(ComplexF64, 2, 2, Nk)
    k1 = zeros(ComplexF64, 2, 2, Nk)
    k2 = similar(k1)
    k3 = similar(k1)
    k4 = similar(k1)
    Rt = similar(k1)
    Efun(t) = field === nothing ? zeros(Float64, ndir) : Float64.(field(t))

    # ---------------- induced screened-exchange self-energy (band basis) ----------------
    function selfenergy!(Sg, R)
        for ik in 1:Nk
            if tda
                S11 = 0.0im; S12 = 0.0im; S21 = R[2, 1, ik]; S22 = 0.0im
            else
                S11 = R[1, 1, ik] - 1.0; S12 = R[1, 2, ik]; S21 = R[2, 1, ik]; S22 = R[2, 2, ik]
            end
            X1[ik, 1] = rt_orb_elem(U, ik, S11, S12, S21, S22, 1, 1)
            X1[ik, 2] = rt_orb_elem(U, ik, S11, S12, S21, S22, 2, 2)
            X2[ik, 1] = rt_orb_elem(U, ik, S11, S12, S21, S22, 1, 2)
            X2[ik, 2] = conj(rt_orb_elem(U, ik, S11, S12, S21, S22, 2, 1))
        end
        mul!(Y1, interaction.V, X1)
        mul!(Y2, interaction.C, X2)
        for ik in 1:Nk
            s11 = Y1[ik, 1]
            s22 = Y1[ik, 2]
            s12 = Y2[ik, 1]
            s21 = conj(Y2[ik, 2])
            for n in 1:2, m in 1:2
                Sg[n, m, ik] = conj(U[1, n, ik]) * (s11 * U[1, m, ik] + s12 * U[2, m, ik]) +
                               conj(U[2, n, ik]) * (s21 * U[1, m, ik] + s22 * U[2, m, ik])
            end
            if tda
                Sg[1, 2, ik] = conj(Sg[2, 1, ik])
                Sg[1, 1, ik] = 0.0
                Sg[2, 2, ik] = 0.0
            end
        end
    end

    # ---------------- right-hand side ----------------
    function rhs!(dR, R, Et)
        interaction === nothing || selfenergy!(Sg, R)
        for ik in 1:Nk
            H11 = Ev[ik]; H22 = Ec[ik]; H12 = 0.0im; H21 = 0.0im
            for id in 1:ndir
                H12 += Et[id] * D[1, 2, id, ik]
                H21 += Et[id] * D[2, 1, id, ik]
            end
            if interaction !== nothing
                H11 += Sg[1, 1, ik]; H22 += Sg[2, 2, ik]
                H12 += Sg[1, 2, ik]; H21 += Sg[2, 1, ik]
            end
            r11 = R[1, 1, ik]; r12 = R[1, 2, ik]; r21 = R[2, 1, ik]; r22 = R[2, 2, ik]
            dR[1, 1, ik] = -1im * ((H11 * r11 + H12 * r21) - (r11 * H11 + r12 * H21))
            dR[1, 2, ik] = -1im * ((H11 * r12 + H12 * r22) - (r11 * H12 + r12 * H22)) - eta * r12
            dR[2, 1, ik] = -1im * ((H21 * r11 + H22 * r21) - (r21 * H11 + r22 * H21)) - eta * r21
            dR[2, 2, ik] = -1im * ((H21 * r12 + H22 * r22) - (r21 * H12 + r22 * H22))
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
                    s += real(R[1, 2, ik] * D[2, 1, id, ik] + R[2, 1, ik] * D[1, 2, id, ik])
                end
                P[id, isave] = -s / Nk
            end
        end
        rhs!(k1, R, Efun(t))
        @. Rt = R + 0.5 * dt * k1
        rhs!(k2, Rt, Efun(t + 0.5 * dt))
        @. Rt = R + 0.5 * dt * k2
        rhs!(k3, Rt, Efun(t + 0.5 * dt))
        @. Rt = R + dt * k3
        rhs!(k4, Rt, Efun(t + dt))
        @. R = R + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if verbose && n % max(1, div(nsteps, 10)) == 0
            println("  real-time step ", n, " / ", nsteps)
        end
    end
    return (times=times[1:isave], P=P[:, 1:isave], Efield=Eout[:, 1:isave])
end

"""
    rt_susceptibility(times, P, freqs, lattice; kick=nothing, Efield=nothing, damping=0.0)

chi(w) = (16 pi/Omega_BZ) * FT[P](w) / E(w), FT[f](w) = int f(t) exp(i w t) dt (trapezoid rule).
`P` is the polarisation along one direction (vector in time). Give either the kick strength
`kick` (E(w) = kick) or the field `Efield` (same direction, same time grid). `damping` multiplies
P and E by exp(-damping*t) (extra broadening). Im chi is the spectrum eps2 normalised as the BSE.
"""
function rt_susceptibility(times::AbstractVector, P::AbstractVector, freqs::AbstractVector, lattice;
                           kick=nothing, Efield=nothing, damping::Real=0.0)
    (kick === nothing) != (Efield === nothing) || error("give exactly one of kick / Efield")
    b1 = lattice.rvectors[1]
    b2 = lattice.rvectors[2]
    prefactor = 16.0 * pi / abs(b1[1] * b2[2] - b1[2] * b2[1])

    nt = length(times)
    dt = times[2] - times[1]
    wt = fill(dt, nt)
    wt[1] *= 0.5
    wt[end] *= 0.5
    win = exp.(-damping .* times)
    Pw = P .* wt .* win
    Ew = Efield === nothing ? nothing : Efield .* wt .* win

    chi = zeros(ComplexF64, length(freqs))
    Threads.@threads for iw in eachindex(freqs)
        w = freqs[iw]
        ph = cis.(w .* times)
        num = sum(Pw .* ph)
        den = Ew === nothing ? kick : sum(Ew .* ph)
        chi[iw] = prefactor * num / den
    end
    return chi
end

"""
    rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=[1.0,0.0], kappa=1e-4,
                tmax, dt, eta, tda=false, damping=0.0)

Delta kick of strength `kappa` along `pol`, propagation, and chi(w) of the polarisation along `pol`.
Returns `(chi, times, P)` with P the polarisation component along `pol`.
"""
function rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=[1.0, 0.0],
                     kappa::Real=1e-4, tmax::Real, dt::Real, eta::Real, tda::Bool=false,
                     damping::Real=0.0, verbose::Bool=true)
    p = Float64.(pol) ./ norm(pol)
    res = rt_propagate(TB_sol, Dip_h; interaction=interaction, kick=kappa .* p,
                       tmax=tmax, dt=dt, eta=eta, tda=tda, verbose=verbose)
    Pp = vec(sum(res.P .* p, dims=1))
    chi = rt_susceptibility(res.times, Pp, freqs, lattice; kick=kappa, damping=damping)
    return chi, res.times, Pp
end

"""
    gaussian_pulse(E0, t0, sigma, w0=0.0)

Field function t -> E0 * exp(-(t-t0)^2/(2 sigma^2)) * cos(w0 (t-t0)); `E0` is a vector (polarisation x amplitude).
Use with `rt_propagate(...; field=...)` and `rt_susceptibility(...; Efield=...)`.
"""
function gaussian_pulse(E0::AbstractVector, t0::Real, sigma::Real, w0::Real=0.0)
    return t -> E0 .* (exp(-(t - t0)^2 / (2 * sigma^2)) * cos(w0 * (t - t0)))
end
