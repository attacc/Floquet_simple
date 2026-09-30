#
# Simple electron-phonon model for monolayer hBN: short-range (SSH) + polar (2D Froehlich)
#
#  * Phonons : in-plane nearest-neighbour force-constant model (radial K_r + tangential K_t),
#              2 atoms per cell (B = orbital 1, N = orbital 2)  -> 4 modes (LA, TA, LO, TO).
#              Out-of-plane (flexural) modes are not included.
#              + non-analytic (dipole-dipole) term that makes the LO branch stiffer than the TO one.
#
#  * e-ph, short range : Su-Schrieffer-Heeger-like modulation of the hopping,
#              t(d) = t_0 exp[-beta (d/a_cc - 1)]   ->   dt/dd = -beta t_0 / a_cc
#
#  * e-ph, polar : potential of the ionic dipoles (Born charges Z_B = -Z_N = Z_star), G = 0 component,
#              screened with a Rytova-Keldysh form and cut with a Gaussian form factor
#
# ---------------------------------------------------------------------------------------------
# Polar terms (atomic units, e^2 = 1). Displacement pattern  u_a(R) = eps_a exp(iq.(R+tau_a)) / sqrt(M_a)
#
#   ionic polarisation   P = (1/A) sum_a Z_a u_a ,   charge  rho = -div P = -i q.P
#   screened 2D Coulomb  v(q) = 2 pi / ( q eps_pol (1 + r0 q) )      (r0 = r0_pol_ang)
#   form factor          g(q) = exp(-q^2 / (2 q_cut^2))     (smeared dipoles; q_cut = Inf -> 1)
#
#   * electron potential energy  dV = -phi = i v(q) g(q) (q.sum_a Z_a u_a)/A  e^{iq.r}
#     -> orbital-diagonal vertex (same on both orbitals, phases of e^{iq.r} cancel in the atomic gauge):
#
#        M_polar(q,nu) = 1_{2x2} * i (2 pi / A) g(q) / (eps_pol (1 + r0 q)) * sum_a Z_a (qhat.eps_a)/sqrt(M_a)
#
#     For q -> 0 this is finite (2D: the 1/q of v cancels the q of the dipole) and only the
#     longitudinal branch couples.
#
#   * energy of the polarisation field -> non-analytic force constants
#
#        Phi^NA_{a al, b be}(q) = 2 pi q g(q)^2 / ( A eps_pol (1 + r0 q) ) * Z_a Z_b qhat_al qhat_be
#
#     which vanishes linearly for q -> 0 (in 2D omega_LO(0) = omega_TO(0)) and pushes the LO branch up
#     at finite q.
#
#   The polar part contains only the G = 0 Fourier component, so it is NOT periodic in the BZ:
#   always use q folded in the first BZ (exciton_phonon_coupling already does that).
# ---------------------------------------------------------------------------------------------
#
# Conventions (same as TB_hBN.jl / BSE_finite_q.jl):
#   * atomic units, atomic gauge: basis |a,k> = sum_R exp(ik.(R+tau_a)) |a,R> / sqrt(N)
#   * H_12(k) = -t_0 sum_n exp(i k.d_n),  d_n = hBN2D.nn[n,:]  (vector atom 1 -> neighbour of atom 2)
#
# The model is returned as a NamedTuple (nmodes, modes, vertex) so that the exciton-phonon
# code in exciton_phonon.jl does not depend on hBN:
#     omega, eps = model.modes(q)        # omega[nu] (Hartree), eps[:,nu] (4-vector, normalised)
#     M          = model.vertex(k, q, e) # norb x norb matrix, <k+q| dH |k> in the orbital basis,
#                                        # WITHOUT the 1/sqrt(2 omega) phonon amplitude
#
# Requires: hBN2D (t_0, nn, a_cc, a_1, a_2) already loaded (TB_hBN.jl).
#

using LinearAlgebra

const AMU2ME       = 1822.888486209     # atomic mass unit in electron masses
const ANG2BOHR_EPH = 1.889726125

"""
    hbn_ep_model(; K_r=0.22, K_t=0.07, beta=2.5, mass_B=10.811, mass_N=14.007,
                   ssh=true, polar=true, Z_star=2.7, r0_pol_ang=12.0, eps_pol=1.0, q_cut=0.3)

Phonons / short range e-ph:
- `K_r`, `K_t` : radial / tangential nearest-neighbour force constants [Hartree/Bohr^2]
                 (K_r+K_t = 0.29 gives omega_LO/TO(Gamma) ~ 170 meV)
- `beta`       : -dln(t)/dln(d), dimensionless (2-3 for sp2 systems)
- masses in amu; `ssh=false` switches the hopping-modulation coupling off

Polar part (`polar=false` switches it off, both in the phonons and in the coupling):
- `Z_star`     : Born effective charge of B (N has the opposite sign) in units of e
- `r0_pol_ang` : screening length of the 2D layer [Angstrom] (default = r0 of the BSE example)
- `eps_pol`    : background dielectric constant
- `q_cut`      : Gaussian form-factor cutoff [1/Bohr]; the long-range (G=0) treatment is only
                 meaningful at small q. `Inf` disables it.
"""
function hbn_ep_model(; K_r::Real=0.22, K_t::Real=0.07, beta::Real=2.5,
                      mass_B::Real=10.811, mass_N::Real=14.007,
                      ssh::Bool=true, polar::Bool=true, Z_star::Real=2.7,
                      r0_pol_ang::Real=12.0, eps_pol::Real=1.0, q_cut::Real=0.3)

    bonds = [Float64[real(hBN2D.nn[n, 1]), real(hBN2D.nn[n, 2])] for n in 1:3]
    ehat  = [b ./ norm(b) for b in bonds]
    Id2   = Matrix{Float64}(I, 2, 2)
    Phi   = [K_r .* (e * e') .+ K_t .* (Id2 .- e * e') for e in ehat]
    Phi_sum = sum(Phi)

    M1 = mass_B * AMU2ME
    M2 = mass_N * AMU2ME
    kappa = ssh ? beta * hBN2D.t_0 / hBN2D.a_cc : 0.0      # |dt/dd|  [Hartree/Bohr]

    # polar ingredients
    A_cell = abs(hBN2D.a_1[1] * hBN2D.a_2[2] - hBN2D.a_1[2] * hBN2D.a_2[1])   # Bohr^2
    r0_pol = r0_pol_ang * ANG2BOHR_EPH
    Zc     = (Float64(Z_star), -Float64(Z_star))
    smear(q)  = isfinite(q_cut) ? exp(-0.5 * (q / q_cut)^2) : 1.0
    screen(q) = 1.0 / (eps_pol * (1.0 + r0_pol * q))

    function modes(q::AbstractVector{<:Real})
        D = zeros(ComplexF64, 4, 4)
        D[1:2, 1:2] = Phi_sum ./ M1
        D[3:4, 3:4] = Phi_sum ./ M2
        D12 = zeros(ComplexF64, 2, 2)
        for n in 1:3
            D12 .-= Phi[n] .* cis(dot(q, bonds[n]))
        end
        D12 ./= sqrt(M1 * M2)
        D[1:2, 3:4] = D12
        D[3:4, 1:2] = D12'

        # non-analytic (dipole-dipole) term, rank one
        qn = norm(q)
        if polar && qn > 1e-10
            qh = q ./ qn
            w = [Zc[1] * qh[1] / sqrt(M1), Zc[1] * qh[2] / sqrt(M1),
                 Zc[2] * qh[1] / sqrt(M2), Zc[2] * qh[2] / sqrt(M2)]
            pref = 2.0 * pi * qn * smear(qn)^2 * screen(qn) / A_cell
            D .+= pref .* (w * w')
        end

        F = eigen(Hermitian(D))
        return sqrt.(max.(F.values, 0.0)), F.vectors
    end

    function vertex(k::AbstractVector{<:Real}, q::AbstractVector{<:Real}, epsv::AbstractVector)
        e1 = epsv[1:2]
        e2 = epsv[3:4]

        # short range: hopping modulation
        M12 = 0.0im
        M21 = 0.0im
        for n in 1:3
            d  = bonds[n]
            s1 = dot(ehat[n], e1) / sqrt(M1)
            s2 = dot(ehat[n], e2) / sqrt(M2)
            M12 += cis(dot(k .+ q, d)) * s2 - cis(dot(k, d)) * s1
            M21 += cis(-dot(k, d)) * s2 - cis(-dot(k .+ q, d)) * s1
        end
        Mat = ComplexF64[0.0 kappa*M12; kappa*M21 0.0]

        # long range: Froehlich-like on-site potential of the ionic dipoles
        qn = norm(q)
        if polar && qn > 1e-10
            qh = q ./ qn
            S  = Zc[1] * dot(qh, e1) / sqrt(M1) + Zc[2] * dot(qh, e2) / sqrt(M2)
            C  = 1.0im * (2.0 * pi / A_cell) * screen(qn) * smear(qn) * S
            Mat[1, 1] += C
            Mat[2, 2] += C
        end
        return Mat
    end

    return (nmodes=4, modes=modes, vertex=vertex,
            K_r=K_r, K_t=K_t, beta=beta, mass_B=mass_B, mass_N=mass_N,
            ssh=ssh, polar=polar, Z_star=Z_star, r0_pol_ang=r0_pol_ang,
            eps_pol=eps_pol, q_cut=q_cut)
end

"""
    phonon_dispersion(model, qpath)

Phonon frequencies (Hartree) on a list of q points: nmodes x nq matrix.
"""
function phonon_dispersion(model, qpath)
    om = zeros(Float64, model.nmodes, length(qpath))
    for (iq, q) in enumerate(qpath)
        om[:, iq] = model.modes(collect(Float64, q))[1]
    end
    return om
end

# =============================================================================================
# Continuum acoustic-phonon model (deformation potential)
# =============================================================================================
#
#  * Phonons : linear dispersion  omega = v |q| ,  TA (polarisation _|_ q) and LA (polarisation || q).
#              Both atoms of the cell move together: u_a = e_pol exp(iq.r) / sqrt(M_cell)
#              (M_cell = M_B + M_N).
#  * e-ph    : on-site deformation potential, orbital dependent,
#                  dV_a = D_a div(u)   ->   M_aa(q) = i D_a (q . e_pol) / sqrt(M_cell) * g(q)
#              so only the LA branch couples (div u = 0 for TA). D_B (orbital 1) and D_N (orbital 2)
#              are the dilation potentials of the two on-site levels; for a neutral exciton what matters
#              is mainly their difference (gap deformation potential).
#              The coupling is that of a long-wavelength mode, so it is cut by the form factor
#              g(q) = exp(-q^2/(2 q_cut^2)).
#
# Same interface as hbn_ep_model (nmodes = 2, modes(q), vertex(k,q,e)), so it can be used in its place
# in exciton_phonon_coupling. Do NOT add its results to those of hbn_ep_model: the latter already
# contains LA/TA branches (the two lowest ones). Optical branches are only in hbn_ep_model.
#

const AUVEL_MS     = 2.18769126e6      # atomic unit of velocity [m/s]
const HA2EV_EPH    = 27.211396132

"""
    hbn_acoustic_model(; v_LA=14.0, v_TA=10.3, D_B=2.0, D_N=-2.0,
                         mass_B=10.811, mass_N=14.007, q_cut=0.3)

- `v_LA`, `v_TA` : sound velocities [km/s] (defaults reproduce the small-q slopes of the
                   nearest-neighbour force-constant model of `hbn_ep_model`)
- `D_B`, `D_N`   : deformation potentials of the on-site levels of orbital 1 (B) and 2 (N) [eV].
                   The default values are placeholders (gap deformation potential 4 eV): tune them,
                   e.g. on a DFT calculation.
- `q_cut`        : Gaussian cutoff of the coupling [1/Bohr]; `Inf` disables it.
"""
function hbn_acoustic_model(; v_LA::Real=14.0, v_TA::Real=10.3,
                            D_B::Real=2.0, D_N::Real=-2.0,
                            mass_B::Real=10.811, mass_N::Real=14.007,
                            q_cut::Real=0.3)

    vLA   = v_LA * 1e3 / AUVEL_MS
    vTA   = v_TA * 1e3 / AUVEL_MS
    Mcell = (mass_B + mass_N) * AMU2ME
    Dloc  = (D_B / HA2EV_EPH, D_N / HA2EV_EPH)
    smear(q) = isfinite(q_cut) ? exp(-0.5 * (q / q_cut)^2) : 1.0

    function modes(q::AbstractVector{<:Real})
        qn = norm(q)
        if qn < 1e-12
            return [0.0, 0.0], Matrix{Float64}(I, 2, 2)
        end
        qh  = Float64.(q) ./ qn
        tq  = [-qh[2], qh[1]]
        om  = [vTA * qn, vLA * qn]
        pol = hcat(tq, qh)                 # columns: TA, LA polarisation vectors
        p   = sortperm(om)                 # ascending energy
        return om[p], pol[:, p]
    end

    function vertex(k::AbstractVector{<:Real}, q::AbstractVector{<:Real}, epol::AbstractVector)
        s = 1.0im * dot(q, epol) / sqrt(Mcell) * smear(norm(q))
        return ComplexF64[Dloc[1]*s 0.0; 0.0 Dloc[2]*s]
    end

    return (nmodes=2, modes=modes, vertex=vertex,
            v_LA=v_LA, v_TA=v_TA, D_B=D_B, D_N=D_N, q_cut=q_cut)
end
