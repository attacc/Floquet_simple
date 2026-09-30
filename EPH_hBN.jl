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
