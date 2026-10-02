# Real-time propagation of the single-particle wavefunction |psi_k(t)> = [c_v; c_c]
# using the Schrödinger equation with BSE interaction and dephasing.

using LinearAlgebra
using Base.Threads
include("real_time.jl")

"""
    rt_propagate_psi(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                     tmax, dt, eta, tda=false, nsave=1, verbose=true)

Propagates the state vector using RK4 in atomic units.
"""
function rt_propagate_psi(TB_sol, Dip_h; interaction=nothing, kick=nothing, field=nothing,
                          tmax::Real, dt::Real, eta::Real, tda::Bool=false, nsave::Int=1,
                          verbose::Bool=true)
    TB_sol.h_dim == 2 || error("rt_propagate_psi: two bands required")
    Nk   = size(TB_sol.eigenval, 2)
    ndir = size(Dip_h, 3)
    Ev   = TB_sol.eigenval[1, :]
    Ec   = TB_sol.eigenval[2, :]
    U    = TB_sol.eigenvec                      # U[a, n, k] (orbital a, band n)

    # Interband dipoles only
    D = zeros(ComplexF64, 2, 2, ndir, Nk)
    for ik in 1:Nk, id in 1:ndir
        D[1, 2, id, ik] = Dip_h[1, 2, id, ik]
        D[2, 1, id, ik] = Dip_h[2, 1, id, ik]
    end

    nsteps = ceil(Int, tmax / dt)

    # Initial condition: valence band occupied, |psi_k(0)> = [1, 0]
    Psi = [zeros(ComplexF64, 2) for _ in 1:Nk]
    for ik in 1:Nk
        Psi[ik][1] = 1.0
        Psi[ik][2] = 0.0
    end

    # Apply delta kick: |psi_k(0+)> = exp(-i kappa . D) |psi_k(0)>
    if kick !== nothing
        length(kick) == ndir || error("kick must have $ndir components")
        for ik in 1:Nk
            X = zeros(ComplexF64, 2, 2)
            for id in 1:ndir
                X .+= D[:, :, id, ik] .* kick[id]
            end
            x = abs(X[1, 2])
            Uk = x > 0 ? cos(x) .* Matrix{ComplexF64}(I, 2, 2) .- 1im * (sin(x) / x) .* X :
                         Matrix{ComplexF64}(I, 2, 2)
            Psi[ik] = Uk * Psi[ik]
        end
    end

    # Buffers
    X1 = zeros(ComplexF64, Nk, 2)
    X2 = zeros(ComplexF64, Nk, 2)
    Y1 = zeros(ComplexF64, Nk, 2)
    Y2 = zeros(ComplexF64, Nk, 2)
    Sg = zeros(ComplexF64, 2, 2, Nk)

    k1 = [zeros(ComplexF64, 2) for _ in 1:Nk]
    k2 = [zeros(ComplexF64, 2) for _ in 1:Nk]
    k3 = [zeros(ComplexF64, 2) for _ in 1:Nk]
    k4 = [zeros(ComplexF64, 2) for _ in 1:Nk]
    Psit = [zeros(ComplexF64, 2) for _ in 1:Nk]

    Efun(t) = field === nothing ? zeros(Float64, ndir) : Float64.(field(t))

    # Self-energy builder from wavefunction density matrix rho = |psi><psi|
    function selfenergy!(Sg, Psi)
        for ik in 1:Nk
            cv = Psi[ik][1]
            cc = Psi[ik][2]
            r11 = abs2(cv)
            r12 = cv * conj(cc)
            r21 = cc * conj(cv)
            r22 = abs2(cc)

            if tda
                S11 = 0.0im; S12 = 0.0im; S21 = r21; S22 = 0.0im
            else
                S11 = r11 - 1.0; S12 = r12; S21 = r21; S22 = r22
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

    # Schrödinger RHS: i d|psi>/dt = H|psi> - i eta |psi_c>
    function rhs!(dPsi, Psi_in, Et)
        interaction === nothing || selfenergy!(Sg, Psi_in)
        Threads.@threads for ik in 1:Nk
            cv = Psi_in[ik][1]
            cc = Psi_in[ik][2]

            H11 = Ev[ik]; H22 = Ec[ik]; H12 = 0.0im; H21 = 0.0im
            for id in 1:ndir
                H12 += Et[id] * D[1, 2, id, ik]
                H21 += Et[id] * D[2, 1, id, ik]
            end
            if interaction !== nothing
                H11 += Sg[1, 1, ik]; H22 += Sg[2, 2, ik]
                H12 += Sg[1, 2, ik]; H21 += Sg[2, 1, ik]
            end

            # d(cv)/dt = -i (H11*cv + H12*cc)
            # d(cc)/dt = -i (H21*cv + H22*cc) - eta * cc
            dPsi[ik][1] = -1.0im * (H11 * cv + H12 * cc)
            dPsi[ik][2] = -1.0im * (H21 * cv + H22 * cc) - eta * cc
        end
    end

    # Time loop (RK4)
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
                    cv = Psi[ik][1]; cc = Psi[ik][2]
                    r12 = cv * conj(cc)
                    r21 = cc * conj(cv)
                    s += real(r12 * D[2, 1, id, ik] + r21 * D[1, 2, id, ik])
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
    rt_spectrum_psi(TB_sol, Dip_h, lattice, freqs; kwargs...)
"""
function rt_spectrum_psi(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=[1.0, 0.0],
                         kappa::Real=1e-4, tmax::Real, dt::Real, eta::Real, tda::Bool=false,
                         damping::Real=0.0, verbose::Bool=true)
    p = Float64.(pol) ./ norm(pol)
    res = rt_propagate_psi(TB_sol, Dip_h; interaction=interaction, kick=kappa .* p,
                           tmax=tmax, dt=dt, eta=eta, tda=tda, verbose=verbose)
    Pp = vec(sum(res.P .* p, dims=1))
    chi = rt_susceptibility(res.times, Pp, freqs, lattice; kick=kappa, damping=damping)
    return chi, res.times, Pp
end
