#
# Bethe-Salpeter equation at finite center-of-mass momentum Q
# (Tamm-Dancoff, two-band model: 1 valence + 1 conduction band)
#
# Basis of e-h pairs:   |k; Q> = c^+_{c,k+Q} c_{v,k} |GS>
#
#   H_{k,k'}(Q) = [E_c(k+Q) - E_v(k)] delta_{k,k'}
#                 - W(k-k') <u_c(k+Q)|u_c(k'+Q)> <u_v(k')|u_v(k)> / (Nk A_cell)   (direct)
#                 + v(Q)    <u_c(k+Q)|u_v(k)>  <u_c(k'+Q)|u_v(k')>^* / (Nk A_cell) (exchange, optional)
#
# For Q = 0 and exchange=false this reduces to `solve_bse` of BSE.jl.
#
# Conventions (same as BSE.jl):
#   * atomic units (Hartree, Bohr); `lattice.rvectors` are the reciprocal vectors in 1/Bohr
#   * TB eigenvectors are in the "atomic" gauge: u(k+G)_a = exp(-i G.tau_a) u(k)_a
#   * W is the Rytova-Keldysh potential, cell-averaged at q = 0
#
# Requires: LatticeTools (min_image_shifts), a uniform grid from BZ_sampling.generate_unif_grid,
#           and a TB Hamiltonian function  Hamiltonian(k::Vector) -> Matrix  (k in 1/Bohr).
#

using LinearAlgebra
using Base.Threads
using .LatticeTools: Lattice, min_image_shifts
using .hBN2D: rytova_keldysh_kernel
using .Units


include("units.jl")
"""
    BSEKernel

Q-independent part of the BSE (direct screened interaction and umklapp phases).
Because the k-grid is uniform, k_i - k_j depends only on the integer index difference,
so the kernel is stored on the (2n1-1)x(2n2-1) table of index differences instead of Nk^2 pairs.
"""
struct BSEKernel
    nk::Int
    n1::Int
    n2::Int
    norb::Int
    pref::Float64                              # area_per_k / (4 pi^2) = 1/(Nk A_cell)
    eps_bg::Float64
    W_diag::Float64                            # diagonal direct term (Hartree), q=0 averaged
    V::Matrix{Float64}                         # -W(q)*pref for each index difference
    phases::Matrix{Vector{Vector{ComplexF64}}} # umklapp phase vectors for each index difference
    ikx::Vector{Int}                           # integer grid indices (1-based) of each k-point
    iky::Vector{Int}
end

"""
    build_bse_kernel(k_grid, lattice, orbitals, r0_ang, eps_bg)

Precomputes the Q-independent BSE ingredients. `r0_ang` is the screening length in Angstrom.
"""
function build_bse_kernel(k_grid, lattice, orbitals, r0_ang::Float64, eps_bg::Float64)
    Nk = k_grid.nk
    n1 = k_grid.nk_dir[1]
    n2 = k_grid.nk_dir[2]
    r0_au = r0_ang * ANG2BOHR_FQ

    b1 = lattice.rvectors[1]
    b2 = lattice.rvectors[2]
    BZ_area = abs(b1[1] * b2[2] - b1[2] * b2[1])
    area_per_k = BZ_area / Nk
    pref = area_per_k / (4.0 * pi^2)

    # cut-off and cell-averaged W(q=0)  (identical to BSE.jl)
    q0_au = sqrt(area_per_k / pi)
    W_q0 = (2.0 * pi / eps_bg) * (2.0 * pi / area_per_k) * (log(1.0 + r0_au * q0_au) / r0_au)
    W_diag = -W_q0 * pref

    tau = orbitals.tau
    norb = length(tau)

    V = zeros(Float64, 2 * n1 - 1, 2 * n2 - 1)
    phases = Matrix{Vector{Vector{ComplexF64}}}(undef, 2 * n1 - 1, 2 * n2 - 1)

    for dx in -(n1 - 1):(n1 - 1), dy in -(n2 - 1):(n2 - 1)
        sx = dx + n1
        sy = dy + n2
        if dx == 0 && dy == 0
            phases[sx, sy] = [ones(ComplexF64, norb)]
            continue
        end
        dk = b1 .* (dx / n1) .+ b2 .* (dy / n2)     # = k_i - k_j
        q_au, Gs = min_image_shifts(dk, lattice)
        phs = Vector{Vector{ComplexF64}}()
        for G in Gs
            Gp = -G                                  # k_j' = k_j + Gp
            push!(phs, ComplexF64[cis(-dot(Gp, tau[a])) for a in 1:norb])
        end
        phases[sx, sy] = phs
        V[sx, sy] = -rytova_keldysh_kernel(q_au, r0_au, eps_bg, q0_au) * pref
    end

    ikx = [k_grid.ik_map_inv[1, ik] for ik in 1:Nk]
    iky = [k_grid.ik_map_inv[2, ik] for ik in 1:Nk]

    return BSEKernel(Nk, n1, n2, norb, pref, eps_bg, W_diag, V, phases, ikx, iky)
end

"""
    solve_bse_finite_q(kernel, tb_sol, k_grid, Hamiltonian, Q;
                       nstates=4, exchange=false, return_vectors=false,
                       v_band=1, c_band=2)

Solve the BSE for center-of-mass momentum `Q` (Cartesian, 1/Bohr, any vector, not
necessarily on the k-grid).

Returns a NamedTuple `(energies, vectors, continuum)`:
- `energies`  : lowest `nstates` exciton energies (Hartree)
- `vectors`   : Nk x nstates envelopes A_S(k) (or `nothing`)
- `continuum` : min_k [E_c(k+Q) - E_v(k)], lower edge of the e-h continuum (Hartree)

`exchange=true` adds the bare (G=0) exchange term v(Q) = 2 pi / (eps_bg |Q|), which is
omitted for |Q| ~ 0 (as in BSE.jl at Q=0).
"""
function solve_bse_finite_q(kernel::BSEKernel, tb_sol, k_grid, Hamiltonian, Q::AbstractVector{<:Real};
                            nstates::Int=4, exchange::Bool=false, return_vectors::Bool=false,
                            v_band::Int=1, c_band::Int=2)
    Nk = kernel.nk
    norb = kernel.norb
    Qv = collect(Float64, Q)
    Qnorm = norm(Qv)

    # valence states at k (from the grid) 
    E_v = tb_sol.eigenval[v_band, :]
    UV = tb_sol.eigenvec[:, v_band, :]

    # conduction states at k+Q
    if Qnorm < 1e-12
        E_c = tb_sol.eigenval[c_band, :]
        UC = tb_sol.eigenvec[:, c_band, :]
    else
        E_c = zeros(Float64, Nk)
        UC = zeros(ComplexF64, norb, Nk)
        Threads.@threads for ik in 1:Nk
            H = Hamiltonian(k_grid.kpt[:, ik] .+ Qv)
            F = eigen(Hermitian(Matrix(H)))
            E_c[ik] = F.values[c_band]
            UC[:, ik] = F.vectors[:, c_band]
        end
    end

    continuum = minimum(E_c .- E_v)

    # exchange ingredients
    do_x = exchange && Qnorm > 1e-8
    rho = zeros(ComplexF64, Nk)
    vQ = 0.0
    if do_x
        vQ = 2.0 * pi / (kernel.eps_bg * Qnorm)
        for ik in 1:Nk
            s = 0.0im
            for a in 1:norb
                s += conj(UC[a, ik]) * UV[a, ik]
            end
            rho[ik] = s
        end
    end

    H_BSE = zeros(ComplexF64, Nk, Nk)
    n1, n2 = kernel.n1, kernel.n2

    Threads.@threads for ik in 1:Nk
        # diagonal
        d = (E_c[ik] - E_v[ik]) + kernel.W_diag
        if do_x
            d += kernel.pref * vQ * abs2(rho[ik])
        end
        H_BSE[ik, ik] = d

        # upper triangle, lower one by hermiticity
        for jk in (ik + 1):Nk
            sx = kernel.ikx[ik] - kernel.ikx[jk] + n1
            sy = kernel.iky[ik] - kernel.iky[jk] + n2
            phs = kernel.phases[sx, sy]

            acc = 0.0im
            for ph in phs
                sc = 0.0im
                sv = 0.0im
                for a in 1:norb
                    sc += conj(UC[a, ik]) * ph[a] * UC[a, jk]
                    sv += conj(ph[a]) * conj(UV[a, jk]) * UV[a, ik]
                end
                acc += sc * sv
            end
            h = kernel.V[sx, sy] * (acc / length(phs))

            if do_x
                h += kernel.pref * vQ * rho[ik] * conj(rho[jk])
            end

            H_BSE[ik, jk] = h
            H_BSE[jk, ik] = conj(h)
        end
    end

    nst = min(nstates, Nk)
    if return_vectors
        F = eigen(Hermitian(H_BSE), 1:nst)
        return (energies=F.values, vectors=F.vectors, continuum=continuum)
    else
        vals = eigvals(Hermitian(H_BSE), 1:nst)
        return (energies=vals, vectors=nothing, continuum=continuum)
    end
end

"""
    solve_bse_finite_q(tb_sol, k_grid, lattice, orbitals, Hamiltonian, Q, r0_ang, eps_bg; kwargs...)

Convenience wrapper that builds the kernel and solves a single Q.
"""
function solve_bse_finite_q(tb_sol, k_grid, lattice, orbitals, Hamiltonian, Q, r0_ang::Float64, eps_bg::Float64; kwargs...)
    kernel = build_bse_kernel(k_grid, lattice, orbitals, r0_ang, eps_bg)
    return solve_bse_finite_q(kernel, tb_sol, k_grid, Hamiltonian, Q; kwargs...)
end

"""
    bse_dispersion(qpath, tb_sol, k_grid, lattice, orbitals, Hamiltonian, r0_ang, eps_bg;
                   nstates=4, exchange=false)

Exciton dispersion along a list of Q points (Cartesian, 1/Bohr).
Returns `(E_exc, E_cont)`:
- `E_exc`  : nstates x nQ matrix of exciton energies (Hartree)
- `E_cont` : nQ vector with the lower edge of the e-h continuum (Hartree)
"""
function bse_dispersion(qpath, tb_sol, k_grid, lattice, orbitals, Hamiltonian,
                        r0_ang::Float64, eps_bg::Float64;
                        nstates::Int=4, exchange::Bool=false, verbose::Bool=true)
    kernel = build_bse_kernel(k_grid, lattice, orbitals, r0_ang, eps_bg)
    nq = length(qpath)
    E_exc = zeros(Float64, nstates, nq)
    E_cont = zeros(Float64, nq)

    for (iq, Q) in enumerate(qpath)
        verbose && println("BSE at Q-point ", iq, " / ", nq, "   Q = ", round.(Q, digits=5))
        res = solve_bse_finite_q(kernel, tb_sol, k_grid, Hamiltonian, Q;
                                 nstates=nstates, exchange=exchange)
        E_exc[:, iq] = res.energies
        E_cont[iq] = res.continuum
    end
    return E_exc, E_cont
end

"""
    hex_high_symmetry_points(lattice)

Gamma, K, M (Cartesian, 1/Bohr) for a hexagonal 2D lattice, from its reciprocal vectors.
Works for both 60 and 120 degrees between b1 and b2.
"""
function hex_high_symmetry_points(lattice)
    b1 = Float64.(lattice.rvectors[1])
    b2 = Float64.(lattice.rvectors[2])
    c = dot(b1, b2) / (norm(b1) * norm(b2))
    if abs(norm(b1) - norm(b2)) > 1e-6 * norm(b1) || abs(abs(c) - 0.5) > 1e-6
        error("Lattice does not look hexagonal (|b1|=$(norm(b1)), |b2|=$(norm(b2)), cos=$c)")
    end
    Gamma = zeros(Float64, length(b1))
    K = c > 0 ? (b1 .+ b2) ./ 3 : (2 .* b1 .+ b2) ./ 3
    M = b1 ./ 2
    return (Gamma=Gamma, K=K, M=M)
end

"""
    path_distance(qpath)

Cumulative distance along the path (1/Bohr), for the x axis of the plot.
"""
function path_distance(qpath)
    d = zeros(Float64, length(qpath))
    for i in 2:length(qpath)
        d[i] = d[i-1] + norm(qpath[i] - qpath[i-1])
    end
    return d
end
