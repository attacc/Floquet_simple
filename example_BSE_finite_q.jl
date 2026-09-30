#
# Exciton dispersion along Gamma -> K -> M -> Gamma
# (Bethe-Salpeter equation at finite momentum, TB approximation)
#
using LinearAlgebra
using Base.Threads
using PyPlot

include("units.jl")
using .Units

include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

include("lattice.jl")
using .LatticeTools

include("bz_sampling.jl")
using .BZ_sampling

include("BSE_finite_q.jl")

lattice = set_Lattice(2, [a_1, a_2])

# ---------------- parameters ----------------
n_k1 = 36
n_k2 = 36

r0       = 12.0    # screening length [Angstrom]
eps_bg   = 1.0     # 1.0 suspended, 2.45 SiO2
nstates  = 10       # excitonic bands to plot
n_steps  = 15      # Q points per segment
exchange = false   # true: add the bare exchange term (non-analytic dispersion near Gamma)

# ---------------- TB on the k-grid ----------------
k_grid = generate_unif_grid(n_k1, n_k2, lattice)
TB_sol = Solve_TB_on_grid(k_grid, Hamiltonian)

# ---------------- Q path: Gamma -> K -> M -> Gamma ----------------
pts   = hex_high_symmetry_points(lattice)
qpath = generate_circuit([pts.M, pts.Gamma, pts.K], n_steps)

# ---------------- BSE at every Q ----------------
E_exc, E_cont = bse_dispersion(qpath, TB_sol, k_grid, lattice, orbitals, Hamiltonian,
                               r0, eps_bg; nstates=nstates, exchange=exchange)

E_exc_ev  = E_exc  .* ha2ev
E_cont_ev = E_cont .* ha2ev
xpath     = path_distance(qpath)

println("Exciton energies at M     [eV]: ", E_exc_ev[:, 1])
println("Exciton energies at Gamma [eV]: ", E_exc_ev[:, n_steps + 1])
println("Exciton energies at K     [eV]: ", E_exc_ev[:, 2 * n_steps + 1])

# ---------------- save data ----------------
open("exciton_dispersion.dat", "w") do io
    println(io, "# path[1/Bohr]  E_cont[eV]  E_exc_1..E_exc_$(nstates) [eV]")
    for iq in eachindex(qpath)
        println(io, xpath[iq], " ", E_cont_ev[iq], " ", join(E_exc_ev[:, iq], " "))
    end
end

# ---------------- plot ----------------
fig, ax = subplots(figsize=(5, 5))
ax.plot(xpath, E_cont_ev, "--", color="gray", label="e-h continuum edge")
for s in 1:nstates
    ax.plot(xpath, E_exc_ev[s, :], "-o", markersize=3, label="Exciton $s")
end

node_idx = [1, n_steps + 1, 2 * n_steps + 1]
ax.set_xticks(xpath[node_idx])
ax.set_xticklabels(["M", "Γ","K"])
for i in node_idx
    ax.axvline(xpath[i], color="k", linewidth=0.5)
end
ax.set_xlim(xpath[1], xpath[end])
ax.set_ylabel("Energy (eV)")
ax.set_title("Exciton dispersion (exchange = $exchange)")
ax.legend(fontsize=8)
tight_layout()
savefig("exciton_dispersion.png", dpi=200)
println("Saved exciton_dispersion.png and exciton_dispersion.dat")
