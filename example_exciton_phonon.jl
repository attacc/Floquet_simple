#
# Exciton-phonon coupling on a regular Q grid and phonon-induced lifetime
# of the zero-momentum exciton (hBN, TB + BSE + simple SSH electron-phonon model)
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

include("BSE_finite_q.jl")     # patched version: solve_bse_finite_q also returns UC
include("EPH_hBN.jl")          # phonons + electron-phonon model
include("exciton_phonon.jl")   # exciton-phonon coupling + linewidth

lattice = set_Lattice(2, [a_1, a_2])

# ---------------- parameters ----------------
n_k1 = 36
n_k2 = 36

r0       = 12.0     # screening length [Angstrom]
eps_bg   = 1.0
exchange = false    # exchange is dropped at Q=0, so keep it off for a consistent Q->0 limit

# Q grid = k grid with stride q_step  (36/3 -> 12x12 Q points). Increase for convergence.
q_step   = 3
n_init   = 4        # initial excitons at Q=0 (4 = two doublets; do not cut a degenerate pair)
n_final  = 8        # final excitons at Q

sigma_ev     = 0.010                          # broadening of energy conservation [eV]
temperatures = [0.0, 50.0, 100.0, 200.0, 300.0]   # K

# electron-phonon model (force constants in Ha/Bohr^2, beta = -dlnt/dlnd)
model = hbn_ep_model(K_r=0.22, K_t=0.07, beta=2.5)

# ---------------- TB and BSE kernel ----------------
k_grid = generate_unif_grid(n_k1, n_k2, lattice)
TB_sol = Solve_TB_on_grid(k_grid, Hamiltonian)
kernel = build_bse_kernel(k_grid, lattice, orbitals, r0, eps_bg)

# ---------------- phonon dispersion (check of the model) ----------------
pts   = hex_high_symmetry_points(lattice)
n_steps = 30
qpath = generate_circuit([pts.Gamma, pts.K, pts.M, pts.Gamma], n_steps)
om_path_mev = phonon_dispersion(model, qpath) .* ha2ev .* 1000
xpath = path_distance(qpath)
println("Phonons at Gamma [meV]: ", round.(om_path_mev[:, 1], digits=1))
println("Phonons at K     [meV]: ", round.(om_path_mev[:, n_steps + 1], digits=1))
println("Phonons at M     [meV]: ", round.(om_path_mev[:, 2 * n_steps + 1], digits=1))

# ---------------- exciton-phonon coupling on the Q grid ----------------
cpl = exciton_phonon_coupling(kernel, TB_sol, k_grid, lattice, orbitals, Hamiltonian, model;
                              q_step=q_step, n_init=n_init, n_final=n_final, exchange=exchange)

println("Excitons at Q=0 [eV]: ", cpl.E_i .* ha2ev)

# ---------------- lifetime of the Q=0 excitons ----------------
gam_meV = zeros(Float64, n_init, length(temperatures))
tau_fs  = zeros(Float64, n_init, length(temperatures))
for (it, T) in enumerate(temperatures)
    lw = exciton_phonon_linewidth(cpl; T=T, sigma_ev=sigma_ev)
    gam_meV[:, it] = lw.gamma_avg .* ha2ev .* 1000
    tau_fs[:, it]  = lw.tau_fs
    println("\nT = $T K")
    for S in 1:n_init
        println("  exciton $S  E = ", round(cpl.E_i[S] * ha2ev, digits=4), " eV   ",
                "hbar*Gamma = ", round(gam_meV[S, it], digits=4), " meV   ",
                "tau = ", tau_fs[S, it], " fs")
    end
    println("  mode-resolved hbar*Gamma of exciton 1 [meV] (branches sorted by energy): ",
            round.(lw.gamma_mode[1, :] .* ha2ev .* 1000, digits=4))
end

# ---------------- save ----------------
open("exciton_phonon_linewidth.dat", "w") do io
    println(io, "# T[K]  then hbar*Gamma[meV] for exciton 1..$(n_init) (degenerate-averaged)")
    for (it, T) in enumerate(temperatures)
        println(io, T, " ", join(gam_meV[:, it], " "))
    end
end

open("exciton_phonon_coupling.dat", "w") do io
    println(io, "# Qx Qy [1/Bohr]  omega_nu[meV]...  sum_S' |G_S'1|^2 [meV^2] per branch (exciton 1)")
    for iq in eachindex(cpl.Q)
        g2 = [sum(abs2.(cpl.G[:, 1, nu, iq])) * (ha2ev * 1000)^2 for nu in 1:model.nmodes]
        println(io, join(cpl.Q[iq], " "), " ", join(cpl.omega[:, iq] .* ha2ev .* 1000, " "),
                " ", join(g2, " "))
    end
end

# ---------------- plots ----------------
fig, axs = subplots(1, 2, figsize=(11, 4.5))

ax = axs[1]
for nu in 1:model.nmodes
    ax.plot(xpath, om_path_mev[nu, :], "-")
end
node_idx = [1, n_steps + 1, 2 * n_steps + 1, 3 * n_steps + 1]
ax.set_xticks(xpath[node_idx])
ax.set_xticklabels(["Γ", "K", "M", "Γ"])
for i in node_idx
    ax.axvline(xpath[i], color="k", linewidth=0.5)
end
ax.set_xlim(xpath[1], xpath[end])
ax.set_ylabel("Phonon energy (meV)")
ax.set_title("In-plane phonons (NN force-constant model)")

ax = axs[2]
for S in 1:n_init
    ax.plot(temperatures, gam_meV[S, :], "-o",
            label="E = $(round(cpl.E_i[S] * ha2ev, digits=3)) eV")
end
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("ħΓ (meV)")
ax.set_title("Phonon-induced linewidth of the Q=0 exciton")
ax.legend(fontsize=8)

tight_layout()
savefig("exciton_phonon.png", dpi=200)
println("Saved exciton_phonon.png, exciton_phonon_linewidth.dat, exciton_phonon_coupling.dat")
