#
# Floquet check of the linear response (hBN, TB approximation)
#
# A weak monochromatic field E(t) = E0 cos(w t) shifts the quasi-energy of the Floquet state
# connected to the valence band by the optical Stark shift  -(E0^2/4) chi(w),  so the dynamical
# polarisability (and its imaginary part, the absorption) can be extracted from the Floquet
# quasi-energies and compared with Linear_response.
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

include("Dipoles.jl")
include("Linear_response.jl")

include("FloquetTB.jl")
using .FloquetTB

# ---------------- parameters (energies in eV here, converted to Hartree below) ----------------
n_k1 = 36
n_k2 = 36
dk   = 0.001                        # finite-difference step for the dipoles

Emin_ev, Emax_ev, nE = 6.0, 16.0, 100   # hBN gap = 7.25 eV: the window must contain the absorption
eta_ev  = 0.05
E_probe = [1.0, 0.0]                # polarisation of the probe / drive
E0      = 1e-5                      # a.u., weak field: |E0.D|/2 << eta
N_floq  = 1

freqs = collect(LinRange(Emin_ev / ha2ev, Emax_ev / ha2ev, nE))   # Hartree
eta   = eta_ev / ha2ev
E0_field = E0 .* E_probe

# ---------------- TB, dipoles, linear response ----------------
lattice = set_Lattice(2, [a_1, a_2])
k_grid  = generate_unif_grid(n_k1, n_k2, lattice)
TB_sol  = Solve_TB_on_grid(k_grid, Hamiltonian)

println("Building dipole matrix elements...")
Dip_h, ∇H_w = Build_Dipole(k_grid, lattice, TB_sol, orbitals, Hamiltonian, dk)

println("Linear response...")
chi_lr = Linear_response(TB_sol, Dip_h, freqs, E_probe, eta, lattice)   # resonant term only

# resonant + anti-resonant term, same normalisation (what the Floquet calculation contains)
function chi_full(TB_sol, Dip_h, freqs, E_probe, eta, lattice)
    b1 = lattice.rvectors[1]; b2 = lattice.rvectors[2]
    pref = 16.0 * pi / abs(b1[1] * b2[2] - b1[2] * b2[1])
    nk = size(TB_sol.eigenval, 2)
    chi = zeros(ComplexF64, length(freqs))
    for (iw, w) in enumerate(freqs), ik in 1:nk
        res2  = abs2(sum(Dip_h[1, 2, :, ik] .* E_probe))
        Delta = TB_sol.eigenval[2, ik] - TB_sol.eigenval[1, ik]
        chi[iw] += res2 * (1 / (Delta - w - 1im * eta) + 1 / (Delta + w - 1im * eta))
    end
    return pref .* chi ./ nk
end
chi_ref = chi_full(TB_sol, Dip_h, freqs, E_probe, eta, lattice)

# ---------------- Floquet ----------------
println("Floquet, length gauge (dipoles of Build_Dipole)...")
chi_fl_len = [floquet_linear_response(Hamiltonian, k_grid, TB_sol, lattice, E0_field, w, eta;
                                      Dip_h=Dip_h, N_floq=N_floq) for w in freqs]

println("Floquet, velocity gauge (Peierls substitution H(k+A(t)))...")
chi_fl_vel = [floquet_linear_response(Hamiltonian, k_grid, TB_sol, lattice, E0_field, w, eta;
                                      N_floq=N_floq) for w in freqs]

# ---------------- comparison ----------------
relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
println("\n================ Floquet vs linear response ================")
println("Im chi: length-gauge Floquet vs Linear_response : ", relmax(imag.(chi_fl_len), imag.(chi_lr)))
println("Im chi: velocity-gauge Floquet vs Linear_response: ", relmax(imag.(chi_fl_vel), imag.(chi_lr)))
println("Re chi: length-gauge Floquet vs (res+antires)    : ", relmax(real.(chi_fl_len), real.(chi_ref)))
println("Re chi: velocity-gauge Floquet vs (res+antires)  : ", relmax(real.(chi_fl_vel), real.(chi_ref)))
println("(differences are normalised to the maximum; the anti-resonant term of chi_ref is absent in Linear_response)")

# ---------------- plot ----------------
fx = freqs .* ha2ev
fig, axs = subplots(2, 1, figsize=(7, 7), sharex=true)
axs[1].plot(fx, imag.(chi_lr), "-", label="Linear_response")
axs[1].plot(fx, imag.(chi_fl_len), "--", label="Floquet (length gauge)")
axs[1].plot(fx, imag.(chi_fl_vel), ":", label="Floquet (velocity gauge)")
axs[1].set_ylabel("Im χ  (absorption)")
axs[1].legend()
axs[2].plot(fx, real.(chi_ref), "-", label="res + anti-res")
axs[2].plot(fx, real.(chi_fl_len), "--", label="Floquet (length gauge)")
axs[2].plot(fx, real.(chi_fl_vel), ":", label="Floquet (velocity gauge)")
axs[2].set_xlabel("Energy (eV)")
axs[2].set_ylabel("Re χ")
axs[2].legend()
tight_layout()
savefig("floquet_vs_linear_response.png", dpi=200)
println("Saved floquet_vs_linear_response.png")
