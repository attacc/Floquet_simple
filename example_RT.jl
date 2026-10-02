#
# Real-time propagation (IPA and BSE) and spectra from the Fourier transform of the polarisation,
# compared with Linear_response (IPA) and with the BSE spectrum of solve_bse (TDA).
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
include("BSE.jl")
include("BSE_dipoles.jl")
include("BSE_finite_q.jl")      # build_bse_kernel (interaction kernel of the real-time BSE)
include("real_time.jl")

# ---------------- parameters ----------------
n_k1 = 24
n_k2 = 24
dk   = 0.001                       # finite-difference step for the dipoles

r0     = 12.0                      # screening length [Angstrom]
eps_bg = 1.0
eta_ev = 0.1                       # dephasing / broadening [eV]
eta    = eta_ev / ha2ev
pol    = [1.0, 0.0]                # polarisation of the kick and of the response

kappa  = 1e-4                      # delta-kick strength (a.u.), linear regime: kappa*|D| << 1
dt     = 0.5                       # RK4 time step (a.u.); must resolve the largest transition (~19 eV)
tmax   = 10.0 / eta                # a.u.: the coherences decay as exp(-eta t)

# The window must contain the bound excitons: with r0 = 12 A the lowest ones are at ~5.8 eV (about half of
# the oscillator strength), below the IPA gap of 7.25 eV.
freqs_nsteps = 240
freqs = collect(LinRange(4.0 / ha2ev, 16.0 / ha2ev, freqs_nsteps))   # Hartree

run_pulse_test = true              # IPA with a Gaussian pulse instead of a kick (checks the `field` option)

# ---------------- TB, dipoles ----------------
lattice = set_Lattice(2, [a_1, a_2])
k_grid  = generate_unif_grid(n_k1, n_k2, lattice)
TB_sol  = Solve_TB_on_grid(k_grid, Hamiltonian)
Dip_h, ∇H_w = Build_Dipole(k_grid, lattice, TB_sol, orbitals, Hamiltonian, dk)

relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# ---------------- IPA: linear response and exact formula ----------------
chi_lr = Linear_response(TB_sol, Dip_h, freqs, pol, eta, lattice; antires=true)   # resonant + anti-resonant

# causal (retarded) form, 1/(Delta-w-i eta) + 1/(Delta+w+i eta): what a real-time calculation gives.
# (Linear_response with antires=true uses 1/(Delta+w-i eta): same Re chi, opposite sign of the tiny
#  anti-resonant contribution to Im chi.)
function chi_ip_exact(TB_sol, Dip_h, freqs, pol, eta, lattice)
    b1 = lattice.rvectors[1]; b2 = lattice.rvectors[2]
    pref = 16.0 * pi / abs(b1[1] * b2[2] - b1[2] * b2[1])
    nk = size(TB_sol.eigenval, 2)
    chi = zeros(ComplexF64, length(freqs))
    for (iw, w) in enumerate(freqs), ik in 1:nk
        res2  = abs2(sum(Dip_h[1, 2, :, ik] .* pol))
        Delta = TB_sol.eigenval[2, ik] - TB_sol.eigenval[1, ik]
        chi[iw] += res2 * (1 / (Delta - w - 1im * eta) + 1 / (Delta + w + 1im * eta))
    end
    return pref .* chi ./ nk
end
chi_ex = chi_ip_exact(TB_sol, Dip_h, freqs, pol, eta, lattice)

# ---------------- real time, IPA ----------------
println("\nReal time, IPA...")
chi_rt_ip, t_ip, P_ip = rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=nothing, pol=pol,
                                    kappa=kappa, tmax=tmax, dt=dt, eta=eta)

# ---------------- BSE reference (solve_bse, TDA) ----------------
println("\nBSE (solve_bse)...")
En, Psi = solve_bse(TB_sol, k_grid, lattice, orbitals, r0, eps_bg)
D_exc, f_osc = Build_Exciton_Dipoles(Dip_h, Psi, k_grid, lattice)
eps2_bse = Build_Dielectric_Function(D_exc, En, freqs, lattice, k_grid; eta=eta, pol_dir=pol)

# ---------------- real time, BSE ----------------
kernel = build_bse_kernel(k_grid, lattice, orbitals, r0, eps_bg)
inter  = rt_build_interaction(kernel)

println("\nReal time, BSE (TDA)...")
chi_rt_tda, _, _ = rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=inter, pol=pol,
                               kappa=kappa, tmax=tmax, dt=dt, eta=eta, tda=true)

println("\nReal time, BSE (full, resonant-antiresonant coupling included)...")
chi_rt_full, _, _ = rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=inter, pol=pol,
                                kappa=kappa, tmax=tmax, dt=dt, eta=eta, tda=false)

# ---------------- optional: Gaussian pulse (IPA) ----------------
if run_pulse_test
    println("\nReal time, IPA with a Gaussian pulse...")
    sigma = 3.0
    t0    = 8.0 * sigma
    Evec  = 1e-4 .* pol
    res   = rt_propagate(TB_sol, Dip_h; field=gaussian_pulse(Evec, t0, sigma), tmax=tmax, dt=dt,
                         eta=eta, verbose=false)
    chi_pulse = rt_susceptibility(res.times, res.P[1, :], freqs, lattice; Efield=res.Efield[1, :])
    println("IPA, pulse:  Im chi vs exact  : ", relmax(imag.(chi_pulse), imag.(chi_ex)))
end

# ---------------- comparison ----------------
println("\n================ real time vs frequency domain ================")
println("IPA : RT vs exact (res+antires)          Im: ", relmax(imag.(chi_rt_ip), imag.(chi_ex)),
        "   Re: ", relmax(real.(chi_rt_ip), real.(chi_ex)))
println("IPA : RT vs Linear_response(antires=true) Im: ", relmax(imag.(chi_rt_ip), imag.(chi_lr)),
        "   Re: ", relmax(real.(chi_rt_ip), real.(chi_lr)))
println("BSE : RT (TDA) vs solve_bse               Im: ", relmax(imag.(chi_rt_tda), eps2_bse))
println("BSE : RT (full) vs solve_bse (TDA)        Im: ", relmax(imag.(chi_rt_full), eps2_bse),
        "   (difference = resonant-antiresonant coupling)")
println("BSE : RT (TDA) vs IPA                     Im: ", relmax(imag.(chi_rt_tda), imag.(chi_rt_ip)),
        "   (size of the electron-hole effect)")
println("lowest excitons [eV]: ", En[1:4] .* ha2ev)
for (name, y) in (("IPA (RT)", imag.(chi_rt_ip)), ("BSE solve_bse", eps2_bse), ("BSE RT TDA", imag.(chi_rt_tda)),
                   ("BSE RT full", imag.(chi_rt_full)))
    println(rpad(name, 14), ": max = ", round(maximum(y), digits=2), " at ", round(freqs[argmax(y)] * ha2ev, digits=3), " eV")
end
println("(normalisation: Im chi = eps2 of Build_Dielectric_Function; differences are relative to the maximum)")

# ---------------- save ----------------
open("rt_spectra.dat", "w") do io
    println(io, "# E[eV]  ImChi_LinearResponse  ImChi_RT_IPA  eps2_BSE(solve_bse)  ImChi_RT_BSE_TDA  ImChi_RT_BSE_full")
    for iw in eachindex(freqs)
        println(io, freqs[iw] * ha2ev, " ", imag(chi_lr[iw]), " ", imag(chi_rt_ip[iw]), " ",
                eps2_bse[iw], " ", imag(chi_rt_tda[iw]), " ", imag(chi_rt_full[iw]))
    end
end

# ---------------- plot ----------------
fx = freqs .* ha2ev
fig, axs = subplots(2, 1, figsize=(7, 8), sharex=true)
axs[1].plot(fx, imag.(chi_lr), "-", lw=3, alpha=0.5, label="Linear_response (antires)")
axs[1].plot(fx, imag.(chi_rt_ip), "k--", lw=1.2, label="real time (IPA)")
axs[1].set_ylabel("Im χ  (IPA)")
axs[1].legend()

# BSE: reference spectrum as a thick line underneath, the real-time curves on top
axs[2].plot(fx, eps2_bse, "-", lw=4, alpha=0.4, color="tab:blue", label="BSE (solve_bse, TDA)")
axs[2].plot(fx, imag.(chi_rt_tda), "--", lw=1.5, color="tab:red", label="real time BSE, TDA")
axs[2].plot(fx, imag.(chi_rt_full), ":", lw=2, color="k", label="real time BSE, full")
axs[2].plot(fx, imag.(chi_rt_ip), "-", lw=1, color="gray", label="IPA")
for E in En[1:4]
    axs[2].axvline(E * ha2ev, color="tab:blue", lw=0.5, alpha=0.5)    # lowest exciton energies
end
axs[2].set_xlabel("Energy (eV)")
axs[2].set_ylabel("ε₂  (BSE)")
axs[2].legend()
tight_layout()
savefig("rt_spectra.png", dpi=200)
println("Saved rt_spectra.png, rt_spectra.dat")
