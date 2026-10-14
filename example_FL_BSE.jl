#
# Floquet + electron-hole interaction (Floquet-BSE) for hBN, TB approximation
#
# Part 1 (linear): the Floquet-Liouville equations at first order in the field are the BSE solved by
#   inversion at each frequency. Compared with
#     - solve_bse + Build_Dielectric_Function  (TDA, resonant only)  -> identical to machine precision
#     - rt_spectrum (real time, full BSE)                             -> identical up to time-step / tmax errors
#     - Linear_response (no interaction)                             -> identical to machine precision
#
# Part 2 (non-perturbative): self-consistent Floquet steady state (screened exchange, Newton-GMRES) for a
#   drive resonant with the lowest bright exciton, as a function of E0, with dephasing eta AND population
#   relaxation gamma1 (needed: no steady state otherwise). Checked against a real-time CW run (rt_propagate
#   with the same gamma1) for one field amplitude: harmonics P_1, P_2, P_3 must agree.
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
include("BSE_finite_q.jl")      # build_bse_kernel
include("real_time.jl")         # rt_build_interaction, rt_spectrum, rt_propagate
include("FloquetBSE.jl")

# ---------------- parameters ----------------
n_k1   = 24
n_k2   = 24
dk     = 0.001                     # finite-difference step for the dipoles
r0     = 12.0                      # screening length [Angstrom]
eps_bg = 1.0
eta_ev = 0.1                       # dephasing / broadening [eV]
eta    = eta_ev / ha2ev
pol    = [1.0, 0.0]
freqs  = collect(LinRange(4.0 / ha2ev, 16.0 / ha2ev, 240))

run_rt_linear = true               # real-time full BSE for the linear comparison (a few minutes)

# non-perturbative part (smaller grid)
n_k_nl     = 12
gamma1_ev  = 0.05                  # population relaxation [eV]  (T1 = 1/gamma1)
gamma1     = gamma1_ev / ha2ev
N_floq     = 3                     # harmonics -3..3
E0_list    = collect(LinRange(2.5e-4, 5e-3, 20))   # a.u. (1 a.u. = 514 V/nm)
run_rt_cw  = true                  # real-time CW check at the largest E0

relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

lattice = set_Lattice(2, [a_1, a_2])

# =============================== Part 1: linear ===============================
k_grid = generate_unif_grid(n_k1, n_k2, lattice)
TB_sol = Solve_TB_on_grid(k_grid, Hamiltonian)
Dip_h, ∇H_w = Build_Dipole(k_grid, lattice, TB_sol, orbitals, Hamiltonian, dk)

chi_lr = Linear_response(TB_sol, Dip_h, freqs, pol, eta, lattice; antires=true)

println("\nBSE (solve_bse, TDA)...")
En, Psi = solve_bse(TB_sol, k_grid, lattice, orbitals, r0, eps_bg)
D_exc, f_osc = Build_Exciton_Dipoles(Dip_h, Psi, k_grid, lattice)
eps2_bse = Build_Dielectric_Function(D_exc, En, freqs, lattice, k_grid; eta=eta, pol_dir=pol)

kernel = build_bse_kernel(k_grid, lattice, orbitals, r0, eps_bg)
inter  = rt_build_interaction(kernel)

println("\nFloquet-BSE, linear order...")
chi_fl_ip     = floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; pol=pol)
chi_fl_tda_r  = floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; interaction=inter, pol=pol, tda=true, antires=false)
chi_fl_tda    = floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; interaction=inter, pol=pol, tda=true)
chi_fl_full   = floquet_bse_linear(TB_sol, Dip_h, lattice, freqs, eta; interaction=inter, pol=pol, tda=false)

if run_rt_linear
    println("\nReal time, full BSE (reference)...")
    chi_rt_full, _, _ = rt_spectrum(TB_sol, Dip_h, lattice, freqs; interaction=inter, pol=pol,
                                    kappa=1e-4, tmax=10.0 / eta, dt=0.5, eta=eta, tda=false, verbose=false)
end

println("\n================ Floquet-BSE (linear) checks ================")
println("IPA      : Floquet vs Linear_response          Im: ", relmax(imag.(chi_fl_ip), imag.(chi_lr)),
        "  Re: ", relmax(real.(chi_fl_ip), real.(chi_lr)))
println("TDA      : Floquet (resonant) vs solve_bse     Im: ", relmax(imag.(chi_fl_tda_r), eps2_bse))
println("TDA      : Floquet (res+antires) vs solve_bse  Im: ", relmax(imag.(chi_fl_tda), eps2_bse),
        "   (anti-resonant term only)")
if run_rt_linear
    println("full BSE : Floquet vs real time                Im: ", relmax(imag.(chi_fl_full), imag.(chi_rt_full)),
            "  Re: ", relmax(real.(chi_fl_full), real.(chi_rt_full)), "   (RT: finite dt, tmax)")
end
println("full vs TDA (resonant-antiresonant coupling)   Im: ", relmax(imag.(chi_fl_full), imag.(chi_fl_tda)))

open("floquet_bse_linear.dat", "w") do io
    println(io, "# E[eV]  eps2_BSE(solve_bse)  ImChi_FL_TDA_res  ImChi_FL_TDA  ImChi_FL_full  ImChi_FL_IPA",
            run_rt_linear ? "  ImChi_RT_full" : "")
    for iw in eachindex(freqs)
        print(io, freqs[iw] * ha2ev, " ", eps2_bse[iw], " ", imag(chi_fl_tda_r[iw]), " ", imag(chi_fl_tda[iw]), " ",
              imag(chi_fl_full[iw]), " ", imag(chi_fl_ip[iw]))
        run_rt_linear && print(io, " ", imag(chi_rt_full[iw]))
        println(io)
    end
end

# =============================== Part 2: non-perturbative ===============================
println("\n================ Floquet-BSE, non-perturbative steady state ================")
kg_nl  = generate_unif_grid(n_k_nl, n_k_nl, lattice)
TB_nl  = Solve_TB_on_grid(kg_nl, Hamiltonian)
Dip_nl, _ = Build_Dipole(kg_nl, lattice, TB_nl, orbitals, Hamiltonian, dk)
En_nl, Psi_nl = solve_bse(TB_nl, kg_nl, lattice, orbitals, r0, eps_bg)
_, fosc_nl = Build_Exciton_Dipoles(Dip_nl, Psi_nl, kg_nl, lattice)
S_bright = argmax(fosc_nl[1:min(8, length(fosc_nl))])
omega    = En_nl[S_bright]                              # resonant with the lowest bright exciton
inter_nl = rt_build_interaction(build_bse_kernel(kg_nl, lattice, orbitals, r0, eps_bg))
println("drive resonant with exciton ", S_bright, " at ", omega * ha2ev, " eV (IPA gap ",
        minimum(TB_nl.eigenval[2, :] .- TB_nl.eigenval[1, :]) * ha2ev, " eV)")

chi_lin_nl = floquet_bse_linear(TB_nl, Dip_nl, lattice, [omega], eta; interaction=inter_nl, pol=pol)[1]

nE   = length(E0_list)
nc   = zeros(nE)
Pabs = zeros(nE, N_floq)                                 # |P_1|, |P_2|, ... |P_N|
chi1 = zeros(ComplexF64, nE)
Sig  = nothing
for (ie, E0) in enumerate(E0_list)
    t0 = time()
    sol = floquet_bse_steady_state(TB_nl, Dip_nl, lattice, omega, E0; interaction=inter_nl, pol=pol,
                                   eta=eta, gamma1=gamma1, N_floq=N_floq, Sigma0=Sig, verbose=false)
    global Sig = sol.Sigma                               # continuation: start the next E0 from this one
    nc[ie]   = sol.nc
    chi1[ie] = sol.chi1
    for n in 1:N_floq
        Pabs[ie, n] = abs(sol.P[N_floq + 1 + n])
    end
    println("E0 = ", round(E0, sigdigits=3), " a.u. (", round(E0 * EAMPAU2VM * 1e-9, sigdigits=3), " V/nm)",
            "  n_c = ", round(nc[ie], sigdigits=5), "  |chi1/chi_lin| = ", round(abs(chi1[ie] / chi_lin_nl), sigdigits=5),
            "  |P2| = ", round(Pabs[ie, 2], sigdigits=4), "  |P3| = ", round(Pabs[ie, 3], sigdigits=4),
            "  Newton ", sol.newton_it, (sol.converged ? "" : " NOT CONVERGED"), "  ", round(time() - t0, digits=1), " s")
end

if run_rt_cw
    E0  = E0_list[end]
    T   = 2pi / omega
    nst = 64                                             # time steps per period
    dt  = T / nst
    nper = ceil(Int, 10.0 / gamma1 / T)                  # 10 T1: steady state reached
    tramp = 200.0
    p  = pol ./ norm(pol)
    fld = t -> (E0 * cos(omega * t) * sin(0.5pi * min(t / tramp, 1.0))^2) .* p
    println("\nReal-time CW check at E0 = ", E0, " (", nper, " periods)...")
    res = rt_propagate(TB_nl, Dip_nl; interaction=inter_nl, field=fld, tmax=nper * T, dt=dt,
                       eta=eta, gamma1=gamma1, verbose=false)
    Pt  = vec(sum(res.P .* p, dims=1))
    nlast = 20 * nst
    idx = (length(Pt) - nlast + 1):length(Pt)
    println("harmonic   |P_n| real time      |P_n| Floquet")
    for n in 1:N_floq
        Prt = abs(sum(Pt[idx] .* cis.(n * omega .* res.times[idx])) / nlast)
        println("   ", n, "        ", Prt, "    ", Pabs[end, n], "    rel. diff ", abs(Prt - Pabs[end, n]) / Pabs[end, n])
    end
end

open("floquet_bse_nonlinear.dat", "w") do io
    println(io, "# omega = ", omega * ha2ev, " eV, eta = ", eta_ev, " eV, gamma1 = ", gamma1_ev, " eV, N_floq = ", N_floq)
    println(io, "# E0[a.u.]  n_c  Re(chi1)  Im(chi1)  |P_1| ... |P_N|")
    for ie in 1:nE
        println(io, E0_list[ie], " ", nc[ie], " ", real(chi1[ie]), " ", imag(chi1[ie]), " ", join(Pabs[ie, :], " "))
    end
end

# ---------------- plot ----------------
fx = freqs .* ha2ev
fig, axs = subplots(1, 3, figsize=(15, 4.5))
axs[1].plot(fx, eps2_bse, "-", lw=4, alpha=0.4, color="tab:blue", label="BSE (solve_bse, TDA)")
axs[1].plot(fx, imag.(chi_fl_tda_r), "--", lw=1.5, color="tab:red", label="Floquet-BSE, TDA")
axs[1].plot(fx, imag.(chi_fl_full), "-", lw=1, color="k", label="Floquet-BSE, full")
run_rt_linear && axs[1].plot(fx, imag.(chi_rt_full), ":", lw=2, color="tab:orange", label="real time, full")
axs[1].plot(fx, imag.(chi_fl_ip), "-", lw=1, color="gray", label="IPA")
axs[1].set_xlabel("Energy (eV)"); axs[1].set_ylabel("ε₂"); axs[1].legend(fontsize=8)
axs[1].set_title("linear order")

axs[2].plot(E0_list, nc, "o-", label="n_c (conduction population)")
axs[2].plot(E0_list, abs.(chi1 ./ chi_lin_nl), "s-", label="|χ₁(E0)/χ_lin|")
axs[2].set_xlabel("E0 (a.u.)"); axs[2].legend(fontsize=8)
axs[2].set_title("resonant drive at " * string(round(omega * ha2ev, digits=2)) * " eV: saturation")

for n in 1:N_floq
    axs[3].loglog(E0_list, Pabs[:, n], "o-", label="|P_$n|")
end
axs[3].set_xlabel("E0 (a.u.)"); axs[3].set_ylabel("|P_n|"); axs[3].legend(fontsize=8)
axs[3].set_title("harmonics of the polarisation")
tight_layout()
savefig("floquet_bse.png", dpi=200)
println("Saved floquet_bse.png, floquet_bse_linear.dat, floquet_bse_nonlinear.dat")
