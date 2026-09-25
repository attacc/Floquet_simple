using FFTW
using .RealTimesHF

# 1. Compute dH/dk list using Grad_H from your TB_tools.jl
dH_dk_list = [Grad_H(ik, k_grid, lattice, tb_sol; Hamiltonian=Hamiltonian) for ik in 1:k_grid.nk]

# 2. Simulation parameters
dt = 0.05               # Time step (fs or atomic units; match your Hamiltonian units)
Nt = 4000               # Propagation steps
r0 = 33.5               # hBN screening length (Angstroms)
eps_bg = 1.0            # Vacuum background
E0_pulse = [0.001, 0.0] # Weak x-polarized delta pulse

# 3. Propagate real-time dynamics
time_axis, P_t = propagate_real_time(tb_sol, k_grid, lattice, dH_dk_list, r0, eps_bg, dt, Nt, E0_pulse)

# 4. Extract absorption spectrum via Fourier Transform of P(t) with Lorentzian broadening eta
eta = 0.05 # Dephasing broadening (eV)
damping = exp.(-eta .* time_axis)
P_x_damped = P_t[1, :] .* damping

freq_axis = fftfreq(Nt, 2*pi/dt)
P_omega = fft(P_x_damped)

# eps_2(omega) is proportional to Im[P(omega) / E0]
eps_2 = imag.(P_omega ./ E0_pulse[1])
