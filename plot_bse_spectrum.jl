using PyPlot

"""
    plot_bse_spectrum(freqs_ev, eps_2_bse, eps_2_ip, exciton_energies_ev, f_osc; filename="bse_spectrum.png")

Plots the BSE dielectric function against the independent particle (IP) response,
highlighting discrete exciton energy levels and their oscillator strengths.
"""
function plot_bse_spectrum(freqs_ev::Vector{Float64}, eps_2_bse::Vector{Float64}, eps_2_ip::Vector{Float64}, exciton_energies_ev::Vector{Float64}, gap::Float64, f_osc::Vector{Float64}; filename::String="bse_spectrum.png")

    fig, ax1 = subplots(figsize=(8, 5))

    # 1. Plot Independent Particle vs BSE Dielectric Function
    ax1.plot(freqs_ev, eps_2_ip, "k--", label="IP (No Interaction)", linewidth=1.5)
    ax1.plot(freqs_ev, eps_2_bse, "r-", label="BSE (Excitonic)", linewidth=2.0)
    
    ax1.set_xlabel("Energy (eV)", fontsize=12)
    ax1.set_ylabel(L"\Im[\varepsilon(\omega)]", color="r", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.grid(true, linestyle=":", alpha=0.6)
    ax1.axvline(x=gap, color="red", linestyle="--", linewidth=1.5, label="Gap")
    
    # 2. Overlay discrete exciton oscillator strengths on a twin axis
    ax2 = ax1.twinx()
    
    # Filter excitons within the frequency window for clear plotting
    min_w, max_w = minimum(freqs_ev), maximum(freqs_ev)
    valid_mask = (exciton_energies_ev .>= min_w) .& (exciton_energies_ev .<= max_w)
    
    E_ex_plot = exciton_energies_ev[valid_mask]
    f_osc_plot = f_osc[valid_mask]
    
    if !isempty(E_ex_plot)
        markerline, stemlines, _ = ax2.stem(
            E_ex_plot, 
            f_osc_plot, 
            linefmt="b-", 
            markerfmt="bo", 
            basefmt=" "
        )
        setp(stemlines, alpha=0.5, linewidth=1.2)
        setp(markerline, markersize=4)
    end

    ax2.set_ylabel(L"Oscillator Strength $|D_S|^2$ (a.u.)", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.set_ylim(bottom=0.0)

    # Legends & Title
    ax1.legend(loc="upper right", frameon=true)
    title(L"Optical Absorption Spectrum $\varepsilon_2(\omega)$ with BSE", fontsize=13)
    
    tight_layout()
    savefig(filename, dpi=300)
    println("Saved BSE spectrum plot to: ", filename)
    close(fig)
end
