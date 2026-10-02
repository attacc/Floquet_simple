#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads
#
# Function that calculate the linear respone
#
# The result is normalised exactly as the BSE dielectric function (Build_Dielectric_Function):
#
#   chi(w) = (16 pi / Omega_BZ) * (1/Nk) sum_k sum_{v,c} |E.D_vc(k)|^2 / (E_c-E_v-w-i eta)
#
# so that Im chi(w) coincides with the BSE eps2(w) when the electron-hole interaction is switched off
# (BSE: eps2 = (16 pi^2/Omega_BZ) <|E.D|^2 delta_eta>_k ,  Im[(1/Nk) sum |E.D|^2/(Delta-w-i eta)] = pi <|E.D|^2 delta_eta>_k).
# Omega_BZ is the Brillouin-zone area (1/Bohr^2), taken from the reciprocal vectors of `lattice`.
# If the prefactor of Build_Dielectric_Function is changed, change `prefactor` below as well.
#
function Linear_response(TB_sol, Dip_h, freqs, E_field_ver, η, lattice, nv=1; antires=false)
   h_dim=TB_sol.h_dim
   nk=size(TB_sol.eigenval,2)
   xhi=zeros(Complex{Float64},length(freqs))
   Res2=zeros(Complex{Float64},h_dim,h_dim,nk)
   #
   # normalisation compatible with the BSE spectrum
   #
   b1=lattice.rvectors[1]
   b2=lattice.rvectors[2]
   BZ_area=abs(b1[1]*b2[2]-b1[2]*b2[1])
   prefactor=16.0*pi/BZ_area

   println("Residuals: ")
   Threads.@threads for ik in ProgressBar(1:nk)
     for iv in 1:nv,ic in nv+1:h_dim
        Res2[iv,ic,ik]=abs2(sum(Dip_h[iv,ic,:,ik].*E_field_ver[:]))
     end
   end
   print("Xhi: ")
   Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:nk,iv in 1:nv,ic in nv+1:h_dim
         e_v=TB_sol.eigenval[iv,ik]
         e_c=TB_sol.eigenval[ic,ik]
         xhi[ifreq]+=Res2[iv,ic,ik]/(e_c-e_v-freqs[ifreq]-η*1im)
     end
   end
    
   if antires
     print("Xhi antirex: ")
     Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:nk,iv in 1:nv,ic in nv+1:h_dim
         e_v=TB_sol.eigenval[iv,ik]
         e_c=TB_sol.eigenval[ic,ik]
         xhi[ifreq]-=Res2[iv,ic,ik]/(e_c-e_v+freqs[ifreq]-η*1im)
     end
     end
   end
   xhi.=prefactor.*xhi./nk
   return xhi
end
