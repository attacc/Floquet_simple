#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads
#
#
# Function that calculate the linear respone
#
function Linear_response(TB_sol, Dip_h, freqs,E_field_ver, η, nv=1)
   h_dim=TB_sol.h_dim
   xhi=zeros(Complex{Float64},length(freqs))
   Res=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

   println("Residuals: ")
   Threads.@threads for ik in ProgressBar(1:k_grid.nk)
     for iv in 1:nv,ic in nv+1:h_dim
        Res[iv,ic,ik]=sum(Dip_h[iv,ic,:,ik].*E_field_ver[:])
     end
   end
   print("Xhi: ")
   Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:k_grid.nk,iv in 1:nv,ic in nv+1:h_dim
         e_v=TB_sol.eigenval[iv,ik]
         e_c=TB_sol.eigenval[ic,ik]
         xhi[ifreq]=xhi[ifreq]+abs(Res[iv,ic,ik])^2/(e_c-e_v-freqs[ifreq]-η*1im)
     end
   end
   xhi.=xhi/k_grid.nk
   return xhi
end
