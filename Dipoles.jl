#
# Dipole matrix elements (Berry connection)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads
#
#
# * * * DIPOLES * * * #
#
# dipoles are calculated using dH/dh
#
function Build_Dipole(k_grid,lattice,TB_sol,orbitals,Hamiltonian,dk)
  #      
  println("Building Dipoles using dH/dk:")
  # 
  # a generic off-diagonal matrix example (0 1; 1 0)
  #
  s_sim=lattice.dim
  h_dim=TB_sol.h_dim
  off_diag=.~I(h_dim)
  #
  Dip_h=zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
  ∇H_w =zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
     #  
     ∇H_w[:,:,:,ik]=Grad_H(ik,k_grid,lattice,TB_sol; Hamiltonian=Hamiltonian,deltaK=dk)
     #
     for id in 1:s_dim
       Dip_h[:,:,id,ik]=WH_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
# I set to zero the diagonal part of dipoles
       Dip_h[:,:,id,ik]=Dip_h[:,:,id,ik].*off_diag
     end
     #
#    
# Now I have to divide for the energies
#
#  p = \grad_k H 
#
#  r_{ij} = i * p_{ij}/(e_j - e_i)
#
#  (diagonal terms are set to zero)
#
     for i in 1:h_dim
       for j in i+1:h_dim
          Dip_h[i,j,:,ik]= 1im*Dip_h[i,j,:,ik]/(TB_sol.eigenval[j,ik]-TB_sol.eigenval[i,ik])
          Dip_h[j,i,:,ik]=conj(Dip_h[i,j,:,ik])
       end
     end
     #
    #
  end
  #
  return Dip_h,∇H_w
  #
end
