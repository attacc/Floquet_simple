module TB_tools

using ProgressBars

export ProgressBar,Grad_H,HW_rotate,WH_rotate,TB_Solution

mutable struct TB_Solution
        h_dim::Int
	eigenval::Array{Float64,2}
	eigenvec::Array{Complex{Float64},3}
	H_w::Array{Complex{Float64},3}
	TB_Solution() = new()
end

# 
 function fermi_function(E, E_f, Temp)
   fermi_function=1.0/((exp(E-E_f)/Temp))
   return fermi_function
 end
 #
 function lorentzian(x,x_0, Gamma)
	 return 1.0/pi*(0.5*Gamma)/((x-x_0)^2+(0.5*Gamma)^2)
 end
 #
 function gaussian(x,x_0, Sigma)
	 return 1.0/(Sigma*sqrt(2.0*pi))*exp(-(x-x_0)^2/(2*Sigma^2))
 end
 #
end

@inline function HW_rotate(M,eigenvec)
  return eigenvec*M*adjoint(eigenvec)
end 

@inline function WH_rotate(M,eigenvec)
  return adjoint(eigenvec)*M*eigenvec
end 

function Grad_H(ik, k_grid, lattice, TB_sol; Hamiltonian=nothing, deltaK=nothing)
    #
    # calculate dH/dk in the Wannier Gauge
    # derivatives are in cartesian coordinates
    #
    h_dim=TB_sol.h_dim    # hamiltonian dimension
    s_dim=lattice.dim     # space dimension
    #
    dH_w       =zeros(Complex{Float64},h_dim,h_dim,s_dim)
    #
    # Derivative by finite differences, 
    # recalculating the Hamiltonian
    #
    for id in 1:s_dim
      #
      #  
      k_plus =copy(k_grid.kpt[:,ik])
      k_minus=copy(k_grid.kpt[:,ik])
      #
      if deltaK==nothing
        vec_dk=lattice.rvectors[id]/k_grid.nk_dir[id]
      else
        vec_dk=zeros(Float64,s_dim)
        vec_dk[id]=deltaK
      end
      #
      dk=norm(vec_dk)
      #
      k_plus =k_plus +vec_dk
      k_minus=k_minus-vec_dk
      #
      H_plus =Hamiltonian(k_plus)
      H_minus=Hamiltonian(k_minus)
      #
      dH_w[:,:,id]=(H_plus-H_minus)/(2.0*dk)
      #
    end
    #
    return dH_w
    #
end

"""
    Solve_TB_on_grid(k_grid, Hamiltonian; nv=nothing)

Diagonalises `Hamiltonian(k)` on all the k-points of `k_grid`, for any matrix dimension: `h_dim` is taken
from the size of the Hamiltonian at the first k-point. Bands are sorted in ascending energy.
`nv` (number of valence bands) is only used to print the gaps (default `h_dim ÷ 2`).
"""
function Solve_TB_on_grid(k_grid,Hamiltonian; nv=nothing)
  #
  h_dim = size(Hamiltonian(k_grid.kpt[:,1]), 1)     # dimension of the TB Hamiltonian
  nv_gap = nv === nothing ? max(1, h_dim ÷ 2) : nv
  #
  TB_sol=TB_Solution()
  #
  TB_sol.h_dim=h_dim
  TB_sol.eigenval=zeros(Float64,h_dim,k_grid.nk)
  TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
  TB_sol.H_w     =zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
  #
  println(" K-point list ")
  println(" nk = ",k_grid.nk)
  println(" Hamiltonian dimension = ",h_dim)
  #
  #print_k_grid(k_grid, lattice)
  #

  println("Building Hamiltonian: ")

  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
    TB_sol.H_w[:,:,ik]=Hamiltonian(k_grid.kpt[:,ik])
    data= eigen(Hermitian(TB_sol.H_w[:,:,ik]))      # Diagonalize the matrix (ascending eigenvalues)
    TB_sol.eigenval[:,ik]   = data.values
    TB_sol.eigenvec[:,:,ik] = data.vectors
  end
  #
  #Print Hamiltonian info (gap between band nv and nv+1)
  #
  if 1 <= nv_gap < h_dim
    dir_gap=minimum(TB_sol.eigenval[nv_gap+1,:]-TB_sol.eigenval[nv_gap,:])
    println("Direct gap : ",dir_gap*ha2ev," [eV] ")
    ind_gap=minimum(TB_sol.eigenval[nv_gap+1,:])-maximum(TB_sol.eigenval[nv_gap,:])
    println("Indirect gap : ",ind_gap*ha2ev," [eV] ")
  end
  #
  return TB_sol
  #
end
