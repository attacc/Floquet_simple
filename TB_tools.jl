module TB_tools

using ProgressBars

export evaluate_DOS,fix_eigenvec_phase,ProgressBar,K_crys_to_cart,props,IO_output,dyn_props,Grad_H,Grad_U,W_gauge,H_gauge,HW_rotate,WH_rotate,RK2,RK4,rk2_step,rk4_step,generate_header,TB_lattice,TB_atomic,TB_Solution,Gauge_Correction

const W_gauge=true
const H_gauge=false

TB_lattice="lattice"
TB_atomic ="atomic"

mutable struct Properties
	eval_curr   ::Bool
	eval_pol    ::Bool
	print_dm    ::Bool
        curr_gauge  ::Bool
end

mutable struct Dynamics_Properties
	dyn_gauge    :: Bool
	damping	     :: Bool
	use_dipoles  :: Bool
	include_drho_dk :: Bool
	include_A_w	:: Bool
        use_UdU_for_dipoles :: Bool
end

mutable struct TB_Solution
        h_dim::Int
	eigenval::Array{Float64,2}
	eigenvec::Array{Complex{Float64},3}
	H_w::Array{Complex{Float64},3}
	TB_Solution() = new()
end

# 
# Possible integrators
#
RK2=1
RK4=2


dyn_props = Dynamics_Properties(true,true,false,true,true,false)

props = Properties(true, true, false, W_gauge)

mutable struct IO_Output
	dm_file  ::Union{IOStream, Nothing}
	IO_Output(dm_file=nothing) =new(dm_file)
end

IO_output=IO_Output()

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
 function evaluate_DOS(bands::Matrix{Float64},E_range::Vector{Float64}, n_points::Integer, smearing::Float64)
	dE=(E_range[2]-E_range[1])/n_points
        DOS=zeros(Float64,n_points,2)
	for i in 1:n_points
		e_dos=E_range[1]+dE*(i-1)
		DOS[i,1]=e_dos
		for eb in bands
#			DOS[i,2]=DOS[i,2]+lorentzian(e_dos,eb,smearing)
			DOS[i,2]=DOS[i,2]+gaussian(e_dos,eb,smearing)
		end
	end
	return DOS
 end
 #
end

@inline function HW_rotate(M,eigenvec)
  return eigenvec*M*adjoint(eigenvec)
end 

@inline function WH_rotate(M,eigenvec)
  return adjoint(eigenvec)*M*eigenvec
end 

function init_output()
	if props.print_dm    
      		println("Write density matrix on disk for k=1")
		IO_output.dm_file  =open("density_matrix.txt","w")	
	end
end

function print_density_matrix(time,rho_i)
        ik=1
	if dyn_props.h_gauge
		rho=rho_i[:,:,2]
	else
		rho=HW_rotate(rho_i[:,:,2],TB_sol.eigenvec[:,:,2],"W_to_H")
	end
	write(IO_output.dm_file," $(time) ")
	write(IO_output.dm_file," $(real(rho[1,1])) $(imag(rho[1,1])) ")
	write(IO_output.dm_file," $(real(rho[1,2])) $(imag(rho[1,2])) ")
	write(IO_output.dm_file," $(real(rho[2,1])) $(imag(rho[2,1])) ")
	write(IO_output.dm_file," $(real(rho[2,2])) $(imag(rho[2,2])) \n")
end

function finalize_output()
	if IO_output.dm_file != nothing  
		close(IO_output.dm_file)	
	end
end 

function rk2_step!(rho_in, d_rho, time, it)
  h = time[it+1] - time[it]
  rho_in.=rho_in + h * d_rho(rho_in + d_rho(rho_in, time[it]) * h/2, time[it] + h/2)
  return nothing
end

function rk4_step!(rho_in, d_rho, time, it)
  h = time[it+1] - time[it]
  rho1 = d_rho(rho_in, time[it])
  rho2 = d_rho(rho_in+rho1*h/2.0, time[it]+h/2.0)
  rho3 = d_rho(rho_in+rho2*h/2.0, time[it]+h/2.0)
  rho4 = d_rho(rho_in+rho3*h, time[it]+h)
  rho_in .= rho_in + h*(rho1/6.0+rho2/3.0+rho3/3.0+rho4/6.0)
  return nothing
end
#
# Fix phase of the eigenvectors in such a way
# to have a positive definite diagonal
#
function fix_eigenvec_phase(eigenvec,ik=nothing,k_grid=nothing)
#	println("Before phase fixing : ")
#	show(stdout,"text/plain",eigenvec)
	#
	# Rotation phase matrix
	#
	phase_m=zeros(Complex{Float64},2)
	phase_m[1]=exp(-1im*angle(eigenvec[1,1]))
        phase_m[2]=exp(-1im*angle(eigenvec[2,2]))
	#
	# New eigenvectors
	#
#        if ik!=nothing
#          k_xy=k_grid.ik_map_inv[:,ik]
#          dk=zeros(Float64,2)
#          dk[1]=2.0*pi*(k_xy[1]-1.0)/k_grid.nk_dir[1]
#          dk[2]=2.0*pi*(k_xy[2]-1.0)/k_grid.nk_dir[2]
#          phase_m[1]*=exp(-1im*(dk[1]+dk[2]))
#          phase_m[2]*=exp(-1im*(dk[1]+dk[2]))
#        end
        #
	eigenvec[:,1]=eigenvec[:,1]*phase_m[1]
	eigenvec[:,2]=eigenvec[:,2]*phase_m[2]
#	println("\nAfter phase fixed : ")
#	show(stdout,"text/plain",eigenvec)
	#
	# Norm
	#
#	println(norm(eigenvec[:,1]))
#	println(norm(eigenvec[:,2]))
	#
	return eigenvec
end
#
# For the derivative of the Hamiltonian
# I recalcualte it because H(k+G)/=H(k)

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

function Solve_TB_on_grid(k_grid,Hamiltonian)
  # 
  TB_sol=TB_Solution()
  #
  TB_sol.h_dim=2
  TB_sol.eigenval=zeros(Float64,h_dim,k_grid.nk)
  TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
  TB_sol.H_w     =zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
  #
  println(" K-point list ")
  println(" nk = ",k_grid.nk)
  #
  #print_k_grid(k_grid, lattice)
  #
  println("Delta-k for derivatives : $dk ")

  println("Building Hamiltonian: ")

  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
    TB_sol.H_w[:,:,ik]=Hamiltonian(k_grid.kpt[:,ik])
    data= eigen(TB_sol.H_w[:,:,ik])      # Diagonalize the matrix
    TB_sol.eigenval[:,ik]   = data.values
    TB_sol.eigenvec[:,:,ik] = data.vectors
    TB_sol.eigenvec[:,:,ik] = fix_eigenvec_phase(TB_sol.eigenvec[:,:,ik])
  end
  #
  #Print Hamiltonian info
  #
  dir_gap=minimum(TB_sol.eigenval[2,:]-TB_sol.eigenval[1,:])
  println("Direct gap : ",dir_gap*ha2ev," [eV] ")
  ind_gap=minimum(TB_sol.eigenval[2,:])-maximum(TB_sol.eigenval[1,:])
  println("Indirect gap : ",ind_gap*ha2ev," [eV] ")
  #
  return TB_sol
  #
end
