#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#
module hBN2D

using LinearAlgebra

include("lattice.jl")
using .LatticeTools

include("TB_tools.jl")
using .TB_tools

include("units.jl")
using .Units

#
# In this program the lattice constant is equal to 1
#
# Default TB paramters from PRB 94, 125303 
#t_0 = 2.92/ha2ev  # eV
#E_gap=2.81*2.0/ha2ev  # eV
#
# Parameters from Ducastelle, Paleari etc...
t_0  =2.92/ha2ev
E_gap=3.625*2.0/ha2ev
#
s_dim=2 # space dimension
h_dim=2 # hamiltonian dimension
#
# Distance between neighbor 
a_cc=2.504 # a.u.

# Atom positions
d_1=      [0.0,0.0]
d_2=-a_cc*[1.0,0.0]

# Lattice constant a = sqrt(3) * a_cc
a_lat = sqrt(3.0) * a_cc

# Standard 2D hexagonal lattice vectors (a_lat along x/y)

a_1 = a_cc / 2.0 * [3.0,  sqrt(3.0)]
a_2 = a_cc / 2.0 * [3.0, -sqrt(3.0)]

nn=zeros(Complex{Float64},3,2)
nn[1,:] = a_cc / 2.0 * [1.0,  sqrt(3.0)]
nn[2,:] = a_cc / 2.0 * [2.0, -sqrt(3.0)]
nn[3,:] = -a_cc * [1.0, 0.0]

orbitals=set_Orbitals(2,[d_1,d_2])

export Hamiltonian,Berry_Connection,a_1,a_2,s_dim,h_dim,a_cc,orbitals
  #
  global ndim=2
  #
  global off_diag=.~I(h_dim)
  #
  function Hamiltonian(k)::Matrix{Complex{Float64}}
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]= E_gap/2.0
	H[2,2]=-E_gap/2.0
        #
        # Off diagonal part
        # f(k)=e^{-i*k_y*a} * (1+2*e^{ i*k_y*3*a/2} ) * cos(sqrt(3)*a/2*k_x)
        # f_k=exp(-1im*k[1]*a_cc)*(1.0+2.0*exp(1im*k[1]*3.0*a_cc/2.0)*cos(sqrt(3.0)*k[2]*a_cc/2.0))
        #
        f_k=0.0
        for inn in 1:3
            f_k+=exp(1im*dot(k[:],nn[inn,:]))
        end

	H[1,2]=-t_0*f_k
        #
	H[2,1]=conj(H[1,2])
        #
	return H
   end
   #
end
