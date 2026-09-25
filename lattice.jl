module LatticeTools

using LinearAlgebra

export Lattice,set_Lattice,generate_R_grid,set_Orbitals,wrap_k_to_BZ

mutable struct Lattice
    dim::Int8
    vectors::Array{Array{Float64,1},1}
    rvectors::Array{Array{Float64,1},1} #reciplocal lattice vectors
    rv_norm::Array{Float64,1}  # norm of the reciprocal lattice vectors
    vol::Float64
    r_vol::Float64
    b_mat_inv::Array{Float64,2}
end

mutable struct R_grid
    R_vec::Array{Float64,2}
    nR_dir::Array{Int,1}
    nR::Int
end

mutable struct Orbitals_tau
    nOrb::Int
    tau::Array{Array{Float64,1}}
end


"""
    set_Lattice(dim::Integer,vectors::Array{Array{Float64,1},1})
Initialize lattice.
We have to call this before making Hamiltonian.
dim: Dimension of the system.
vector:: Primitive vectors.

Example:

1D system

```julia
la1 = set_Lattice(1,[[1.0]])
```

2D system

```julia
a1 = [sqrt(3)/2,1/2]
a2 = [0,1]
la2 = set_Lattice(2,[a1,a2])
```

3D system

```julia
a1 = [1,0,0]
a2 = [0,1,0]
a3 = [0,0,1]
la2 = set_Lattice(3,[a1,a2,a3])
```
"""
function set_Lattice(dim, vectors)
    #making primitive vectors
    pvector_1 = zeros(Float64, 3)
    pvector_2 = zeros(Float64, 3)
    pvector_3 = zeros(Float64, 3)
    if dim == 1
        pvector_1[1] = vectors[1][1]
        pvector_2[2] = 1.0 # 0 1 0
        pvector_3[3] = 1.0 # 0 0 1
    elseif dim == 2
        pvector_1[1:2] = vectors[1][1:2]
        pvector_2[1:2] = vectors[2][1:2]
        pvector_3[3] = 1.0 # 0 0 1
    elseif dim == 3
        pvector_1[1:3] = vectors[1][1:3]
        pvector_2[1:3] = vectors[2][1:3]
        pvector_3[1:3] = vectors[3][1:3]
    end
    #making reciplocal lattice vectors
    vol = dot(pvector_1, cross(pvector_2, pvector_3))
    rvector_1 = 2π * cross(pvector_2, pvector_3) / vol
    rvector_2 = 2π * cross(pvector_3, pvector_1) / vol
    rvector_3 = 2π * cross(pvector_1, pvector_2) / vol

    if vol< 0.0
       println("Axis vectors are left handed")
       vol=abs(vol)
    end
    
    rv_norm=zeros(Float64,dim)

    if dim == 1
        rvectors  = [[rvector_1[1]]]
	rv_norm[1] =norm(rvector_1[1])
    elseif dim == 2
        rvectors = [rvector_1[1:2], rvector_2[1:2]]
	rv_norm[1] =norm(rvector_1[1:2])
	rv_norm[2] =norm(rvector_1[1:2])

    elseif dim == 3
        rvectors = [rvector_1[1:3], rvector_2[1:3], rvector_3[1:3]]
	rv_norm[1] =norm(rvector_1[1:3])
	rv_norm[2] =norm(rvector_1[1:3])
	rv_norm[3] =norm(rvector_1[1:3])
    end

    r_vol=(2π)^3/vol

    println("Lattice vectors : ")
    for id in 1:dim
	    println(vectors[id][1:dim])
    end
    println("Reciprocal lattice vectors : ")
    for id in 1:dim
       println(rvectors[id][1:dim])
    end

    b_mat_inv=zeros(Float64,dim,dim)
    for id in 1:dim
        b_mat_inv[:,id]=rvectors[id][1:dim]
    end
    b_mat_inv=inv(b_mat_inv)

    println("Direct lattice volume     : ",vol, " [a.u.]")
    println("Reciprocal lattice volume : ",r_vol, " [a.u.]")

    lattice = Lattice(
        dim,
	vectors,
	rvectors,
	rv_norm,
	vol,
	r_vol,
        b_mat_inv
    )
    return lattice
end


"""
    wrap_k_to_BZ(dk_cart::Vector{Float64}, lattice::Lattice)

Applies the minimum-image convention to a momentum vector or k-difference `dk_cart`.
Converts to reduced fractional coordinates, shifts into the first Brillouin Zone [-0.5, 0.5],
and maps back to Cartesian space.

# Returns
- `q_bz`: Minimum-image vector in Cartesian space
- `q_norm`: Magnitude ||q_bz||
"""
function wrap_k_to_BZ(dk_cart::Vector{Float64}, lattice::Lattice)
    dim = Int(lattice.dim)
    # 1. Convert Cartesian to fractional reciprocal coordinates directly
    dk_frac = lattice.b_mat_inv * dk_cart[1:dim]

    # 2. Wrap into [-0.5, 0.5]
    dk_frac_bz = dk_frac .- round.(dk_frac)

    # 3. Convert back to Cartesian space
    q_bz = zeros(Float64, dim)
    for id in 1:dim
        q_bz .+= dk_frac_bz[id] .* lattice.rvectors[id][1:dim]
    end
    return q_bz
end


function set_Orbitals(nOrb, tau_vectors)
    orbitals_tau=Orbitals_tau(
      nOrb,
      tau_vectors)
    return orbitals_tau
end

end

