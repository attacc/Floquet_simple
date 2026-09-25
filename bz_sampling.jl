module BZ_sampling

using Printf

export generate_circuit,generate_unif_grid,get_k_neighbor,print_k_grid

mutable struct K_Grid
	kpt::Array{Float64,2}
	nk_dir::Array{Int,1}
        nk::Int
	ik_map::Array{Int,3}
	ik_map_inv::Array{Int,2}
end

function generate_circuit(points, n_steps)
	println("Generate k-path ")
	n_points=length(points)
	@printf("number of points:%5i \n",n_points)
	if n_points <= 1
		error("number of points less or equal to 1 ")
	end
	for i in 1:n_points
		@printf("v(%d) = %s \n",i,points[i])
	end
	path=Any[]
	for i in 1:(n_points-1)
		for j in 0:(n_steps-1)	
			dp=(points[i+1]-points[i])/n_steps
			push!(path,points[i]+dp*j)
		end
	end
	push!(path,points[n_points])
	return path
 end
 #
 function generate_unif_grid(n_kx, n_ky, lattice)
     scale=1.0
     nk   =n_kx*n_ky
     kpt  =zeros(Float64,lattice.dim,nk)
     ik_grid    =zeros(Int,n_kx,n_ky,1)
     ik_grid_inv=zeros(Int,lattice.dim,nk)

     vec_crystal=zeros(Float64,lattice.dim)
     dx=scale/n_kx
     dy=scale/n_ky

     ik=1
     for ix in 0:(n_kx-1),iy in 0:(n_ky-1)
           ik_grid[ix+1,iy+1,1]=ik
           ik_grid_inv[:,ik]   =[ix+1,iy+1]
	   vec_crystal[1]=dx*ix
	   vec_crystal[2]=dy*iy
	   kpt[:,ik]=lattice.rvectors[1]*vec_crystal[1]+lattice.rvectors[2]*vec_crystal[2]
	   ik=ik+1
     end

     k_grid = K_Grid(
		kpt,
		[n_kx,n_ky,1],
                nk,
		ik_grid,
		ik_grid_inv
	       )
     return k_grid
 end
 #
 function print_k_grid(k_grid, lattice)
    println("K-points grid ")
    println("grid dimensions : ",k_grid.nk_dir, "\n\n")
    for ik in 1:size(k_grid.kpt, 2)
       k_crystal=lattice.b_mat_inv*k_grid.kpt[:,ik]
       println("ik ",ik," kpt ",k_crystal," ik_grid ",k_grid.ik_map_inv[:,ik])
       ik_left =get_k_neighbor(ik,1,1,k_grid)
       ik_right=get_k_neighbor(ik,1,-1,k_grid)
       println("x-neighboar ",ik_left," - ",ik_right)
    end
 end
 #
 function get_k_neighbor(ik,id,istep,k_grid)
     s_dim=2

     # position of k-point in the integer grid
     ik_n  =k_grid.ik_map_inv[:,ik]

     # shift the k-points by an integer step
     ik_n[id] =ik_n[id]+istep 
#
     d_size=k_grid.nk_dir[id]
     if ik_n[id] > d_size
        ik_n[id]=ik_n[id]-d_size
     elseif ik_n[id]<=0
        ik_n[id]=ik_n[id]+d_size
     end
#
     ik_indx=[1,1,1]
     for id in 1:s_dim
        ik_indx[id]=ik_n[id]
     end
     ik_n=k_grid.ik_map[ik_indx[1],ik_indx[2],ik_indx[3]]
     return ik_n
 end
 #
end
