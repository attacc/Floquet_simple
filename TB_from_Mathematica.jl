using LinearAlgebra
using SpecialFunctions  # For Bessel functions
using PyPlot
# using PyCall
using ProgressBars


# Constants (assuming these are defined elsewhere)
Nm = 80  # Example value, replace with actual value
Q = 0.0  # Example value, replace with actual value
W = 0.25  # Example value, replace with actual value
F = 10.0  # Example value, replace with actual value
Nk =20 # Example value, replace with actual value
h_dim=2 # TB-Hamiltonian dimension

kpt_with_border=true #include or not k-points at the border of the BZ

max_N0=20
if isodd(max_N0)
    print("Max N0 should be even!!! ")
    exit(0)
end

# Pauli matrices
σ0 = [1 0; 0 1]
σ1 = [0 1; 1 0]
σ3 = [1 0; 0 -1]

# Define the Hamiltonian matrix H[k]
# Notice that this matrix is not Hermitian
# I found that if you force it to be Hermitian results are the same
# (see the commented lines)
#
function H_FLQ(k)
    H_matrix = zeros(ComplexF64, (h_dim*(2*Nm+1), h_dim*(2*Nm+1)))
    for n in -Nm:Nm
    for m in -Nm:Nm
#      for m in n:Nm
       if n == m
         H_matrix[h_dim*(n+Nm)+1:h_dim*(n+Nm)+h_dim, h_dim*(m+Nm)+1:h_dim*(m+Nm)+h_dim] = (n * W * σ0 + Q * σ3)
       end
       H_matrix[h_dim*(n+Nm)+1:h_dim*(n+Nm)+h_dim, h_dim*(m+Nm)+1:h_dim*(m+Nm)+h_dim] += (1.0im)^(m-n) * besselj(m-n, F) * cos(k - (m-n)*π/2) * σ1
#      H_matrix[2*(m+Nm)+1:2*(m+Nm)+2, 2*(n+Nm)+1:2*(n+Nm)+2] = conj(H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2])
    end
    end
    return H_matrix
end

function H_tb(k)
  H_tb = zeros(ComplexF64, (h_dim, h_dim))
  H_tb = Q * σ3
  H_tb+= cos(k)*σ1
  return H_tb
end

# Define a0[k]
a0(k) = -1/sqrt(2) * sqrt(1 - Q / sqrt(Q^2 + cos(k)^2))

# Define b0[k]
b0(k) = 1/sqrt(2) * sqrt(1 + Q / sqrt(Q^2 + cos(k)^2))

# Define E0[k]
E0(k) = -sqrt(Q^2 + cos(k)^2)

# Define A[k] (eigenvector corresponding to the (2*Nm+1)-th eigenvalue)
function A(k)
     eig = eigen(H_FLQ(k))
     return eig.vectors[:, 2*Nm+1]
end

# Define Xa[k]
function Xa(k)
    A_vec = A(k)
    return [sum(A_vec[2i-1] for i in 1:2*Nm+1), sum(A_vec[2i] for i in 1:2*Nm+1)]
end

#
if kpt_with_border
    # include end points 
   kpts=range(-pi/2.0,pi/2.0,Nk)
else
  # Define kpts in such a way to exclude borders
  full_kpts=range(-pi/2.0,pi/2.0,Nk+2)
  kpts=(full_kpts[ik] for ik in 2:Nk+1)
end

wa(k) = conj(Xa(k)) ⋅ [a0(k), b0(k)]
function wa_diag(Xa_diag,TB_vec,k)
    wa_out=dot(Xa_diag, TB_vec[:,1])
    return wa_out
end

# Define B[k] (eigenvector corresponding to the (2*Nm+2)-th eigenvalue)
function B(k)
    eig = eigen(H_FLQ(k))
    return eig.vectors[:, 2*Nm+2]
end

# Define Xb[k]
function Xb(k)
    B_vec = B(k)
    return [sum(B_vec[2i-1] for i in 1:2*Nm+1), sum(B_vec[2i] for i in 1:2*Nm+1)]
end

function Xvec(eigenvecs)
    A_vec=eigenvecs[:,2*Nm+1]
    B_vec=eigenvecs[:,2*Nm+2]
    Xa_diag=[sum(A_vec[2i-1] for i in 1:2*Nm+1), sum(A_vec[2i] for i in 1:2*Nm+1)]
    Xb_diag=[sum(B_vec[2i-1] for i in 1:2*Nm+1), sum(B_vec[2i] for i in 1:2*Nm+1)]
    return Xa_diag,Xb_diag
end

# Define wb[k]
wb(k) = conj(Xb(k)) ⋅ [a0(k), b0(k)]
function wb_diag(Xb_diag,TB_vec,k)
    wb_out=dot(Xb_diag,TB_vec[:,1])
    return wb_out
end

FLQ_bands    =zeros(Float64,Nk, (h_dim*(2*Nm+1)))
FLQ_vecs     =zeros(ComplexF64,Nk, (h_dim*(2*Nm+1)),  (h_dim*(2*Nm+1)) )
TB_bands    =zeros(Float64,Nk, h_dim)
TB_vecs     =zeros(ComplexF64,Nk, h_dim, h_dim)

print("Calculate band structure: ")
for ik in ProgressBar(1:length(kpts))
#    print("Doing $k is H_FLQ(k) hermitian $(ishermitian(H(k))) \n")
    diag_FLQ = eigen(H_FLQ(kpts[ik]))
    FLQ_bands[ik,:]  =real(diag_FLQ.values)
    FLQ_vecs[ik,:,:] =diag_FLQ.vectors
    diag_TB = eigen(H_tb(kpts[ik]))
    TB_bands[ik,:]  =real(diag_TB.values)
    TB_vecs[ik,:,:] =diag_TB.vectors
end

title("Tight binding band structure ")
for n in 1:h_dim
  plot(kpts,TB_bands[:,n], label="Band $n")
end
PyPlot.show()


title("Floquet band structure ")
for n in 1:h_dim*(2*Nm+1)
  plot(kpts,FLQ_bands[:,n], label="Band $n")
end
PyPlot.show()


print("Calculate Wa and Wb: ")
wa_d=zeros(ComplexF64,Nk)
wb_d=zeros(ComplexF64,Nk)
wa2=zeros(Float64,Nk)
wb2=zeros(Float64,Nk)
for ik in ProgressBar(1:length(kpts))
    Xa_diag,Xb_diag=Xvec(FLQ_vecs[ik,:,:])
    wa_d[ik]=wa_diag(Xa_diag,TB_vecs[ik,:,:],kpts[ik])
    wb_d[ik]=wb_diag(Xb_diag,TB_vecs[ik,:,:],kpts[ik])    
    wa2[ik]=abs(wa_d[ik]^2)
    wb2[ik]=abs(wb_d[ik]^2)    
#    wa2[ik]=abs(wa(kpts[ik])^2)
#    wb2[ik]=abs(wb(kpts[ik])^2)    
end

plot(kpts,wb2-wa2, label="W difference")
plot(kpts,wb2+wa2, label="W sum")
PyPlot.show()

# Calculate Current

# Define Bound[l]
Bound(l) = (-Nm-1 < l < Nm+1) ? 1 : 0


# Define ChiA[k, j]
function ChiA(A_k, j)
    if -Nm-1 < j < Nm+1
        return [A_k[h_dim*(j+Nm)+1], A_k[h_dim*(j+Nm)+h_dim]]
    else
        return [0, 0]
    end
end

# Define ChiB[k, j]
function ChiB(B_k, j)
    if -Nm-1 < j < Nm+1
        return [B_k[h_dim*(j+Nm)+1], B_k[h_dim*(j+Nm)+h_dim]]
    else
        return [0, 0]
    end
end

# Define IHknl[k, N0, n, l]
# function IHknl(ik, N0, n, l)
#     term1 = abs(wa_d[ik])^2 * conj(ChiA(A_vec[ik], n-l+N0)) ⋅ (σ1 * ChiA(A_vec[ik], n))
#     term2 = abs(wb_d(ik))^2 * conj(ChiB(B_vec[ik], n-l+N0)) ⋅ (σ1 * ChiB(B_vec[ik], n))
#     return (term1 + term2) * im^l * besselj(l, F) * sin(k + l*π/2)
#     return N0+ik
# end

# Define IHk[N0, k]
# function IHk(N0, ik)
#     return sum(IHknl(ik, N0, n, l) for n in -Nm:Nm, l in -Nm:Nm)
# end
#
function I_func(kpt,n,F)
    return -(1.0im)^n * besselj(n, F) * sin(kpt + n*π/2)
end
#

# Define IH[N0]
function IH(N0)
  Nk=length(kpts)
  I_Nk=zeros(ComplexF64,Nk)
  for ik in 1:Nk
      A_vec=FLQ_vecs[ik,:,2*Nm+1]
      B_vec=FLQ_vecs[ik,:,2*Nm+2]
      for l in -Nm:Nm
         term_a = 0.0+0.0im
         term_b = 0.0+0.0im
         for n in -Nm:Nm
             term_a += abs(wa_d[ik])^2*dot(ChiA(A_vec,n-l+N0),σ1*ChiA(A_vec, n))
             term_b += abs(wb_d[ik])^2*dot(ChiB(B_vec,n-l+N0),σ1*ChiB(B_vec, n))
         end
         I_Nk[ik]+=(term_a+term_b)*I_func(kpts[ik],l,F)
      end
  end
  return sum(I_Nk)
end

print(" Calculate current up to $max_N0:")
I_N=zeros(ComplexF64,Int(max_N0/2))
N_order=zeros(Integer,Int(max_N0/2))

for iN in ProgressBar(1:2:max_N0)
    I_N[Int((iN+1)/2)]=abs(IH(iN))
    N_order[Int((iN+1)/2)]=iN
end
fig = PyPlot.figure("MyFigure",figsize=(10,5))
ax= PyPlot.axes()
PyPlot.yscale("log")
PyPlot.title("Current for the different Harmonics")
PyPlot.plot(N_order,I_N, label="Current",marker="o")
PyPlot.xlabel("Harmonic Order N")
PyPlot.ylabel("|I_H(N)/e|")
PyPlot.xticks(N_order)
PyPlot.show()

