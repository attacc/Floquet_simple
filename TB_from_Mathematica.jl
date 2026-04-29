using LinearAlgebra
using SpecialFunctions  # For Bessel functions
using PyPlot
using ProgressBars


# Constants (assuming these are defined elsewhere)
Nm = 15  # Example value, replace with actual value
Q = 0.1  # Example value, replace with actual value
W = 0.25  # Example value, replace with actual value
F = 0.5  # Example value, replace with actual value
Nk = 100 # Example value, replace with actual value

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
    H_matrix = zeros(ComplexF64, (2*(2*Nm+1), 2*(2*Nm+1)))
    for n in -Nm:Nm
       for m in -Nm:Nm
#        for m in n:Nm
            if n == m
                H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] = (n * W * σ0 + Q * σ3)
            end
            H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] += (1.0im)^(m-n) * besselj(m-n, F) * cos(k - (m-n)*π/2) * σ1
#            H_matrix[2*(m+Nm)+1:2*(m+Nm)+2, 2*(n+Nm)+1:2*(n+Nm)+2] = conj(H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2])
        end
    end
    return H_matrix
end

function H_tb(k)
  H_tb = zeros(ComplexF64, (2, 2))
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

# Define wa[k]
kpts=range(0,pi/2.0,Nk)

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

FLQ_bands    =zeros(Float64,Nk, (2*(2*Nm+1)))
FLQ_vecs     =zeros(ComplexF64,Nk, (2*(2*Nm+1)),  (2*(2*Nm+1)) )
TB_bands    =zeros(Float64,Nk, 2)
TB_vecs     =zeros(ComplexF64,Nk, 2, 2)

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
for n in 1:2
  plot(kpts,TB_bands[:,n], label="Band $n")
end
PyPlot.show()


title("Floquet band structure ")
for n in 1:2*(2*Nm+1)
  plot(kpts,FLQ_bands[:,n], label="Band $n")
end
PyPlot.show()

print("Calculate Wa and Wb: ")
# wa=zeros(Float64,Nk)
# wb=zeros(Float64,Nk)
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
        return [A_k[2*(j+Nm)+1], A_k[2*(j+Nm)+2]]
    else
        return [0, 0]
    end
end

# Define ChiB[k, j]
function ChiB(B_k, j)
    if -Nm-1 < j < Nm+1
        return [B_k[2*(j+Nm)+1], B_k[2*(j+Nm)+2]]
    else
        return [0, 0]
    end
end

# Define IHknl[k, N0, n, l]
function IHknl(k, N0, n, l)
    term1 = abs(wa(k))^2 * conj(ChiA(k, n-l+N0)) ⋅ (σ1 * ChiA(k, n))
    term2 = abs(wb(k))^2 * conj(ChiB(k, n-l+N0)) ⋅ (σ1 * ChiB(k, n))
    return (term1 + term2) * im^l * besselj(l, F) * sin(k + l*π/2)
end

# Define IHk[N0, k]
function IHk(N0, k)
    return sum(IHknl(k, N0, n, l) for n in -Nm:Nm, l in -Nm:Nm)
end

# Define IH[N0]
function IH(N0)
    return sum(IHk(N0, -π/2 + π/Nk * i) for i in 0:Nk)
end
