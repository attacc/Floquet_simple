using LinearAlgebra
using SpecialFunctions

# Define Pauli matrices as Complex{Float64}
const σ0 = Complex{Float64}[1 0; 0 1]
const σ1 = Complex{Float64}[0 1; 1 0]
const σ2 = Complex{Float64}[0 -im; im 0]
const σ3 = Complex{Float64}[1 0; 0 -1]

const PauliMatrix = Dict(0 => σ0, 1 => σ1, 2 => σ2, 3 => σ3)

# Global parameters
global Nm, W, F, Q

function H(k::Real)
    dim = 2 * Nm + 1
    result = zeros(Complex{Float64}, 2*dim, 2*dim)
    im_float = Complex{Float64}(0, 1)  # Explicit complex float i
    
    for n in -Nm:Nm
        for m in -Nm:Nm
            idx_n = n + Nm + 1
            idx_m = m + Nm + 1
            
            # Diagonal block
            if n == m
                block = n * W * σ0 + Q * σ3
                result[2*idx_n-1:2*idx_n, 2*idx_m-1:2*idx_m] += block
            end
            
            # Off-diagonal coupling
            mn_diff = m - n
            # Use explicit Complex{Float64} i raised to power
            factor = (im_float^mn_diff) * besselj(abs(mn_diff), F) * cos(k - mn_diff * π / 2)
            block = factor * σ1
            result[2*idx_n-1:2*idx_n, 2*idx_m-1:2*idx_m] += block
        end
    end
    
    return Hermitian(result)
end

function a0(k::Real)
    return -1/sqrt(2) * sqrt(1 - Q / sqrt(Q^2 + cos(k)^2))
end

function b0(k::Real)
    return 1/sqrt(2) * sqrt(1 + Q / sqrt(Q^2 + cos(k)^2))
end

function E0(k::Real)
    return -sqrt(Q^2 + cos(k)^2)
end

function get_eigenvector(k::Real, index::Int)
    eig = eigen(H(k))
    sorted_indices = sortperm(eig.values)
    return eig.vectors[:, sorted_indices[index]]
end

function A(k::Real)
    return get_eigenvector(k, 2*Nm + 1)
end

function B(k::Real)
    return get_eigenvector(k, 2*Nm + 2)
end

function Xa(k::Real)
    vec = A(k)
    dim = 2*Nm + 1
    sum_odd = sum(vec[2*i - 1] for i in 1:dim)
    sum_even = sum(vec[2*i] for i in 1:dim)
    return [sum_odd, sum_even]
end

function Xb(k::Real)
    vec = B(k)
    dim = 2*Nm + 1
    sum_odd = sum(vec[2*i - 1] for i in 1:dim)
    sum_even = sum(vec[2*i] for i in 1:dim)
    return [sum_odd, sum_even]
end

function wa(k::Real)
    result = conj(Xa(k)) ⋅ [a0(k), b0(k)]
    return result
end

function wb(k::Real)
    result = conj(Xb(k)) ⋅ [a0(k), b0(k)]
    return result
end

function Bound(l::Int)
    return (-Nm - 1 < l < Nm + 1) ? 1 : 0
end

function ChiA(k::Real, j::Int)
    if -Nm - 1 < j < Nm + 1
        vec = A(k)
        return Complex{Float64}[vec[2*(j + Nm) + 1], vec[2*(j + Nm) + 2]]
    else
        return Complex{Float64}[0.0 + 0.0im, 0.0 + 0.0im]
    end
end

function ChiB(k::Real, j::Int)
    if -Nm - 1 < j < Nm + 1
        vec = B(k)
        return Complex{Float64}[vec[2*(j + Nm) + 1], vec[2*(j + Nm) + 2]]
    else
        return Complex{Float64}[0.0 + 0.0im, 0.0 + 0.0im]
    end
end

function IHknl(k::Real, N0::Int, n::Int, l::Int)
    im_float = Complex{Float64}(0, 1)
    
    chi_a_n = ChiA(k, n)
    chi_a_nl = ChiA(k, n - l + N0)
    term_a = abs2(wa(k)) * (conj(chi_a_nl)' * σ1 * chi_a_n)
    
    chi_b_n = ChiB(k, n)
    chi_b_nl = ChiB(k, n - l + N0)
    term_b = abs2(wb(k)) * (conj(chi_b_nl)' * σ1 * chi_b_n)
    
    return (term_a + term_b) * (im_float^l) * besselj(abs(l), F) * sin(k + l * π / 2)
end

function IHk(N0::Int, k::Real)
    sum_val = 0.0 + 0.0im
    for n in -Nm:Nm
        for l_val in -Nm:Nm
            sum_val += IHknl(k, N0, n, l_val)
        end
    end
    return sum_val
end

function IH(N0::Int, Nk::Int)
    sum_val = 0.0 + 0.0im
    for i in 0:Nk
        k = -π/2 + (π/Nk) * i
        sum_val += IHk(N0, k)
    end
    return sum_val
end

# Example usage
function example1()
    global Nm, W, F, Q
    Nm = 3
    W = 1.0
    F = 0.5
    Q = 0.1
    NN = 50
    
    eigenvalues_lower = Float64[]
    eigenvalues_upper = Float64[]
    
    for i in 0:NN
        k = i * π / (2 * NN)
        vals = eigvals(H(k))
        sorted_vals = sort(real(vals))
        push!(eigenvalues_lower, sorted_vals[2*Nm + 1])
        push!(eigenvalues_upper, sorted_vals[2*Nm + 2])
    end
    
    return eigenvalues_lower, eigenvalues_upper
end

function example2()
    global Nm, W, F, Q
    Nm = 3
    W = 1.0
    F = 0.5
    Q = 0.1
    NN = 50
    
    differences = Float64[]
    
    for i in 0:NN
        k = i * π / (2 * NN)
        diff = abs2(wb(k)) - abs2(wa(k))
        push!(differences, real(diff))
    end
    
    return differences
end

# Run examples
println("Running example 1...")
lower, upper = example1()
println("Lower eigenvalues (first 5): ", lower[1:min(5, end)])
println("Upper eigenvalues (first 5): ", upper[1:min(5, end)])

println("\nRunning example 2...")
diffs = example2()
println("Differences (first 5): ", diffs[1:min(5, end)])
