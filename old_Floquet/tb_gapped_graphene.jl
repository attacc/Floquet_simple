using LinearAlgebra
using Plots

# 1. Physical Parameters
t = 2.8  # Hopping parameter in eV
a = 1.42 # Carbon-Carbon distance in Angstroms
gap = 2.0

# 2. Lattice Vectors & High Symmetry Points
# Primitive lattice vectors
a1 = [3a/2, sqrt(3)*a/2]
a2 = [3a/2, -sqrt(3)*a/2]

# High symmetry points in Reciprocal Space
K = [2π/(3a), 2π/(3*sqrt(3)*a)]
M = [2π/(3a), 0.0]
Γ = [0.0, 0.0]

# 3. Create a Path in K-space (Γ -> M -> K -> Γ)
function generate_path(pts, num_points=100)
    path = []
    for i in 1:(length(pts)-1)
        p1, p2 = pts[i], pts[i+1]
        for α in range(0, 1, length=num_points)
            push!(path, p1 * (1-α) + p2 * α)
        end
    end
    return path
end

k_path = generate_path([Γ, M, K, Γ])

# 4. The Bloch Hamiltonian
# For Graphene, the off-diagonal element is f(k) = -t * Σ exp(i k · δ)
function get_bands(k)
    # Nearest neighbor vectors
    δ1 = [a, 0]
    δ2 = [-a/2, sqrt(3)*a/2]
    δ3 = [-a/2, -sqrt(3)*a/2]
    
    f_k = -t * (exp(im * dot(k, δ1)) + exp(im * dot(k, δ2)) + exp(im * dot(k, δ3)))
    
    # Hamiltonian Matrix
    H = [gap/2.0      f_k;
         conj(f_k) -gap/2.0]
    
    return eigvals(H)
end

# 5. Calculation
energies = [get_bands(k) for k in k_path]
band1 = [e[1] for e in energies]
band2 = [e[2] for e in energies]

# 6. Plotting (The Display Fix)
p = plot(real.(band1), color=:blue, label="Valence Band", title="Graphene Band Structure")
plot!(p, real.(band2), color=:red, label="Conduction Band")
ylabel!("Energy (eV)")
xticks!([1, 100, 200, 300], ["Γ", "M", "K", "Γ"])

display(p) # This ensures the plot pops up
gui()
readline()
