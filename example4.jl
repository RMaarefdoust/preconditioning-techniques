using SparseArrays
using LinearAlgebra
using BenchmarkTools
using MatrixDepot
function gauss_seidel_preconditioner(A)
    L = tril(A, -1)
    diag_A = diag(A)
    # Check if any diagonal entry is zero
    if any(diag_A .== 0)
        error("Matrix A has a zero diagonal entry. Preconditioning failed.")
    end
    # Add a small positive value to the diagonal entries to avoid division by zero
    diag_A .= max.(diag_A, eps())
    D_inv = spdiagm(0 => 1.0 ./ diag_A)
    M = D_inv * L

    return M
end

# Fallback to Jacobi preconditioning if Gauss-Seidel fails
function preconditioned_conjugate_gradient(A, b, x0, tol=1e-6, max_iter=1000)
    try
        M_gs = gauss_seidel_preconditioner(A)
        x, iterations, residuals = conjugate_gradient(A, b, x0, M_gs, tol, max_iter)
    catch e
        println("Gauss-Seidel preconditioning failed. Falling back to Jacobi preconditioning.")
        M_jacobi = jacobi_preconditioner(A)
        x, iterations, residuals = conjugate_gradient(A, b, x0, M_jacobi, tol, max_iter)
    end
    return x, iterations, residuals
end

# Load the Poisson matrix from MatrixDepot
A = matrixdepot("poisson", 5)
f = ones(size(A, 1))
x0 = zeros(length(f))
tol = 1e-6
max_iter = 100

# Benchmark the conjugate gradient with preconditioning
@time x, iterations, residuals = preconditioned_conjugate_gradient(A, f, x0, tol, max_iter)

println("Total iterations: ", iterations)
println("Final residual norm: ", residuals[end])
