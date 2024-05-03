using SparseArrays
using LinearAlgebra
using MatrixDepot

function preconditioned_CG(A, M, f, x0, tol, max_iter)
    n = length(f)
    x = copy(x0)
    r = f - A*x
    z = M \ r
    p = copy(z)
    P_prev = dot(r, z)
    iterations = 0  # Initialize iteration counter
    residuals = Float64[]  # Initialize array to store residual norms
    for i in 1:max_iter
        q = A * p
        alpha = P_prev / dot(p, q)
        x += alpha * p
        r -= alpha * q
        residual_norm = norm(r)
        push!(residuals, residual_norm)  # Store residual norm
        iterations += 1  # Increment iteration counter

        if residual_norm ≤ tol
            println("Method converged in $iterations iterations")
            return x, residuals
        end

        z = M \ r
        P = dot(r, z)
        beta = P / P_prev
        p = z + beta * p
        P_prev = P
    end

    println("Method did not converge within the maximum number of iterations.")
    return x, residuals
end

A = matrixdepot("poisson", 10)
f = ones(size(A, 1))
x0 = zeros(length(f))
M = incomplete_cholesky_factorization(A)
tol = 1e-6
max_iter = 100

@time x, residuals = preconditioned_CG(A, M, f, x0, tol, max_iter)

println("Total iterations: ", length(residuals))
println("Final residual norm: ", residuals[end])

