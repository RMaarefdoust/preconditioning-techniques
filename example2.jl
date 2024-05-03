using SparseArrays
using LinearAlgebra
using MatrixDepot
function preconditioned_BI_CG(A, M, f, x0, tol, max_iter)
    n = length(f)
    x = copy(x0)
    r = f - A * x
    p = copy(r)
    er = copy(r)
    W = 0.0
    P_prev = dot(er, r)  # Initialize P_prev here
    alpha = 0.0  # Initialize alpha here

    iterations = 0  # Initialize iteration counter
    residuals = Float64[]  # Initialize array to store residual norms

    for iter in 1:max_iter
        Pi = dot(er, r)
        if Pi == 0.0
            println("Method failed: Pi is zero")
            return x, residuals
        end

        if iter > 1
            if W == 0.0
                println("Method failed: W is zero")
                return x, residuals
            end

            beta = (Pi / P_prev) * alpha / W
            p = r + beta * p
        end

        p̂ = M \ p
        q = A * p̂
        alpha = Pi / dot(er, q)
        x += alpha * p̂

        r -= alpha * q

        residual_norm = norm(r)
        push!(residuals, residual_norm)  # Store residual norm
        iterations += 1  # Increment iteration counter

        if residual_norm ≤ tol
            println("Method converged in $iterations iterations")
            return x, residuals
        end

        ŝ = M \ r
        t = A * ŝ
        W = dot(t, r) / dot(t, t)
        x += W * ŝ
        r -= W * t
        P_prev = Pi
        er = copy(r)
    end

    println("Method did not converge within the maximum number of iterations.")
    return x, residuals
end

# Load the Poisson matrix from MatrixDepot
A = matrixdepot("poisson", 10)
f = ones(size(A, 1))
x0 = zeros(length(f))

# Define the preconditioning matrix M (incomplete Cholesky factorization of A)
M = incomplete_cholesky_factorization(A)

# Set the tolerance and maximum iteration count
tol = 1e-6
max_iter = 100

# Call the preconditioned_BI_CG function
@time x, residuals = preconditioned_BI_CG(A, M, f, x0, tol, max_iter)

println("Total iterations: ", length(residuals))
println("Final residual norm: ", residuals[end])

@time x2 = conjugate_gradient(A, f, x, M, tol, max_iter)
println("Solution x:")
println(x2)
