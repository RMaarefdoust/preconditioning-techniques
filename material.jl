using SparseArrays
using LinearAlgebra
using BenchmarkTools
using MatrixDepot

function conjugate_gradient(A, b, x0, M, tol=1e-6, max_iter=1000)
    n = length(b)
    x = copy(x0)
    r = b - A*x
    z = M \ r  # Solve Mz = r
    ro_prev = dot(r, z)
    p = copy(z)
    for i in 1:max_iter
        q = A*p
        alpha = ro_prev / dot(p, q)
        x += alpha*p
        r -= alpha*q

        if norm(r) < tol
            break
        end

        z = M \ r
        ro = dot(r, z)
        Beta = ro / ro_prev
        p = z + Beta*p

        ro_prev = ro
    end
    return x
end

function jacobi_preconditioner(A)
    diag_A = diag(A)
    if any(diag_A .== 0)
        error("Matrix A has a zero diagonal entry. Preconditioning failed.")
    end
    D_inv = spdiagm(0 => 1.0 ./ diag_A)

    return D_inv
end

function gauss_seidel_preconditioner(A)
    L = tril(A, -1)
    diag_A = diag(A)
    if any(diag_A .== 0)
        # Add a small positive value to avoid division by zero
        diag_A .= max.(diag_A, eps())
    end
    D_inv = spdiagm(0 => 1.0 ./ diag_A)
    M = D_inv * L

    if any(M .== 0)
        error("Preconditioner M is singular. Preconditioning failed.")
    end

    return M
end


function incomplete_cholesky_preconditioner(A)
    # Compute incomplete Cholesky factorization
    L = cholesky(A, check=false)
    return L
end
function incomplete_cholesky_factorization(A)
    n = size(A, 1)
    L = spzeros(n, n)
    
    for j in 1:n
        Ljj = A[j, j]
        for k in 1:j-1
            if L[k, k] ≠ 0.0
                Ljj -= L[j, k]^2
            end
        end
        if Ljj ≤ 0.0
            error("Matrix is not positive definite.")
        end
        L[j, j] = sqrt(Ljj)
        for i in j+1:n
            Lij = A[i, j]
            for k in 1:j-1
                if L[k, k] ≠ 0.0
                    Lij -= L[i, k] * L[j, k]
                end
            end
            L[i, j] = Lij / L[j, j]
        end
    end
    
    return L
end


function preconditioned_CG(A, M, f, x0, tol, max_iter)
    n = length(f)
    x = copy(x0)
    r = f - A*x
    z = M \ r
    p = copy(z)
    P_prev = dot(r, z)

    for i in 1:max_iter
        q = A * p
        alpha = P_prev / dot(p, q)
        x += alpha * p
        r -= alpha * q

        if norm(r) ≤ tol
            println("Method converged")
            return x
        end

        z = M \ r
        P = dot(r, z)
        beta = P / P_prev
        p = z + beta * p
        P_prev = P
    end
end
# Define parameters for the metal plate simulation
n = 10  # Number of elements along one side of the square plate
num_elements = n^2  # Total number of elements
connectivity_factor = 0.7  # Connectivity factor between neighboring elements

# For example A representing the connectivity between elements
A = spdiagm(-1 => fill(connectivity_factor, num_elements - 1),
             0 => fill(1 + connectivity_factor, num_elements),
             1 => fill(connectivity_factor, num_elements - 1))
A = kron(sparse(I, n, n), A) + kron(A, sparse(I, n, n))  # Connect elements in both x and y directions
print("----------------",size(A))
# Generate the right-hand side vector b (random for demonstration purposes)
#b = rand(num_elements)
b = rand(size(A, 1))

# Initial guess for the solution
#x0 = zeros(num_elements)
x0 = zeros(size(A, 1))
# Tolerance and maximum number of iterations for the Conjugate Gradient method
tol = 1e-6
max_iter = 1000


# Incomplete Cholesky preconditioning
print("\n ##########   CG   ###########")
try  
    CG = preconditioned_CG(A, M, f, x0, tol, max_iter)
    @btime $x_BI_CG = conjugate_gradient($A, $b, $x0, $CG, $tol, $max_iter)
catch e
    println("Error in Incomplete CG preconditioning: ", e)
end

# Gauss-Seidel preconditioning
print("\n ##########   gauss_seidel   ###########\n")
try
    M_gs = gauss_seidel_preconditioner(A)
    x_gs = zeros(size(A, 1))  # Define x_gs before using it
    @btime $x_gs = conjugate_gradient($A, $b, $x0, $M_gs, $tol, $max_iter)
catch e
    println("Error in Gauss-Seidel preconditioning: ", e)
end



print("\n ##########   Jacobi   ###########\n")
# Jacobi preconditioning
 try
    M_jacobi = jacobi_preconditioner(A)
    x_jacobi = zeros(size(A, 1))  # Define x_jacobi before using it
    @btime $x_jacobi = conjugate_gradient($A, $b, $x0, $M_jacobi, $tol, $max_iter)
catch e
    println("Error in Jacobi preconditioning: ", e)
end

# Incomplete Cholesky preconditioning
print("\n ##########   Cholesky   ###########\n")
try  
    M_ic = incomplete_cholesky_preconditioner(A)
    x_ic = zeros(size(A, 1))  # Define x_ic before using it
    @btime $x_ic = conjugate_gradient($A, $b, $x0, $M_ic, $tol, $max_iter)
catch e
    println("Error in Incomplete Cholesky preconditioning: ", e)
end


