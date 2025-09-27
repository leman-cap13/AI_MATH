import numpy as np
from scipy.linalg import lu as scipy_lu  

def print_matrix(name, m):
    """
    Helper function to print a matrix with its name.
    Handles None for non-invertible matrices.
    """
    print(f"{name}:")
    if m is None:
        print("None (Matrix is singular or function not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(m)
    print("-" * 30)

A = np.array([
    [2., 1., 3.],
    [4., 4., 7.],
    [2., 5., 9.]
])

print_matrix("Original Matrix A", A)    

# ====================================================================
# Part 3.1: Matrix Inverse via Gauss-Jordan Elimination
# ====================================================================


def invert_matrix(A):
    """
    Computes the inverse of a square matrix A using Gauss-Jordan elimination.
    Args:
        A (np.ndarray): A square numpy array. 
    Returns:
        np.ndarray: The inverse of A, or None if A is singular.
    """

    A = A.astype(float)
    
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    # 1. Create the augmented matrix [A | I]
    identity = np.identity(n)
    augmented_A = np.hstack((A, identity))
    print("Initial Augmented Matrix [A|I]:")
    print(augmented_A)
    print("\nStarting Gauss-Jordan Elimination...")
    print("-" * 30)
    # in the first part we need to calculate pivot place
    for i in range(n):
        pivot_row = i
        for k in range(i + 1, n):
            if abs(augmented_A[k][i]) > abs(augmented_A[pivot_row][i]):
                pivot_row = k

        if pivot_row != i:
            augmented_A[[i, pivot_row]] = augmented_A[[pivot_row, i]]

        pivot_val = augmented_A[i][i] # it can be written like pivot_val=0: but using np.isclose is more
                                       # professional
        if np.isclose(pivot_val, 0):
            print("Matrix is singular, an inverse cannot be found.")
            return None

        #  Normalize the pivot row
        augmented_A[i] = augmented_A[i] / pivot_val

        #  Elimination: Zero out other elements in the current column
        for j in range(n):
            if i != j:
                factor = augmented_A[j][i]
                augmented_A[j] -= factor * augmented_A[i]
        
        print(f"After step {i+1} (pivot at row {i}):")
        print(augmented_A)
        print("-" * 30)
    

    inverse_A = augmented_A[:, n:]
    return inverse_A



# --- Calling the function for Part 3.1 ---
print("--- Part 3.1: Matrix Inverse from Scratch ---")
A_inv_scratch = invert_matrix(A.copy()) # Use a copy to keep original A intact
print_matrix("Inverse A (from scratch)", A_inv_scratch)   

if A_inv_scratch is not None:
    print("\nVerification (A * A^-1):")
    print(np.dot(A, A_inv_scratch))

# ====================================================================
# Part 3.2: LU Decomposition from Scratch
# ====================================================================


def lu_decomposition(A):
    """
    Performs LU decomposition of a square matrix A using Doolittle's algorithm.
    
    Args:
        A (np.ndarray): A square numpy array.
        
    Returns:
        (np.ndarray, np.ndarray): A tuple of (L, U) matrices.
    """
    A = A.astype(float)
    
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    L = np.identity(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i,n):
            sum_val=sum(L[i,k] * U[k,j] for k in range(i))
            U[i,j]=A[i,j]-sum_val

        for j in range(i+1,n):
            sum_val=sum(L[j,k] * U[k,i] for k in range(i))
            if np.isclose(U[i,i],0):
               return None
            L[j,i]=(A[j,i]-sum_val) / U[i,i]
    
    return L, U


print("--- Part 3.2: LU Decomposition from Scratch ---")
L_scratch, U_scratch = lu_decomposition(A.copy())
print_matrix("L (from scratch)", L_scratch)
print_matrix("U (from scratch)", U_scratch)


# ====================================================================
# Part 3.3: NumPy Verification
# ====================================================================
print("--- Part 3.3: NumPy Verification ---")


print("Verifying Matrix Inverse...")
A_inv_numpy = np.linalg.inv(A)
print_matrix("Inverse A (NumPy)", A_inv_numpy)


print("Verifying LU Decomposition...")

if L_scratch is not None and U_scratch is not None:
    product_LU = L_scratch @ U_scratch
    print_matrix("L @ U (from scratch)", product_LU)
    print_matrix("Original A (for comparison)", A)
    

    is_correct = np.allclose(A, product_LU)
    print(f"Verification Check (A == L @ U): {is_correct}\n")
else:
    print("LU decomposition not yet implemented.\n")
    
# Optional: Compare with SciPy's LU decomposition
from scipy.linalg import lu as scipy_lu
P, L_scipy, U_scipy = scipy_lu(A)
print_matrix("L (SciPy)", L_scipy)
print_matrix("U (SciPy)", U_scipy)
print(f"Verification with SciPy (P@L@U == A): {np.allclose(A, P @ L_scipy @ U_scipy)}")