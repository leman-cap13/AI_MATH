import numpy as np
from scipy.linalg import null_space

def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_vectors(name, vecs):
    """Prints a list of vectors."""
    print(f"{name}:")
    if not vecs:
        print("[]")
    else:
        for i, v in enumerate(vecs):
            # Reshape to ensure it's a column vector for printing
            print(f"  Vector {i+1}:\n{v.reshape(-1, 1)}")
    print("-" * 40)

# --- Problem Setup ---
# The matrix A and vector b for this assignment
A = np.array([
    [1., 2., 3., 5.],
    [2., 4., 8., 12.],
    [3., 6., 7., 13.]
])

b = np.array([4., 10., 10.])

print_matrix("Original Matrix A", A)
print_matrix("Original Vector b", b.reshape(-1, 1))

def to_rref(M):
    """
    Converts a matrix M to its Reduced Row Echelon Form (RREF).
    
    Args:
        M (np.ndarray): The input matrix.
        
    Returns:
        np.ndarray: The RREF of M.
    """
    M_copy=M.copy()
    n,m=M_copy.shape
    row=0
    for i in range(m): # 0
        pivot_index=row
        # print('---',pivot_index)
        for j in range(row+1,n): # 1
            if abs(M_copy[j][i]) > abs(M_copy[pivot_index][i]):
                pivot_index=j  # i wrote this beacuse i want to define piwot max number for normalization after
                # print(pivot_index) 

        if pivot_index != row:
            M_copy[[row,pivot_index]]=M_copy[[pivot_index,row]]  # this is for swap now my first pivot is 3
            # print('copy',M_copy)   
            
            
        pivot_value=M_copy[row][i]
        # print(pivot_value)
        if np.isclose(M_copy[row][i], 0):
            continue
        M_copy[row]=M_copy[row] / pivot_value
     
        
        for k in range(n):
            if row != k:
                subtract_val=M_copy[k][i]
                M_copy[k]-=subtract_val*M_copy[row]
        row+=1        

    print("RREF function is not implemented. Returning original matrix.")
    return M_copy

# ====================================================================
# Part 4.1: Finding the Nullspace Basis
# ====================================================================

def find_nullspace_basis(A):
    """
    Finds the basis for the nullspace of matrix A.
    
    Args:
        A (np.ndarray): The input matrix.
        
    Returns:
        list: A list of numpy arrays, where each array is a basis vector for the nullspace.
    """
    m, n = A.shape
    
    # 1. Compute the RREF of A
    rref_A = to_rref(A)
    print_matrix("RREF of A", rref_A)
    
    basis_vectors = []
    
    pivot_columns=[]
    free_columns=[]
    for rows in rref_A:
        #print(rows)
        for column, element in enumerate(rows):
            if element==1:
                pivot_columns.append(column)
                #print('pivot',pivot_columns)
                break
                
    for j in range(n):
         if j not in pivot_columns:
             free_columns.append(j)
             #print('free',free_columns)
             
    for k in free_columns:
        basis_vector=np.zeros(n) 
        basis_vector[k]=1
        #print('basis', basis_vector)
        
        for pivot_row, pivot_col in enumerate(pivot_columns):
            basis_vector[pivot_col]=-rref_A[pivot_row,k]
            #print('*',basis_vector)
            
    basis_vectors.append(basis_vector)
    
    return basis_vectors

# --- Calling the function for Part 4.1 ---
print("--- Part 4.1: Finding the Nullspace Basis ---")
nullspace_basis = find_nullspace_basis(A.copy())
print_vectors("Nullspace Basis (from scratch)", nullspace_basis)

# ====================================================================
# Part 4.2: Verification of the Nullspace
# ====================================================================
print("--- Part 4.2: Verification of the Nullspace ---")

if not nullspace_basis:
    print("Nullspace basis is empty or not implemented.")
else:
    for i, v in enumerate(nullspace_basis):
        # For each basis vector v, A @ v should be the zero vector
        result = A @ v
        print(f"Verifying basis vector {i+1}: A @ v_{i+1}")
        print_matrix(f"Result (should be zero vector)", result.reshape(-1, 1))
        # Check if it's close to zero to handle floating point errors
        print(f"Is close to zero? {np.allclose(result, 0)}\n")

# --- Verification using SciPy ---
print("--- Verifying with SciPy ---")
scipy_ns = null_space(A)
print_matrix("Nullspace basis from SciPy (orthonormal)", scipy_ns)


# ====================================================================
# Part 4.3: Finding a Particular Solution
# ====================================================================

def find_particular_solution(A, b):
    """
    Finds one particular solution to the system Ax = b.
    
    Args:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The target vector.
        
    Returns:
        np.ndarray: The particular solution vector x_p, or None if no solution exists.
    """
    m, n = A.shape
    
    # 1. Create the augmented matrix [A | b]
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    print_matrix("Augmented Matrix [A|b]", augmented_matrix)
    
    # 2. Compute the RREF of the augmented matrix
    rref_augmented = to_rref(augmented_matrix)
    print_matrix("RREF of Augmented Matrix", rref_augmented)
    
    particular_solution = None
    for rref_row in rref_augmented:
        # print(rref_row[-1])
        if rref_row[-1]!=0 and np.allclose(rref_row[:-1],0):
            return None
    pivot_columns=[]
    for row in rref_augmented:
        for col, element in enumerate(row[:-1]):
            if element==1:
                pivot_columns.append(col)
                # print('pivot', pivot_columns)
                break

    particular_solution=np.zeros(n)
    for row,col in enumerate(pivot_columns) :
            particular_solution[col]=rref_augmented[row,-1] # [row,-1] show last elemnt in the row and we write
            # this elemnet in our particular solution list... in the place of pivot_columsn..for example
            # pivot columns contains 0 and 2 this means 0 index is gonna be 1 in particular solution
            # and 2 index is gonna be 1 beacuse enumerate fuction returns index and value son 1 row 2 column 
            # [0. 0. 1. 1. 1.] this one...and last value is 1 so I add 1 to my particular solution's 2 index                                               
            # print('partcular',particular_solution)
            
    return particular_solution
  

# --- Calling the function for Part 4.3 ---
print("--- Part 4.3: Finding a Particular Solution ---")
x_p = find_particular_solution(A.copy(), b.copy())
print_matrix("Particular Solution x_p (from scratch)", x_p.reshape(-1, 1) if x_p is not None else None)

# ====================================================================
# Part 4.4: Constructing and Verifying the Complete Solution
# ====================================================================
print("--- Part 4.4: Constructing and Verifying the Complete Solution ---")

if x_p is None or not nullspace_basis:
    print("Cannot construct complete solution: particular solution or nullspace is missing.")
else:
    # 1. Get the components
    print("Particular solution x_p and nullspace basis are available.")
    
    # 2. Create a vector x_n from the nullspace
    # Example: x_n = 2*s1 - 3*s2
    # Let's use scalars c1=2, c2=-3 if there are at least two basis vectors
    scalars = [2, -3, 1, -1.5] # Add more if needed
    x_n = np.zeros(A.shape[1])
    
    print("\nConstructing x_n from nullspace basis:")
    for i, v in enumerate(nullspace_basis):
        if i < len(scalars):
            print(f"Adding {scalars[i]} * v_{i+1}")
            x_n += scalars[i] * v
    
    print_matrix("Constructed Nullspace Vector x_n", x_n.reshape(-1, 1))

    # 3. Construct the complete solution x_new = x_p + x_n
    x_new = x_p + x_n
    print_matrix("Complete Solution x_new = x_p + x_n", x_new.reshape(-1, 1))
    
    # 4. The ultimate test: A @ x_new should equal b
    print("--- The Ultimate Test ---")
    final_result = A @ x_new
    print_matrix("A @ x_new", final_result.reshape(-1, 1))
    print_matrix("Original b (for comparison)", b.reshape(-1, 1))
    
    is_correct = np.allclose(final_result, b)
    print(f"Verification Check (A @ x_new == b): {is_correct}")