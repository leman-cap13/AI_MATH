import numpy as np
A = [
    [2, 1, 3],
    [4, 4, 7],
    [2, 5, 9]
]

b = [1, 1, 3]

# =====================================================
# PART 2.1: Gaussian Elimination from Scratch
# =====================================================

def gaussian_elimination(A, b):
    """Solve a system of linear equations Ax = b using Gaussian elimination.

    This function implements the Gaussian elimination algorithm with partial
    pivoting to ensure numerical stability. It can solve for a unique 
    solution or detect cases with no or infinite solutions.

    Parameters
    ----------
    A : list of list of (int or float)
        The NxN coefficient matrix of the linear system.
    b : list of (int or float)
        The N-element constant vector. Must have the same number of rows as A.

    Returns
    -------
    np.ndarray
        If a unique solution exists, returns a NumPy array representing the
        solution vector x.
    str
        If the system is inconsistent or has free variables, returns a
        string message: "No solutions" or "Infinite solutions".

    """
    n=len(A)
    # first i need to merge A and b to get Augmented Matrix
    Ab=[]
    for i in range(n):
        Ab.append(A[i][:]+[b[i]])  # I write [b[i]] because I want to sum two list --- matris's first
        #row and vector's first element but we cant sum list and element so I convert b[i] element
        # into list in order to sum two lists and get [1,2,3,1] and go on....
    # Now second part i need to define pivot and  show forward elimination
    for i in range(n):
        # here there is importand case --- fisrt we we define pivot we make sure that i cant be 0 because of zero 
        # devision and also make sure that it is not small number becasue when we devide it small number it is going
        # to be large amount of number which is we dont want it. in this case we gonna use pivoting 

        row=i
        for j in range(i+1,n): # for comparing row with other row i wrote (i+1)
            if abs(Ab[row][i]) < abs(Ab[j][i]):  #abs --beacuse positive or negative is okay we need to find biggest number without sign
                row=j
        # here if we find bigger element for first row and  we just swap
        Ab[i],Ab[row]=Ab[row],Ab[i]   
        if abs(Ab[i][i]) < 1e-10:
            if abs(Ab[i][n]) < 1e-10:
                return " Infinite solutions"
            else:
                return 'No solutions'
            

        # now lets define  pivot
        pivot=Ab[i][i]
        for j in range(i,n+1):
            Ab[i][j] /= pivot # Normalization in this part ve get pivot=1 and it makes easy for elimination
        # now add elimination part
        for k in range(i+1,n): # after first row start i+1 row our goal is to make zero under the pivot
            make_zero=Ab[k][i]  # k row and i column i need to make zero
            for j in range(i,n+1):
                Ab[k][j]-=make_zero*Ab[i][j]    

        # here we gonna add back substitution
    answer_list=[0]*n
    for i in range(n-1,-1,-1): # from last row to the first
        my_sum=0
        for j in range(i+1,n):
            my_sum+= (Ab[i][j]* answer_list[j] )
        answer_list[i] = Ab[i][n] - my_sum
    return np.array(answer_list  )      

solution_scratch=gaussian_elimination(A, b)
print(solution_scratch)


# =====================================================
# PART 2.2: NumPy Verification
# =====================================================


np_A = np.array(A, dtype=float)
np_b = np.array(b, dtype=float)

try:
    np_solution = np.linalg.solve(np_A, np_b)
    print("Solution (NumPy):", np_solution)
except np.linalg.LinAlgError as e:
    print("NumPy could not solve the system:", e)


# =====================================================
# Verification
# =====================================================
if isinstance(solution_scratch, list) or isinstance(solution_scratch, np.ndarray):
    are_solutions_close = np.allclose(solution_scratch, np_solution) # solution scratch is my result and np.solution is
    # numpy automatic solution when i use np.allclose i can compare this two result and come to conclution if they 
    # are similar or not.It is using  rtol=1e-05 ( relative tolerance), atol=1e-08 (absolute tolerance) as a default
    # value
    
 
    if are_solutions_close:
        print("succesfull")
    else:
        print("not seccusfull")
        print(f"Our result: {solution_scratch}")
        print(f"NumPy result: {np_solution}")
        
else:
    print(f": '{solution_scratch}'")

  

help(gaussian_elimination)