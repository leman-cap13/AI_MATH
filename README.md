Linear Algebra Algorithms from Scratch
This repository contains Python implementations of fundamental linear algebra algorithms, built from the ground up using NumPy. The primary goal is to demonstrate the underlying mechanics of these operations without relying on high-level library functions like np.linalg.inv().

The script includes:

Matrix Inversion using Gauss-Jordan Elimination with partial pivoting.

LU Decomposition using Doolittle's Algorithm without pivoting.

Verification of the results against standard SciPy and NumPy functions.

🚀 How It Works
The script is divided into two main parts, each implementing a core algorithm.

1. Matrix Inversion via Gauss-Jordan Elimination
The invert_matrix(A) function calculates the inverse of a square matrix A.

The algorithm follows these steps:

Augmentation: An identity matrix I of the same dimension as A is created. The two are combined to form an augmented matrix [A|I].

Forward Elimination & Pivoting: The script iterates through each row, performing elementary row operations to transform the left side (A) into an upper triangular matrix.

Partial Pivoting: To ensure numerical stability and avoid division by zero, the algorithm finds the row with the largest absolute value in the current column (the pivot) and swaps it with the current row.

Back Substitution: After the forward pass, another set of row operations is performed to zero out the elements above the diagonal, transforming the left side into the identity matrix I.

Result: As the left side becomes I, the right side is transformed into the inverse of the original matrix, A⁻¹. The final form of the augmented matrix is [I|A⁻¹].

2. LU Decomposition via Doolittle's Algorithm
The lu_decomposition(A) function factors a square matrix A into the product of a lower triangular matrix L and an upper triangular matrix U.


Important Note: This implementation does not use pivoting. This is a key reason why its output for L and U may differ from libraries like SciPy, which use pivoting by default.

🛠️ Usage
To run this script, you need Python 3, NumPy, and SciPy installed.

Prerequisites
pip install numpy scipy

Running the Script
Save the code as a Python file (e.g., linear_algebra.py).

Execute it from your terminal:

python linear_algebra.py

Expected Output
The script will print the original matrix, the step-by-step process of the Gauss-Jordan elimination, the calculated inverse, the calculated L and U matrices, and finally, a verification section comparing the results with NumPy and SciPy.

Original Matrix A:
[[2. 1. 3.]
 [4. 4. 7.]
 [2. 5. 9.]]
------------------------------
--- Part 3.1: Matrix Inverse from Scratch ---
Initial Augmented Matrix [A|I]:
[[2. 1. 3. 1. 0. 0.]
 [4. 4. 7. 0. 1. 0.]
 [2. 5. 9. 0. 0. 1.]]

... (step-by-step output) ...

--- Part 3.2: LU Decomposition from Scratch ---
L (from scratch):
[[1. 0. 0.]
 [2. 1. 0.]
 [1. 2. 1.]]
------------------------------
U (from scratch):
[[2. 1. 3.]
 [0. 2. 1.]
 [0. 0. 4.]]
------------------------------
--- Part 3.3: NumPy Verification ---
... (verification output) ...
Verification with SciPy (P@L@U == A): True

🔬 Understanding the SciPy Comparison
You will notice that the L and U matrices calculated by this script are different from those calculated by scipy.linalg.lu(). This is expected and does not mean either result is incorrect.

Our Result: A = L @ U

SciPy's Result: A = P @ L @ U

The difference is the Permutation Matrix (P). SciPy implements LU decomposition with partial pivoting to improve numerical stability. The P matrix keeps track of the row swaps performed during the decomposition. Our from-scratch implementation does not perform these swaps, leading to a different but still valid decomposition for the given matrix.

License
This project is licensed under the MIT License. See the LICENSE file for details.
