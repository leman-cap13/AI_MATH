# ============================================
# Math4AI - Programming Assignment 1
# Vector Operations & Semantic Similarity
# ============================================

# --- Imports ---
import numpy as np

# --- Sample Word Vectors (3D Example) ---
# Note: Values here are just for demonstration
v_king   = [0.8, 0.65, 0.0]
v_man    = [0.6, 0.4,  0.0]
v_woman  = [0.7, 0.3,  0.2]
v_queen  = [0.9, 0.55, 0.2]

# =====================================================
# PART 1.1: King - Man + Woman Analogy (From Scratch)
# =====================================================

def vector_add(u, v):
    """Add two vectors u and v (lists)."""
    # TODO: Implement element-wise addition
    pass

def vector_sub(u, v):
    """Subtract vector v from u (lists)."""
    # TODO: Implement element-wise subtraction
    pass

# --- Analogy computation ---
# v_result = king - man + woman
# TODO: Use your vector_sub and vector_add here
v_result = None

print("Analogy result (from scratch):", v_result)


# =====================================================
# PART 1.2: Cosine Similarity (From Scratch)
# =====================================================

def dot_product(u, v):
    """Compute the dot product of u and v."""
    # TODO: Implement dot product using a loop
    pass

def norm(u):
    """Compute the Euclidean norm (L2 norm) of vector u."""
    # TODO: Implement norm from scratch
    pass

def cosine_similarity(u, v):
    """Compute cosine similarity between u and v."""
    # TODO: Use dot_product() and norm() here
    pass

# --- Cosine similarity between analogy result & queen ---
similarity_scratch = None  # TODO: Compute using your function
print("Cosine similarity (from scratch):", similarity_scratch)


# =====================================================
# PART 1.3: NumPy Verification
# =====================================================

# Convert to numpy arrays
np_king   = np.array(v_king)
np_man    = np.array(v_man)
np_woman  = np.array(v_woman)
np_queen  = np.array(v_queen)

# --- Analogy computation with NumPy ---
np_result = None  # TODO: Implement with NumPy
print("Analogy result (NumPy):", np_result)

# --- Cosine similarity with NumPy ---
# TODO: Use np.dot() and np.linalg.norm()
similarity_numpy = None
print("Cosine similarity (NumPy):", similarity_numpy)


# =====================================================
# Verification
# =====================================================
# TODO: Print comparison of scratch vs NumPy results
