import numpy as np
from scipy.linalg import null_space

def find_eigvalues_eigvectors_via_characteristic_polynomial(A):
    """
    Find the eigenvalues and eigenvectors of a square matrix A using the 
    characteristic polynomial method.

    This function:
    1. Computes the characteristic polynomial of A using np.poly(A).
    2. Finds the roots of the characteristic polynomial (these are the eigenvalues).
    3. For each eigenvalue λ, solves (A - λI)x = 0 to find the corresponding eigenvector(s).

    Parameters
    ----------
    A : numpy.ndarray
        A square matrix of shape (n, n) for which eigenvalues and eigenvectors are to be found.

    Returns
    -------
    eigenvalues : numpy.ndarray
        A 1D array containing the eigenvalues of A.
    eigenvectors : list of numpy.ndarray
        A list of arrays, where each array corresponds to one eigenvalue and contains 
        the eigenvector(s) associated with that eigenvalue. Each element in the list is 
        a 2D array of shape (n, k), where k might be greater than 1 if the eigenvalue has 
        a non-trivial eigenspace (more than one eigenvector).

    Notes
    -----
    - The method is based on the definition of eigenvalues and eigenvectors:
      
      The eigenvalues are the roots of the characteristic polynomial:
      det(A - λI) = 0

      Once λ (eigenvalue) is found, the eigenvector(s) x corresponding to λ are the 
      non-trivial solutions of:
      (A - λI)x = 0

    - This method can become numerically unstable or computationally expensive for large
      matrices, and is more of a theoretical approach rather than a practical one for 
      large-scale computations.

    - For numerical stability and efficiency, the built-in numpy.linalg.eig or similar 
      methods are usually preferred in practice.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    char_poly_coeffs = np.poly(A)
    eigenvalues = np.roots(char_poly_coeffs)
    eigenvectors = []

    for lamb in eigenvalues:
        M = A - lamb * np.eye(A.shape[0])
        nvectors = null_space(M)
        eigenvectors.append(nvectors)

    return eigenvalues, eigenvectors


if __name__ == "__main__":
    # Define all matrices
    A = np.array([[2, 1],
                  [3, 4]])
    B = np.array([[3, 2],
                  [3, 4]])
    C = np.array([[2, 3],
                  [1, 4]])
    D = np.array([[1, 1, 2],
                  [2, 1, 1],
                  [1, 1, 3]])
    E = np.array([[1, 1, 2],
                  [2, 1, 3],
                  [1, 1, 1]])
    F = np.array([[2, 1, 2],
                  [1, 1, 3],
                  [1, 1, 1]])
    G = np.array([[1, 1, 1, 2],
                  [2, 1, 1, 1],
                  [3, 2, 1, 2],
                  [2, 1, 1, 4]])
    H = np.array([[1, 2, 1, 2],
                  [2, 1, 1, 1],
                  [3, 2, 1, 2],
                  [2, 1, 1, 4]])

    # List of matrices and their labels
    matrices = [('A', A),
                ('B', B),
                ('C', C),
                ('D', D),
                ('E', E),
                ('F', F),
                ('G', G),
                ('H', H)]

    # Test each matrix with the characteristic polynomial method
    for name, mat in matrices:
        print(f"Testing matrix {name}:")
        vals, vecs = find_eigvalues_eigvectors_via_characteristic_polynomial(mat)
        print("Eigenvalues:", vals)
        for i, eigval in enumerate(vals):
            print(f"Eigenvectors for eigenvalue {eigval}:\n{vecs[i]}")
        print("-" * 40)
