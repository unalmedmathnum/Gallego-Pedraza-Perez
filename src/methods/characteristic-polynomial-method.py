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
    # Check if A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    # Step 1: Compute the characteristic polynomial of A.
    # np.poly(A) returns the coefficients of the characteristic polynomial of A.
    # If A is an n x n matrix, np.poly(A) returns a vector of length n+1.
    # The polynomial is given by: p(λ) = λ^n + c_{n-1}λ^{n-1} + ... + c_1 λ + c_0
    char_poly_coeffs = np.poly(A)

    # Step 2: Compute the roots of the characteristic polynomial.
    # These roots are the eigenvalues.
    eigenvalues = np.roots(char_poly_coeffs)

    eigenvectors = []

    # Step 3: For each eigenvalue λ, solve (A - λI)x = 0.
    # To find eigenvectors, we need the null space of (A - λI).
    for lamb in eigenvalues:
        # Form (A - λI)
        # np.eye(A.shape[0]) creates an identity matrix of the same dimension as A.
        M = A - lamb * np.eye(A.shape[0])

        # Find the null space of M.
        # The null space of M consists of all vectors x such that Mx = 0.
        # These are precisely the eigenvectors corresponding to λ.
        # null_space(M) returns an orthonormal basis for the null space of M.
        nvectors = null_space(M)

        # Store the eigenvectors in the list.
        # If no non-trivial solution is found, this might be an empty array.
        # Usually, if λ is an eigenvalue, null_space(M) should return at least one vector.
        eigenvectors.append(nvectors)

    return eigenvalues, eigenvectors


# Below is an example usage of the function.
# This code is not part of the solution, it's just an illustration:
if __name__ == "__main__":
    # Example matrix
    A_example = np.array([[2, 1],
                          [1, 2]])

    # Find eigenvalues and eigenvectors using the characteristic polynomial method
    vals, vecs = find_eigvalues_eigvectors_via_characteristic_polynomial(A_example)

    # Print results
    # Note: The order of eigenvalues and eigenvectors might differ from 
    # np.linalg.eig due to numerical differences.
    print("Eigenvalues:", vals)
    for i, v in enumerate(vecs):
        print(f"Eigenvectors for eigenvalue {vals[i]}:\n{v}")
