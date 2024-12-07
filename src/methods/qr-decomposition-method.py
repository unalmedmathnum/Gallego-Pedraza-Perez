import numpy as np

def qr_eigenvalues_eigenvectors(A, max_iterations=1000, tolerance=1e-9):
    """
    Compute the eigenvalues and eigenvectors of a matrix A using the QR decomposition method.

    The QR Method is an iterative algorithm for finding all eigenvalues (and, under certain 
    conditions, the eigenvectors) of a matrix. It works by repeatedly factorizing the matrix 
    A into a product A = Q * R, where Q is an orthogonal matrix and R is an upper-triangular 
    matrix. Then, the matrix is updated as A ← R * Q. Under suitable conditions, this 
    iteration causes A to approach a quasi-diagonal form whose diagonal elements are the 
    eigenvalues of the original matrix.

    Additionally, if we accumulate the Q matrices from each iteration, their product 
    converges to a matrix whose columns are the eigenvectors corresponding to the eigenvalues.

    Parameters
    ----------
    A : numpy.ndarray
        A square matrix of shape (n, n) for which we want to find eigenvalues and eigenvectors.
    max_iterations : int, optional
        The maximum number of iterations to run the algorithm. Default is 1000.
    tolerance : float, optional
        The tolerance for convergence. If the off-diagonal elements of A become sufficiently 
        small (below this threshold), we consider the matrix to be (nearly) upper-triangular 
        and terminate. Default is 1e-9.

    Returns
    -------
    eigenvalues : numpy.ndarray
        A 1D array containing the eigenvalues of A (approximated from the diagonal of the 
        final matrix after convergence).
    eigenvectors : numpy.ndarray
        A matrix whose columns are the eigenvectors corresponding to the eigenvalues. 
        Note that for certain matrices (e.g., non-diagonalizable, complex eigenvalues),
        additional care may be needed, and the method might produce a Schur form rather 
        than a strictly diagonal form.

    Notes
    -----
    - The QR method is typically more stable and efficient if variations such as 
      the QR algorithm with shifts are used. The basic QR iteration shown here 
      may converge slowly or may not converge at all for some matrices.
    - This implementation assumes that the matrix is diagonalizable and that 
      the iteration will converge to a form from which we can read off the eigenvalues.
    - The method as presented is for educational purposes. In practice, one would 
      typically use a built-in linear algebra function (e.g., numpy.linalg.eig) or 
      a more sophisticated QR-based algorithm from libraries like SciPy or LAPACK.

    Example
    -------
    >>> A = np.array([[2, 1],
                      [1, 2]])
    >>> eigenvalues, eigenvectors = qr_eigenvalues_eigenvectors(A)
    >>> print("Eigenvalues:", eigenvalues)
    >>> print("Eigenvectors:\n", eigenvectors)
    """
    # Check if A is square
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")
    
    # Copy A to avoid modifying the original matrix
    A = A.astype(float).copy()
    
    # Initialize Q_accum as the identity matrix so we can accumulate Q's from each iteration
    Q_accum = np.eye(n)

    for _ in range(max_iterations):
        # Perform QR decomposition of A
        Q, R = np.linalg.qr(A)
        
        # Update A by RQ
        A = R @ Q
        
        # Accumulate Q into Q_accum to track the eigenvectors
        Q_accum = Q_accum @ Q
        
        # Check for convergence by looking at the off-diagonal elements
        # If they are sufficiently small, we assume convergence
        off_diag_norm = np.linalg.norm(A - np.diag(np.diag(A)))
        if off_diag_norm < tolerance:
            break
    
    # After convergence, the eigenvalues are on the diagonal of A
    eigenvalues = np.diag(A)
    
    # The columns of Q_accum are the eigenvectors
    eigenvectors = Q_accum

    return eigenvalues, eigenvectors


# Example usage:
if __name__ == "__main__":
    A_example = np.array([[2, 1],
                          [1, 2]])
    eigvals, eigvecs = qr_eigenvalues_eigenvectors(A_example)
    print("Eigenvalues:", eigvals)
    print("Eigenvectors:\n", eigvecs)
