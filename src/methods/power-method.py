import numpy as np

def power_method(A, max_iterations=1000, tolerance=1e-9):
    """
    Compute the dominant eigenvalue and corresponding eigenvector of a matrix A 
    using the Power Method.

    The Power Method is an iterative algorithm that approximates the eigenvalue of 
    largest magnitude (dominant eigenvalue) and its corresponding eigenvector. It does 
    so by repeatedly applying the matrix A to an initial vector x, and normalizing at 
    each step. Under suitable conditions (e.g., A has a unique eigenvalue with the 
    largest magnitude), the sequence converges to the dominant eigenvalue and eigenvector.

    Parameters
    ----------
    A : numpy.ndarray
        A square matrix of shape (n, n) for which we want to find the dominant eigenvalue 
        and eigenvector.
    max_iterations : int, optional
        The maximum number of iterations to run the algorithm. Default is 1000.
    tolerance : float, optional
        The tolerance for convergence. If the change in the estimated eigenvalue between 
        iterations is less than this value, the algorithm will terminate. Default is 1e-9.

    Returns
    -------
    eigenvalue : float
        The estimated dominant eigenvalue.
    eigenvector : numpy.ndarray
        The eigenvector corresponding to the dominant eigenvalue, normalized to have 
        unit norm (length 1).

    Notes
    -----
    - The Power Method converges under certain conditions. If the matrix A has a clear 
      dominant eigenvalue (i.e., one eigenvalue is strictly greater in magnitude than 
      all others), then the method will typically converge to that eigenvalue and its 
      eigenvector.
    - If the initial guess is chosen poorly, or if the conditions are not met, the 
      method may converge to a different eigenvalue or fail to converge.
    - Even if the method converges, the number of iterations needed can vary.

    Example
    -------
    >>> A = np.array([[2, 1],
                      [1, 2]])
    >>> eigenvalue, eigenvector = power_method(A)
    >>> print("Dominant eigenvalue:", eigenvalue)
    >>> print("Corresponding eigenvector:", eigenvector)
    """
    # Check if A is square
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")
    
    # Step 1: Initialize the starting vector x with a random vector or a uniform vector.
    # Using a random vector to start the iteration:
    x = np.random.rand(n)
    
    # Normalize the initial vector so we start with a vector of unit length
    x = x / np.linalg.norm(x)

    # Variable to store the eigenvalue estimate. We will compute it iteratively.
    eigenvalue = 0.0

    # Step 2: Iterate up to max_iterations times
    for _ in range(max_iterations):
        # Apply the matrix to the vector
        Ax = A @ x
        
        # Compute the norm of Ax to normalize and also to estimate the eigenvalue
        new_eigenvalue = np.linalg.norm(Ax)
        
        # Normalize Ax to get the next vector x
        x_new = Ax / new_eigenvalue
        
        # Check for convergence: if the difference between successive eigenvalue 
        # estimates is very small, we consider we have converged
        if np.abs(new_eigenvalue - eigenvalue) < tolerance:
            # Update eigenvalue and eigenvector before breaking
            eigenvalue = new_eigenvalue
            x = x_new
            break
        
        # Update for next iteration
        eigenvalue = new_eigenvalue
        x = x_new
    
    return eigenvalue, x


# Below is an example usage of the function.
# Note: This code is not part of the solution, it's just an illustration:
if __name__ == "__main__":
    # Example matrix
    A_example = np.array([[2, 1],
                          [1, 2]])
    # Apply the power method to find the dominant eigenvalue and eigenvector
    val, vec = power_method(A_example)
    print("Dominant eigenvalue:", val)
    print("Eigenvector:", vec)
