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
    """

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")

    # Initialize the starting vector x
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    eigenvalue = 0.0

    for _ in range(max_iterations):
        Ax = A @ x
        new_eigenvalue = np.linalg.norm(Ax)
        x_new = Ax / new_eigenvalue

        if np.abs(new_eigenvalue - eigenvalue) < tolerance:
            eigenvalue = new_eigenvalue
            x = x_new
            break

        eigenvalue = new_eigenvalue
        x = x_new

    return eigenvalue, x


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

    matrices = [('A', A),
                ('B', B),
                ('C', C),
                ('D', D),
                ('E', E),
                ('F', F),
                ('G', G),
                ('H', H)]

    for name, mat in matrices:
        print(f"Testing matrix {name} with the Power Method:")
        val, vec = power_method(mat)
        print("Dominant eigenvalue:", val)
        print("Corresponding eigenvector:", vec)
        print("-" * 40)
