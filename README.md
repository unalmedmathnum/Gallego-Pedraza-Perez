# Matrix Eigenvalue and Eigenvector Methods

## Authors:

1. Carlos Andres Gallego Montoya
2. Gustavo Adolfo Perez Perez
3. Sebastian Pedraza Rendon

## Introduction:

This project demonstrates three methods for calculating eigenvalues and eigenvectors of matrices:

1. **Characteristic Polynomial Method**  
2. **Power Method**  
3. **QR Decomposition Method**

The source code for these methods is located in the `src/methods` directory.

## Project Structure

```
├─ report
    ├─ Gallego Montoya- Pedraza Rendón- Pérez Pérez.pdf
    └─ Gallego Montoya- Pedraza Rendón- Pérez Pérez.tex
├─ src
    ├─ methods
        ├─ characteristic-polynomial-method.py
        ├─ power-method.py
        └─ qr-decomposition-method.py
    ├─ results
        ├─ characteristic-polynomial-method.pdf
        ├─ power-method.pdf
        └─ qr-decomposition-method.pdf
    └─ tests
        ├─ characteristic-polynomial-method.py
        ├─ power-method.py 
        └─ qr-decomposition-method.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

- `requirements.txt`: Contains the dependencies needed to run the project.
- `src/methods/`: Contains the Python scripts implementing the eigenvalue and eigenvector methods.

## Getting Started

### Prerequisites

- **Python 3.8+** (It's recommended to have at least Python 3.8)
- **git** (if you are pulling the project from a Git repository)

### Setting Up a Virtual Environment

Although a virtual environment is not included in the repository (as it's ignored in `.gitignore`), you can easily create one. This ensures that all dependencies are installed in an isolated environment, preventing version conflicts with other projects.

1. **Create the virtual environment:**
   
   ```bash
   python3 -m venv venv
   ```
   This command creates a new virtual environment named venv in the project root directory.

2. **Activate the virtual environment:**

    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```
    On Windows:
    ```bash
    venv\Scripts\activate
    ```
    After this, you should see the environment name (e.g., (venv)) at the beginning of your terminal prompt, indicating that you are working inside the virtual environment.

### Installing Dependencies

With the virtual environment activated, run:
```bash
    pip install -r requirements.txt
```
This will install all the necessary dependencies (e.g., numpy, scipy) required to run the code.

### Running the Methods

Inside the `src/methods` directory, you will find the Python files that contain the code for each method.

1. **`characteristic_polynomial_method.py`**
2. **`power_method.py`**
3. **`qr_method.py`**

You can run them directly, for example:
```bash
    python src/methods/characteristic-polynomial-method.py
```

### Running the tests

Inside the `src/tests` directory, you will find the Python files that contain the code for each method with each matrix given in the assigments.

1. **`characteristic_polynomial_method.py`**
2. **`power_method.py`**
3. **`qr_method.py`**

You can run them directly, for example:
```bash
    python src/tests/characteristic-polynomial-method.py
```

### Results about tests
Inside the `src/results` directory, you will find the PDF files that contain results, highlighting the advantages and disadvantages of each method for each matrix in the assigments.

This project is for educational and demonstration purposes, showing how various methods for eigenvalue decomposition can be implemented from scratch.