import numpy as np

import mymath.Vector as v
import mymath.TapeMatrix as tm

class ExperimentUtils:
    @staticmethod
    def band_cholesky(matrix):
        """
        Разложение Холецкого для симметричной положительно определенной ленточной матрицы
        Возвращает: (L, L^T) где L - нижняя треугольная ленточная матрица с той же шириной ленты
        """
        if matrix.bandwidth % 2 != 1:
            raise ValueError("Matrix must have odd bandwidth for symmetric storage")
        
        n = matrix.size
        k = matrix.k  # полуширина ленты
        
        # Создаем нижнюю треугольную матрицу L с той же шириной ленты
        L = tm.TapeMatrix(size=n, bandwidth=2*k+1)  # Для нижнего треугольника нужна вся ширина ленты
        
        # Алгоритм Холецкого для ленточных матриц
        for i in range(1, n + 1):
            # Вычисляем диагональный элемент L[i,i]
            sum_val = 0.0
            max_j = max(1, i - k)
            
            for j in range(max_j, i):
                sum_val += L[i, j] ** 2
            
            # Проверка на положительную определенность
            diag_val = matrix[i, i] - sum_val
            if diag_val <= 1e-10:  # Allow for small numerical errors
                if diag_val < 0:
                    # Check if it's just a numerical precision issue (very close to zero)
                    if diag_val >= -1e-12:
                        # Treat as zero and set to a small positive value
                        diag_val = 1e-10
                    else:
                        raise ValueError(f"Matrix is not positive definite at row {i}, diagonal value: {diag_val}")
                else:
                    # For very small positive values, set to small positive value
                    diag_val = 1e-10
            
            L[i, i] = np.sqrt(diag_val)
            
            # Вычисляем недиагональные элементы в строке i
            for m in range(i + 1, min(n, i + k) + 1):
                sum_val = 0.0
                max_j = max(1, i - k, m - k)
                
                for j in range(max_j, i):
                    sum_val += L[i, j] * L[m, j]
                
                if abs(L[i, i]) < 1e-12:
                    raise ZeroDivisionError(f"Zero diagonal element L[{i},{i}]")
                
                L[m, i] = (matrix[m, i] - sum_val) / L[i, i]
        
        return L

    @staticmethod
    def is_symmetric(matrix, tolerance=1e-10):
        """
        Check if the tape matrix is symmetric
        """
        n = matrix.size
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if abs(matrix[i, j] - matrix[j, i]) > tolerance:
                    return False
        return True

    @staticmethod
    def solve_cholesky_band(matrix, exact_x):
        """
        Solve Ax = b using Cholesky decomposition for band matrix
        Steps:
        1. A = L * L^T (Cholesky decomposition)
        2. Solve L * y = b (forward substitution)
        3. Solve L^T * x = y (backward substitution)
        """
        # Check if matrix is symmetric (required for Cholesky decomposition)
        if not ExperimentUtils.is_symmetric(matrix):
            print("Warning: Matrix is not symmetric. Cholesky decomposition requires a symmetric matrix.")
            # Attempt to make it symmetric by averaging with its transpose
            # Only operate within the band to avoid out-of-bounds errors
            for i in range(1, matrix.size + 1):
                for j in range(i + 1, min(matrix.size + 1, i + matrix.k + 1)):  # Only within band
                    if abs(j - i) <= matrix.k:  # Within bandwidth
                        avg_val = (matrix[i, j] + matrix[j, i]) / 2
                        matrix[i, j] = avg_val
                        matrix[j, i] = avg_val

        # Generate a known right-hand side vector b = A * exact_x
        b = matrix @ exact_x
        n = matrix.size
        
        # 1. Cholesky decomposition
        try:
            L = ExperimentUtils.band_cholesky(matrix)
        except ValueError as e:
            print(f"Cholesky decomposition failed: {e}")
            return None
        
        # 2. Forward substitution: L * y = b
        y = v.Vector(size=n)
        
        for i in range(1, n + 1):
            sum_val = 0.0
            max_j = max(1, i - L.k)
            
            for j in range(max_j, i):
                sum_val += L[i, j] * y[j]
            
            if abs(L[i, i]) < 1e-12:
                raise ZeroDivisionError(f"Zero diagonal element L[{i},{i}] in forward substitution")
            
            y[i] = (b[i] - sum_val) / L[i, i]
        
        # 3. Backward substitution: L^T * x = y
        approximate_x = v.Vector(size=n)
        
        for i in range(n, 0, -1):
            sum_val = 0.0
            min_j = min(n, i + L.k)
            
            for j in range(i + 1, min_j + 1):
                # Note: L^T[i, j] = L[j, i]
                sum_val += L[j, i] * approximate_x[j]
            
            if abs(L[i, i]) < 1e-12:
                raise ZeroDivisionError(f"Zero diagonal element L[{i},{i}] in backward substitution")
            
            approximate_x[i] = (y[i] - sum_val) / L[i, i]
        
        return approximate_x
    
    @staticmethod
    def calculate_error(exact_x, approximate_x):
        absolute_errors = v.Vector(size=exact_x.size)
        relative_errors = v.Vector(size=exact_x.size)

        for i in range(1, exact_x.size + 1):
            abs_err = abs(exact_x[i] - approximate_x[i])
            absolute_errors[i] = abs_err

            # Защита от деления на ноль
            if abs(exact_x[i]) > np.finfo(float).eps:
                relative_errors[i] = abs_err / abs(exact_x[i])
            else:
                relative_errors[i] = abs_err  # если точное значение близко к 0

        absolute_error = absolute_errors.norm()
        relative_error = relative_errors.norm()

        return absolute_error, relative_error