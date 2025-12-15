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
            if diag_val <= 1e-10:  # Учитываем небольшие численные ошибки
                if diag_val < 0:
                    # Проверяем, является ли это просто проблемой точности (очень близко к нулю)
                    if diag_val >= -1e-12:
                        # Считаем нулевым и устанавливаем малое положительное значение
                        diag_val = 1e-10
                    else:
                        raise ValueError(f"Матрица не является положительно определенной в строке {i}, диагональное значение: {diag_val}")
                else:
                    # Для очень малых положительных значений устанавливаем малое положительное значение
                    diag_val = 1e-10
            
            L[i, i] = np.sqrt(diag_val)
            
            # Вычисляем недиагональные элементы в строке i
            for m in range(i + 1, min(n, i + k) + 1):
                sum_val = 0.0
                max_j = max(1, i - k, m - k)
                
                for j in range(max_j, i):
                    sum_val += L[i, j] * L[m, j]
                
                if abs(L[i, i]) < 1e-12:
                    raise ZeroDivisionError(f"Нулевой диагональный элемент L[{i},{i}]")
                
                L[m, i] = (matrix[m, i] - sum_val) / L[i, i]
        
        return L

    @staticmethod
    def solve_cholesky_band(matrix, exact_x):
        """
        Решает Ax = b с использованием разложения Холецкого для ленточной матрицы
        Шаги:
        1. A = L * L^T (разложение Холецкого)
        2. Решаем L * y = b (прямая подстановка)
        3. Решаем L^T * x = y (обратная подстановка)
        """

        # Генерируем известный вектор правой части b = A * exact_x
        b = matrix @ exact_x
        n = matrix.size
        
        # 1. Разложение Холецкого
        try:
            L = ExperimentUtils.band_cholesky(matrix)
        except ValueError as e:
            print(f"Разложение Холецкого не выполнено: {e}")
            return None
        
        # 2. Прямая подстановка: L * y = b
        y = v.Vector(size=n)
        
        for i in range(1, n + 1):
            sum_val = 0.0
            max_j = max(1, i - L.k)
            
            for j in range(max_j, i):
                sum_val += L[i, j] * y[j]
            
            if abs(L[i, i]) < 1e-12:
                raise ZeroDivisionError(f"Zero diagonal element L[{i},{i}] in forward substitution")
            
            y[i] = (b[i] - sum_val) / L[i, i]
        
        # 3. Обратная подстановка: L^T * x = y
        approximate_x = v.Vector(size=n)
        
        for i in range(n, 0, -1):
            sum_val = 0.0
            min_j = min(n, i + L.k)
            
            for j in range(i + 1, min_j + 1):
                # Примечание: L^T[i, j] = L[j, i]
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