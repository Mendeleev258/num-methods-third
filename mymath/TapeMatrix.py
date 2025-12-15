import random
import numpy as np
from . import Vector as v

class TapeMatrix:
    def __init__(self, size=0, bandwidth=3):
        """
        Инициализация ленточной матрицы
        
        Параметры:
            size: размерность матрицы (n x n)
            bandwidth: полная ширина ленты (должна быть нечетным числом, например, 3, 5, 7...)
                      bandwidth = 2k + 1, где k - количество диагоналей над/под главной диагональю
        """
        if bandwidth % 2 == 0:
            raise ValueError("Ширина ленты должна быть нечетным числом (3, 5, 7...)")
        
        self.size = size
        self.bandwidth = bandwidth
        self.k = (bandwidth - 1) // 2  # количество диагоналей над/под главной
        
        # Хранение диагоналей в списке
        # diagonals[0] = главная диагональ (b)
        # diagonals[1] = первая верхняя диагональ (c)
        # diagonals[-1] = первая нижняя диагональ (a)
        # и т.д.
        self.diagonals = []
        for i in range(bandwidth):
            # Определяем, какая это диагональ
            diag_index = i - self.k  # -k, -k+1, ..., 0, ..., k-1, k
            diag_len = self._get_diagonal_length(diag_index)
            self.diagonals.append(v.Vector(size=diag_len))
    
    def _get_diagonal_index(self, row, col):
        """
        Преобразование индексов матрицы (row, col) в индекс диагонали
        Возвращает (diag_offset, position), где:
            diag_offset: какая диагональ (0 = главная, положительная = верхняя, отрицательная = нижняя)
            position: позиция вдоль этой диагонали
        """
        offset = col - row
        if abs(offset) > self.k:
            return None, None  # Вне ленты
        
        # Преобразуем смещение в индекс диагонали в списке
        diag_idx = offset + self.k
        
        # Position along diagonal
        if offset <= 0:  # Main or lower diagonal
            position = col
        else:  # Верхняя диагональ
            position = row
        
        return diag_idx, position
    
    def _get_diagonal_length(self, offset):
        """
        Получение длины диагонали с заданным смещением
        offset: 0 для главной, положительное для верхней, отрицательное для нижней
        """
        if offset == 0:  # Main diagonal
            return self.size
        elif offset > 0:  # Upper diagonal
            return self.size - offset
        else:  # Lower diagonal
            return self.size + offset
    
    def __setitem__(self, indices, value):
        """
        Установка элемента матрицы A[i,j] = value
        Индексы начинаются с 1
        """
        row, col = indices
        if row < 1 or row > self.size or col < 1 or col > self.size:
            raise IndexError(f"Indices ({row},{col}) out of range for matrix {self.size}x{self.size}")
        
        diag_idx, pos = self._get_diagonal_index(row, col)
        if diag_idx is None:  # Outside band
            raise ValueError(f"Element ({row},{col}) is outside matrix band (k={self.k})")
        
        self.diagonals[diag_idx][pos] = value
    
    def __getitem__(self, indices):
        """
        Получение элемента матрицы A[i,j]
        Индексы начинаются с 1
        """
        row, col = indices
        if row < 1 or row > self.size or col < 1 or col > self.size:
            raise IndexError(f"Indices ({row},{col}) out of range for matrix {self.size}x{self.size}")
        
        diag_idx, pos = self._get_diagonal_index(row, col)
        if diag_idx is None:  # Outside band
            return 0.0
        
        return self.diagonals[diag_idx][pos]
    
    def __add__(self, other):
        """Сложение ленточных матриц"""
        if self.size != other.size or self.bandwidth != other.bandwidth:
            raise ValueError("Размеры матриц или ширины ленты не совпадают для сложения")
        
        result = TapeMatrix(self.size, self.bandwidth)
        for i in range(self.bandwidth):
            for j in range(1, len(self.diagonals[i].data) + 1):
                result.diagonals[i][j] = self.diagonals[i][j] + other.diagonals[i][j]
        
        return result
    
    def __sub__(self, other):
        """Вычитание ленточных матриц"""
        if self.size != other.size or self.bandwidth != other.bandwidth:
            raise ValueError("Размеры матриц или ширины ленты не совпадают для вычитания")
        
        result = TapeMatrix(self.size, self.bandwidth)
        for i in range(self.bandwidth):
            for j in range(1, len(self.diagonals[i].data) + 1):
                result.diagonals[i][j] = self.diagonals[i][j] - other.diagonals[i][j]
        
        return result
    
    def multiply_vector(self, vector):
        """Умножение матрицы на вектор"""
        if self.size != len(vector):
            raise ValueError("Размеры матрицы и вектора не совпадают")
        
        result = v.Vector(size=self.size)
        
        for i in range(1, self.size + 1):
            sum_val = 0.0
            # Проверяем только столбцы внутри ленты
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                sum_val += self[i, j] * vector[j]
            
            result[i] = sum_val
        
        return result
    
    def __matmul__(self, other):
        """Перегрузка оператора @ для умножения на вектор"""
        return self.multiply_vector(other)
    
    def fill_random(self, low=0.0, high=10.0, condition_type='random'):
        """
        Заполнение матрицы случайными числами
        
        Параметры:
            low, high: диапазон для случайных значений
            condition_type: 'random', 'dominant'
        """
        if condition_type == 'dominant':
            self._fill_random_diagonally_dominant(low, high)
        elif condition_type == 'random':
            self._fill_random_positive_definite(low, high)
        else:  # fallback to regular random
            self._fill_random_regular(low, high)
        
        return self
    
    def _fill_random_regular(self, low, high):
        """Обычная случайная генерация с симметричной матрицей"""
        # Fill main diagonal randomly
        self.diagonals[self.k].fill_random(low, high)
        
        # Для внедиагональных элементов генерируем случайные значения для верхних диагоналей
        # и копируем их в соответствующие нижние диагонали для сохранения симметрии
        for i in range(self.k):
            # Генерируем случайные значения для верхней диагонали
            upper_diag_idx = self.k + i + 1  # Индекс верхней диагонали
            lower_diag_idx = self.k - i - 1  # Индекс соответствующей нижней диагонали
            
            # Заполняем верхнюю диагональ случайными значениями
            self.diagonals[upper_diag_idx].fill_random(low, high)
            
            # Копируем те же значения в нижнюю диагональ для сохранения симметрии
            for j in range(1, len(self.diagonals[lower_diag_idx].data) + 1):
                self.diagonals[lower_diag_idx][j] = self.diagonals[upper_diag_idx][j]

    def _fill_random_positive_definite(self, low, high):
        """Создание симметричной положительно определенной матрицы через A^T*A"""
        # Сначала создаем симметричную матрицу
        self._fill_random_regular(low, high)
        
        # Преобразуем в полную матрицу для вычисления A^T * A
        full_matrix = self.to_full_matrix()
        
        # Генерируем случайную матрицу и вычисляем B = A^T * A, которая гарантированно положительно определена
        # Но для сохранения структуры ленточной матрицы, будем использовать другой подход:
        # Добавим к диагонали достаточно большое значение, чтобы обеспечить положительную определенность
        
        # Получаем собственные значения текущей матрицы
        eigenvalues = np.linalg.eigvals(full_matrix)
        min_eigenvalue = np.min(eigenvalues.real)  # берем вещественную часть
        
        # Если минимальное собственное значение отрицательно или близко к нулю,
        # добавляем положительную константу к диагонали
        shift = 0.0
        if min_eigenvalue <= 0:
            shift = abs(min_eigenvalue) + 1.0  # добавляем запас для гарантии положительности
        
        # Добавляем сдвиг к диагонали
        for i in range(1, self.size + 1):
            old_value = self[i, i]
            self[i, i] = old_value + shift

    def _fill_random_diagonally_dominant(self, low, high):
        """Создание симметричной диагонально-доминантной матрицы"""
        # Сначала заполняем матрицу, чтобы она была симметричной
        # Заполняем главную диагональ случайными значениями
        self.diagonals[self.k].fill_random(low, high)
        
        # Для внедиагональных элементов генерируем случайные значения для верхних диагоналей
        # и копируем их в соответствующие нижние диагонали для сохранения симметрии
        for i in range(self.k):
            # Генерируем случайные значения для верхней диагонали
            upper_diag_idx = self.k + i + 1  # Индекс верхней диагонали
            lower_diag_idx = self.k - i - 1  # Индекс соответствующей нижней диагонали
            
            # Заполняем верхнюю диагональ случайными значениями
            self.diagonals[upper_diag_idx].fill_random(low, high)
            
            # Копируем те же значения в нижнюю диагональ для сохранения симметрии
            for j in range(1, len(self.diagonals[lower_diag_idx].data) + 1):
                self.diagonals[lower_diag_idx][j] = self.diagonals[upper_diag_idx][j]
        
        # Делаем главную диагональ доминирующей, сохраняя симметрию
        main_diag_idx = self.k  # Главная диагональ находится в центре
        for i in range(1, self.size + 1):
            # Вычисляем сумму абсолютных значений в строке i (внутри ленты)
            row_sum = 0.0
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
             
            for j in range(min_col, max_col + 1):
                if j != i:  # Пропускаем диагональный элемент
                    row_sum += abs(self[i, j])
             
            # Делаем диагональный элемент больше суммы строки
            self[i, i] = row_sum + random.uniform(1.0, 5.0)
    
    
    def read_from_console(self):
        """Чтение матрицы из консоли"""
        print(f"Enter elements of tape matrix {self.size}x{self.size} (bandwidth={self.bandwidth}):")
        
        for i in range(1, self.size + 1):
            print(f"\nRow {i}:")
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                value = float(input(f"  A[{i},{j}]: "))
                self[i, j] = value
        
        return self
    
    def read_from_file(self, filename):
        """Чтение матрицы из файла в заданном формате
        
        Формат файла:
        Первая строка: n k   (размер и k = количество диагоналей над/под главной)
        Следующие строки: данные диагоналей, начиная с самой нижней диагонали до самой верхней
        
        Пример для k=1 (трехдиагональная):
        4 1
        3.0 8.0 4.0   # нижняя диагональ (a), длина = n, но a[1] должен быть 0
        5.0 2.0 9.0 1.0   # главная диагональ (b), длина = n
        1.0 6.0 3.0    # верхняя диагональ (c), длина = n, но c[n] должен быть 0
        """
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
                if len(lines) == 0:
                    raise ValueError("Empty file")
                
                # Parse first line: n k
                first_line_parts = lines[0].split()
                if len(first_line_parts) != 2:
                    raise ValueError("First line must contain exactly two numbers: n k")
                
                n = int(first_line_parts[0])
                k = int(first_line_parts[1])
                bandwidth = 2 * k + 1
                
                # Check we have correct number of diagonal lines
                if len(lines) != bandwidth + 1:  # First line + bandwidth diagonals
                    raise ValueError(f"Ожидалось {bandwidth + 1} строк, получено {len(lines)}. "
                                f"Для k={k}, ширина ленты={bandwidth}, нужно {bandwidth} строк диагоналей.")
                
                # Resize matrix
                self.size = n
                self.bandwidth = bandwidth
                self.k = k
                
                # Recreate diagonals with correct lengths
                self.diagonals = []
                for diag_idx in range(bandwidth):
                    diag_offset = diag_idx - k  # -k, -k+1, ..., 0, ..., k
                    diag_len = self._get_diagonal_length(diag_offset)
                    self.diagonals.append(v.Vector(size=diag_len))
                
                # Read diagonal data
                for diag_idx in range(bandwidth):
                    line_idx = diag_idx + 1  # +1 to skip first line
                    values_str = lines[line_idx].split()
                    
                    if len(values_str) != len(self.diagonals[diag_idx].data):
                        expected_len = len(self.diagonals[diag_idx].data)
                        raise ValueError(f"Диагональ {diag_idx} (смещение={diag_idx - k}): "
                                    f"ожидалось {expected_len} значений, получено {len(values_str)}")
                    
                    # Convert to float and store
                    values = list(map(float, values_str))
                    for i in range(len(values)):
                        self.diagonals[diag_idx][i + 1] = values[i]  # Индексация вектора начинается с 1
                
                return self
        
        except FileNotFoundError:
            print(f"Файл {filename} не найден")
            return None
        except ValueError as e:
            print(f"Ошибка чтения файла: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            return None
    
    def write_to_console(self):
        print(f"\nЛенточная матрица {self.size}x{self.size}, k={self.k}")
        print(f"{self.size} {self.k}")
        
        for diag_idx in range(self.bandwidth):
            diag = self.diagonals[diag_idx]
            values = []
            for i in range(1, len(diag.data) + 1):
                values.append(diag[i])
            
            # Форматирование с постоянным интервалом
            line = " ".join(f"{val:10.6f}" for val in values)
            print(line)
        
        return self
    
    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            # Write header: n k
            f.write(f"{self.size} {self.k}\n")
            
            # Write each diagonal on separate line
            for diag_idx in range(self.bandwidth):
                diag = self.diagonals[diag_idx]
                values = []
                for i in range(1, len(diag.data) + 1):
                    values.append(diag[i])
                
                # Format values
                line = " ".join(f"{val:.6f}" for val in values)
                f.write(line + "\n")
        
        return self
    
    def to_full_matrix(self):
        """Преобразование в полную матрицу (для проверки)"""
        full_matrix = np.zeros((self.size, self.size))
        
        for i in range(1, self.size + 1):
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                full_matrix[i-1, j-1] = self[i, j]
        
        return full_matrix
    
    def get_cond(self):
        """Вычисление числа обусловленности"""
        full_matrix = self.to_full_matrix()
        return np.linalg.cond(full_matrix)
    
    def __str__(self):
        """Строковое представление - показать диагонали"""
        result = f"TapeMatrix {self.size}x{self.size}, bandwidth={self.bandwidth}\n"
        result += "Diagonals:\n"
        
        for idx, diag in enumerate(self.diagonals):
            offset = idx - self.k
            result += f"  Offset {offset:3}: {diag}\n"
        
        return result
    
    def get_bandwidth(self):
        """Получение ширины ленты матрицы"""
        return self.bandwidth
    
    def get_k(self):
        """Получение параметра k (количество диагоналей над/под главной)"""
        return self.k