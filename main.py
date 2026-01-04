import numpy as np
import pandas as pd
import os
from mymath import Vector as v
from mymath import TapeMatrix as tm
import experiment.ExperimentUtils as utils
import experiment.DataGenerator as dg

def main_experiment(exp_count: int, low=1.0, high=10.0, condition_type='random'):
    """Основной эксперимент для анализа ошибок решения систем с ленточными матрицами"""
    data_dict = {
        'system size': [],  # размер системы
        'bandwidth': [],    # ширина ленты
        'filling range': [],  # диапазон заполнения
        'absolute error': [],  # абсолютная ошибка
        'relative error': [],  # относительная ошибка
    }

    for size in np.logspace(2, 10, base=2, num=9).astype(int):
        
        for bandwidth in [size // 10, size * 4 // 10, size * 8 // 10]:

            if bandwidth % 2 == 0:
                bandwidth -= 1 # Сделать нечетной
            if bandwidth < 3:
                bandwidth = 3  # Минимальная нечетная ширина ленты
            
            for _ in range(exp_count):
                matrix, exact_x = dg.DataGenerator.generate_data(size, bandwidth, low, high, condition_type)
                
                approximate_x = utils.ExperimentUtils.solve_cholesky_band(matrix, exact_x)
                
                if approximate_x is not None:
                    absolute_error, relative_error = utils.ExperimentUtils.calculate_error(exact_x, approximate_x)
                    
                    data_dict['system size'].append(size)  # размер системы
                    data_dict['bandwidth'].append(bandwidth) # ширина ленты
                    data_dict['filling range'].append([low, high])  # диапазон заполнения
                    data_dict['absolute error'].append(absolute_error)  # абсолютная ошибка
                    data_dict['relative error'].append(relative_error)  # относительная ошибка
    
    return pd.DataFrame(data_dict)

def print_test():
    """Тестирование решения системы с ленточной матрицей"""
    matrix = tm.TapeMatrix(5, 3)  # Инициализировать с правильным значением ширины ленты (2*k+1 = 2*1+1 = 3)
    matrix.read_from_file("data/text.txt")
    
    exact_x = v.Vector(np.array([1.0, -1.0, 2.0, -1.0, 1.0]), size=5)

    approximate_x = utils.ExperimentUtils.solve_cholesky_band(matrix, exact_x)
    
    if approximate_x is not None:
        absolute_error, relative_error = utils.ExperimentUtils.calculate_error(exact_x, approximate_x)

        matrix_to_print = matrix.to_full_matrix()
        print(matrix_to_print)
        print(f"Точное x:        {exact_x}")
        print(f"Вычисленное x:   {approximate_x}")
        print(f"absolute_error:  {absolute_error:10e}, relative_error: {relative_error:10e}")
    else:
        print("Не удалось решить систему.")

def main():
    ranges = [(1.0, 10.0), (1.0, 100.0), (1.0, 1000.0)]

    configurations = [
        ('random', 'results_rand.csv'),
        ('dominant', 'results_dominant.csv'),
    ]

    for condition_type, filename in configurations:
        dataframes = []
        for low, high in ranges:
            print(f"Запуск эксперимента: condition_type={condition_type}, range=[{low}, {high}]")
            df = main_experiment(1, low, high, condition_type=condition_type)
            dataframes.append(df)

        final_df = pd.concat(dataframes, ignore_index=True)
        final_df.to_csv(f'data/{filename}', index=False)

    print("Все эксперименты завершены. Результаты сохранены в папку 'data'.")


if __name__ == '__main__':
    main()