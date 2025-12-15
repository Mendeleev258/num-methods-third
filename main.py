import numpy as np
import pandas as pd
import os
from mymath import Vector as v
from mymath import TapeMatrix as tm
import experiment.ExperimentUtils as utils
import experiment.DataGenerator as dg

def main_experiment(exp_count: int, low=1.0, high=10.0, condition_type='random'):
    data_dict = {
        'system size': [],
        'filling range': [],
        'absolute error': [],
        'relative error': [],
    }

    for size in np.logspace(1, 6, base=2).astype(int):
        # Ensure bandwidth is odd and appropriate for the matrix size
        bandwidth = min(size, 5)
        if bandwidth % 2 == 0:
            bandwidth -= 1  # Make it odd
        if bandwidth < 3:
            bandwidth = 3  # Minimum odd bandwidth
        
        for _ in range(exp_count):
            matrix, exact_x = dg.DataGenerator.generate_data(size, bandwidth, low, high, condition_type)
            
            approximate_x = utils.ExperimentUtils.solve_cholesky_band(matrix, exact_x)
            
            if approximate_x is not None:
                absolute_error, relative_error = utils.ExperimentUtils.calculate_error(exact_x, approximate_x)
                
                data_dict['system size'].append(size)
                data_dict['filling range'].append([low, high])
                data_dict['absolute error'].append(absolute_error)
                data_dict['relative error'].append(relative_error)

    return pd.DataFrame(data_dict)

def print_test():
    matrix = tm.TapeMatrix(5, 3)  # Initialize with correct bandwidth value (2*k+1 = 2*1+1 = 3)
    matrix.read_from_file("data/text.txt")
    
    exact_x = v.Vector(np.array([1.0, -1.0, 2.0, -1.0, 1.0]), size=5)

    approximate_x = utils.ExperimentUtils.solve_cholesky_band(matrix, exact_x)
    
    if approximate_x is not None:
        absolute_error, relative_error = utils.ExperimentUtils.calculate_error(exact_x, approximate_x)

        matrix_to_print = matrix.to_full_matrix()
        print(matrix_to_print)
        print(f"Exact x:         {exact_x}")
        print(f"Calculated x:    {approximate_x}")
        print(f"absolute_error:  {absolute_error:10e}, relative_error: {relative_error:10e}")
    else:
        print("Failed to solve the system.")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    ranges = [(1.0, 1.0), (1.0, 10.0), (1.0, 100.0), (1.0, 1000.0)]

    configurations = [
        ('random', 'results_rand.csv'),
        ('dominant', 'results_dominant.csv'),
    ]

    for condition_type, filename in configurations:
        dataframes = []
        for low, high in ranges:
            print(f"Running experiment: condition_type={condition_type}, range=[{low}, {high}]")
            df = main_experiment(3, low, high, condition_type=condition_type)
            dataframes.append(df)

        final_df = pd.concat(dataframes, ignore_index=True)
        final_df.to_csv(f'data/{filename}', index=False)

    print("All experiments completed. Results saved to the 'results' folder.")