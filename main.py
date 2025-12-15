import numpy as np

from mymath import Vector as v
from mymath import TapeMatrix as tm
import experiment.ExperimentUtils as utils

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


if __name__ == "__main__":
    print_test()