import numpy as np
import mymath.TapeMatrix as tm
import mymath.Vector as v

class DataGenerator:
    @staticmethod
    def generate_data(size, bandwith, low=0.0, high=10.0, condition_type="random"):
       matrix = tm.TapeMatrix(size, bandwith) 
       matrix.fill_random(low, high, condition_type)

       exact_x = v.Vector(size=size)
       exact_x.fill_random(low, high)

       return matrix, exact_x
