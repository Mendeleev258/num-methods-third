import random
import numpy as np
from . import Vector as v

class TapeMatrix:
    def __init__(self, size=0, bandwidth=3):
        """
        Initialize tape matrix
        
        Parameters:
            size: matrix dimension (n x n)
            bandwidth: total width of band (must be odd number, e.g., 3, 5, 7...)
                     bandwidth = 2k + 1, where k is number of diagonals above/below main diagonal
        """
        if bandwidth % 2 == 0:
            raise ValueError("Bandwidth must be an odd number (3, 5, 7...)")
        
        self.size = size
        self.bandwidth = bandwidth
        self.k = (bandwidth - 1) // 2  # number of diagonals above/below main
        
        # Store diagonals in a list
        # diagonals[0] = main diagonal (b)
        # diagonals[1] = first upper diagonal (c)
        # diagonals[-1] = first lower diagonal (a)
        # etc.
        self.diagonals = []
        for i in range(bandwidth):
            # Determine which diagonal this is
            diag_index = i - self.k  # -k, -k+1, ..., 0, ..., k-1, k
            diag_len = self._get_diagonal_length(diag_index)
            self.diagonals.append(v.Vector(size=diag_len))
    
    def _get_diagonal_index(self, row, col):
        """
        Convert matrix indices (row, col) to diagonal index
        Returns (diag_offset, position) where:
            diag_offset: which diagonal (0 = main, positive = upper, negative = lower)
            position: position along that diagonal
        """
        offset = col - row
        if abs(offset) > self.k:
            return None, None  # Outside band
        
        # Convert offset to diagonal index in list
        diag_idx = offset + self.k
        
        # Position along diagonal
        if offset <= 0:  # Main or lower diagonal
            position = col
        else:  # Upper diagonal
            position = row
        
        return diag_idx, position
    
    def _get_diagonal_length(self, offset):
        """
        Get length of diagonal with given offset
        offset: 0 for main, positive for upper, negative for lower
        """
        if offset == 0:  # Main diagonal
            return self.size
        elif offset > 0:  # Upper diagonal
            return self.size - offset
        else:  # Lower diagonal
            return self.size + offset
    
    def __setitem__(self, indices, value):
        """
        Set matrix element A[i,j] = value
        Indices start from 1
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
        Get matrix element A[i,j]
        Indices start from 1
        """
        row, col = indices
        if row < 1 or row > self.size or col < 1 or col > self.size:
            raise IndexError(f"Indices ({row},{col}) out of range for matrix {self.size}x{self.size}")
        
        diag_idx, pos = self._get_diagonal_index(row, col)
        if diag_idx is None:  # Outside band
            return 0.0
        
        return self.diagonals[diag_idx][pos]
    
    def __add__(self, other):
        """Tape matrix addition"""
        if self.size != other.size or self.bandwidth != other.bandwidth:
            raise ValueError("Matrix sizes or bandwidths don't match for addition")
        
        result = TapeMatrix(self.size, self.bandwidth)
        for i in range(self.bandwidth):
            # Assuming both matrices store diagonals in same order
            # Simple element-wise addition
            for j in range(1, len(self.diagonals[i].data) + 1):
                result.diagonals[i][j] = self.diagonals[i][j] + other.diagonals[i][j]
        
        return result
    
    def __sub__(self, other):
        """Tape matrix subtraction"""
        if self.size != other.size or self.bandwidth != other.bandwidth:
            raise ValueError("Matrix sizes or bandwidths don't match for subtraction")
        
        result = TapeMatrix(self.size, self.bandwidth)
        for i in range(self.bandwidth):
            for j in range(1, len(self.diagonals[i].data) + 1):
                result.diagonals[i][j] = self.diagonals[i][j] - other.diagonals[i][j]
        
        return result
    
    def multiply_vector(self, vector):
        """Multiply matrix by vector"""
        if self.size != len(vector):
            raise ValueError("Matrix and vector sizes don't match")
        
        result = v.Vector(size=self.size)
        
        for i in range(1, self.size + 1):
            sum_val = 0.0
            # Only check columns within band
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                sum_val += self[i, j] * vector[j]
            
            result[i] = sum_val
        
        return result
    
    def __matmul__(self, other):
        """Overload @ operator for vector multiplication"""
        return self.multiply_vector(other)
    
    def fill_random(self, low=0.0, high=10.0, condition_type='random'):
        """
        Fill matrix with random numbers
        
        Parameters:
            low, high: range for random values
            condition_type: 'random', 'dominant', or 'ill'
        """
        if condition_type == 'dominant':
            self._fill_random_diagonally_dominant(low, high)
        elif condition_type == 'ill':
            self._fill_random_ill_conditioned(low, high)
        else:  # 'random'
            self._fill_random_regular(low, high)
        
        return self
    
    def _fill_random_regular(self, low, high):
        """Regular random generation"""
        for diag in self.diagonals:
            diag.fill_random(low, high)
    
    def _fill_random_diagonally_dominant(self, low, high):
        """Create diagonally dominant matrix"""
        # First fill all diagonals with random values
        self._fill_random_regular(low, high)
        
        # Make main diagonal dominant
        main_diag_idx = self.k  # Main diagonal is at center
        for i in range(1, self.size + 1):
            # Calculate sum of absolute values in row i (within band)
            row_sum = 0.0
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                if j != i:  # Skip diagonal element
                    row_sum += abs(self[i, j])
            
            # Make diagonal element larger than row sum
            self[i, i] = row_sum + random.uniform(1.0, 5.0)
    
    def _fill_random_ill_conditioned(self, low, high):
        """Create poorly conditioned matrix"""
        # Fill with small random values
        for diag in self.diagonals:
            diag.fill_random(low * 0.1, high * 0.1)
        
        # Make some rows nearly linearly dependent
        if self.size > 2:
            # Make row 2 almost equal to row 1
            for j in range(1, min(3, self.size) + 1):
                self[2, j] = self[1, j] * (1 + random.uniform(-0.01, 0.01))
    
    def read_from_console(self):
        """Read matrix from console"""
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
        """Read matrix from file in specified format
        
        File format:
        First line: n k   (size and k = number of diagonals above/below main)
        Next lines: diagonals data, starting from lowest diagonal to highest
        
        Example for k=1 (tridiagonal):
        4 1
        3.0 8.0 4.0   # lower diagonal (a), length = n, but a[1] should be 0
        5.0 2.0 9.0 1.0   # main diagonal (b), length = n
        1.0 6.0 3.0    # upper diagonal (c), length = n, but c[n] should be 0
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
                    raise ValueError(f"Expected {bandwidth + 1} lines, got {len(lines)}. "
                                f"For k={k}, bandwidth={bandwidth}, need {bandwidth} diagonal lines.")
                
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
                        raise ValueError(f"Diagonal {diag_idx} (offset={diag_idx - k}): "
                                    f"expected {expected_len} values, got {len(values_str)}")
                    
                    # Convert to float and store
                    values = list(map(float, values_str))
                    for i in range(len(values)):
                        self.diagonals[diag_idx][i + 1] = values[i]  # Vector indexing starts from 1
                
                return self
        
        except FileNotFoundError:
            print(f"File {filename} not found")
            return None
        except ValueError as e:
            print(f"Error reading file: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def write_to_console(self):
        print(f"\nTape Matrix {self.size}x{self.size}, k={self.k}")
        print(f"{self.size} {self.k}")
        
        for diag_idx in range(self.bandwidth):
            diag = self.diagonals[diag_idx]
            values = []
            for i in range(1, len(diag.data) + 1):
                values.append(diag[i])
            
            # Format with consistent spacing
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
        """Convert to full matrix (for verification)"""
        full_matrix = np.zeros((self.size, self.size))
        
        for i in range(1, self.size + 1):
            min_col = max(1, i - self.k)
            max_col = min(self.size, i + self.k)
            
            for j in range(min_col, max_col + 1):
                full_matrix[i-1, j-1] = self[i, j]
        
        return full_matrix
    
    def get_cond(self):
        """Calculate condition number"""
        full_matrix = self.to_full_matrix()
        return np.linalg.cond(full_matrix)
    
    def __str__(self):
        """String representation - show diagonals"""
        result = f"TapeMatrix {self.size}x{self.size}, bandwidth={self.bandwidth}\n"
        result += "Diagonals:\n"
        
        for idx, diag in enumerate(self.diagonals):
            offset = idx - self.k
            result += f"  Offset {offset:3}: {diag}\n"
        
        return result
    
    def get_bandwidth(self):
        """Get matrix bandwidth"""
        return self.bandwidth
    
    def get_k(self):
        """Get k parameter (number of diagonals above/below main)"""
        return self.k