import numpy as np

class Vector:
    def __init__(self, data=None, size=0):
        if data is not None:
            self.data = np.array(data, dtype=float)
            self.size = len(data)
        else:
            self.data = np.zeros(size)
            self.size = size


    def __add__(self, other):
        if self.size != other.size:
            raise ValueError("Vector sizes don't match for addition")
        return Vector(self.data + other.data)


    def __sub__(self, other):
        if self.size != other.size:
            raise ValueError("Vector sizes don't match for subtraction")
        return Vector(self.data - other.data)


    def dot(self, other):
        if self.size != other.size:
            raise ValueError("Vector sizes don't match for dot product")
        return np.dot(self.data, other.data)


    def norm(self):
        return np.max(np.abs(self.data))


    def fill_random(self, low=-5.0, high=5.0):
        self.data = np.random.uniform(low, high, self.size)
        return self


    def read_from_console(self):
        print(f"Enter {self.size} vector elements:")
        for i in range(self.size):
            self.data[i] = float(input(f"Element {i + 1}: "))
        return self


    def read_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                data = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        data.extend(map(float, line.split()))

                self.data = np.array(data)
                self.size = len(data)
                return self
        except FileNotFoundError:
            print(f"File {filename} not found")
            return None


    def write_to_file(self, filename):
        """Write vector to file"""
        with open(filename, 'w') as f:
            f.write("# Vector\n")
            for i, val in enumerate(self.data):
                f.write(f"{val:.6f}")
                if i < self.size - 1:
                    f.write(" ")
            f.write("\n")
        return self


    def __str__(self):
        res = ''
        for val in self.data:
            res += f"{val:.15f} "
        return res


    def __getitem__(self, index):
        """Access elements by index (starting from 1)"""
        if index < 1 or index > self.size:
            raise IndexError("Index out of range")
        return self.data[index - 1]


    def __setitem__(self, index, value):
        """Set element value (indexing from 1)"""
        if index < 1 or index > self.size:
            raise IndexError("Index out of range")
        self.data[index - 1] = value


    def __len__(self):
        """Return vector size"""
        return self.size
