import numpy as np

# Create a sample numpy array
arr = np.array([[1, 8, 3], [4, 2, 6], [7, 9, 5]])

# Create a boolean array based on a condition
condition_array = arr > 5

# Create another array with the same shape as the original array
another_array = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

# Replace values in another_array based on the condition_array
another_array[condition_array] = 999  # Replace with any desired value

print("Original array:")
print(arr)

print("\nBoolean array based on condition (arr > 5):")
print(condition_array)

print("\nAnother array after replacing values based on the boolean array:")
print(another_array)