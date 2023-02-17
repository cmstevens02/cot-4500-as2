# Christine Stevens
import numpy as np
from numpy.linalg import inv

np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Neville's method, find 2nd degree interpolating value for f(3.7)

def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((3, 3))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # the end of the first loop are how many columns you have...
    num_of_points = len(x_points)

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        for j in range(1, i + 1 ):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]

            denominator = x_points[i] - x_points[i-j]

            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator
            matrix[i][j] = coefficient

    
    return matrix

def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            matrix[i][j] = operation

    return matrix




def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index - 1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    
    # final result
    return reoccuring_px_result



def apply_div_diff(matrix):

    size = len(matrix)
    for i in range(2, size):
        for j in range(2,i+2):
            
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # something get left and diag left
            left = matrix[i][j-1]
            diag_left = matrix[i-1][j-1]
            numerator = left - diag_left

            denominator = matrix[i][0] - matrix[i-j+1][0]

            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix


def hermite_interpolation(x_points, y_points, slopes):

    #main difference with hermite's method , using instances with x 

    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2))

    #populate x values

    index = 0
    for x in range (0, len(matrix), 2):
        matrix[x][0]= x_points[index]
        matrix[x+1][0] = x_points[index]
        index += 1

    # prepopulate y values
    index = 0
    for x in range (0, len(matrix), 2):
        matrix[x][1]= y_points[index]
        matrix[x+1][1] = y_points[index]
        index += 1

    #prepopulate with derivatives (every other row)
    index = 0
    for x in range(1, len(matrix), 2):
        matrix[x][2] = slopes[index]
        index += 1

    apply_div_diff(matrix)
    print(matrix)

def matrix_a(x_points, h_points):
    
    size = len(x_points)
    
    matrix = np.zeros((size, size))

    # slides told me to do this so i did it
    matrix[0][0] = 1
    matrix[size-1][size-1] = 1

    for i in range(1, size - 1):
        matrix[i][i - 1] = h_points[i - 1]
        matrix[i][i] =  2  * (h_points[i - 1] + h_points[i])
        matrix[i][i + 1] = h_points[i]

    return matrix

def vector_b(y_points, h_points):

    size = len(y_points)
    matrix = np.zeros(size)

    # again, the slides do this
    matrix[0] = 0
    matrix[size-1] = 0

    for i in range(1, size - 1):
        mult1 = (3 / h_points[i]) * (y_points[i + 1] - y_points[i])
        mult2 = (3 / h_points[i - 1]) * (y_points[i] - y_points[i - 1])

        subtraction = mult1 - mult2
        matrix[i] = subtraction
    
    return matrix



def problem1():
    x_points = [3.6,3.8,3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    matrix = nevilles_method(x_points, y_points, approximating_value)
    print(matrix[2][2])

def problem2and3():
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    value = 7.3
    matrix = divided_difference_table(x_points, y_points)

    poly_approx : np.array = np.zeros(3)
    poly_approx[0] = matrix[1][1]
    poly_approx[1] = matrix[2][2]
    poly_approx[2] = matrix[3][3]

    # print(matrix)
    print(poly_approx)
    print()
    print(get_approximate_result(matrix, x_points, value))

def problem4():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_interpolation(x_points, y_points, slopes)

    # x = BA^-1, A^inv is one line, to multiply, dot = matrix * vector
def problem5():
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    h_points = []

    for i in range(0, len(x_points) - 1):
        h_points.append(x_points[i + 1] - x_points[i])

    a = matrix_a(x_points,h_points)
    print(a)
    print()
    b = vector_b(y_points, h_points)
    print(b)
    print()
    x = np.dot(inv(a), b)
    print(x)


if __name__ == "__main__":
    problem1()
    print()
    problem2and3()
    print()
    problem4()
    print()
    problem5()

    

