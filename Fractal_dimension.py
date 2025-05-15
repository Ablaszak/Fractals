import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.ma.core import asarray
from sympy import Basic
from PIL import Image

horizontal_box_count = 64

def make_callable(f, x):
    if isinstance(f, Basic):
        return sp.lambdify(x, f, 'numpy')
    elif callable(f):
        return f
    else:
        raise TypeError("Brzydko >:C", type(f))

def find_xs_numeric(f, x, domain, ran, count, res=10000):
    b_size = float((ran.sup - ran.inf) / count)
    x_step = (domain.sup - domain.inf) / res
    xs = [[' ' for _ in range(res)] for _ in range(count)]

    N = make_callable(f, x)
    # Check f(x) values for different x:
    current = float(domain.inf - (x_step / 2) )# To get average value
    for i in range(res):
        current += x_step
        # Exclude dangerous values:
        try:
            y = N(float(current))
        except(ZeroDivisionError, ValueError, TypeError):
            print("WTF", current)
            continue
        #print("UDALO SIE UDALO SIE UDALO SIE ---------------------")
        if(y < ran.inf or y > ran.sup):
            continue
        # Choosing a row in xs where the y is:
        row = round((float(ran.sup) - y) / b_size) # Trust me, it works
        if(0 <= row < count): # Additional protection
            xs[row][i] = 'X'
        else:
            print("fuck you", row)
    return xs

def find_next_two(num):
    two = 2
    while(two <= abs(num)):
        two *= 2

    if(num < 0):
        return -two
    return two


def create_grid_x(f): # For one variable functions

    x = sp.symbols('x')
    print("Podaj dziedzinę: ")
    x1 = float(input())
    x2 = float(input())
    domain_f = sp.Interval(x1, x2)
    print("Dziedzina: ", domain_f)
    try:
        print("Jeżeli proecs wykonuje się w nieskończoność, wciśnij ctrl+c")
        range_f = sp.calculus.util.function_range(f, x, domain_f)
    except:
        print("funkcja dąży do nieskończoności, podaj przedziały do analizy: ")
        y1 = float(input())
        y2 = float(input())
        range_f = sp.Interval(y1, y2)

    print("Przedział wartości: ", range_f)
    # check if interval is open:
    if(range_f.is_left_unbounded or range_f.is_right_unbounded):
        print("funkcja dąży do nieskończoności, podaj przedziały do analizy: ")
        y1 = float(input())
        y2 = float(input())
        range_f = sp.Interval(y1, y2)
    else:
        print("Czy chcesz przeanalizować fraktal w innym zakresie (w pionie)? (y/n): ")
        y1 = input()
        if(y1 == 'Y' or y1 == 'y'):
            print("Podaj przedziały (wartości zostaną lekko zwiększone): ")
            y1 = float(input())
            y2 = float(input())
            range_f = sp.Interval(y1, y2)

    # Now we actually create the grid:

    # Init:
    # First, we have to find grid dimensions
    middle = (((domain_f.sup - domain_f.inf)/2) , ((range_f.sup - range_f.inf)/2))
    global horizontal_box_count
    box_size = ((domain_f.sup - domain_f.inf) / horizontal_box_count)
    # Now check how many boxes will fit vertically:
    vert_box_count = (range_f.sup - range_f.inf) // box_size
    print("firstly we can fit ", vert_box_count, "boxes")
    # And expand it up to next_two() :
    vert_box_count = find_next_two(vert_box_count)
    range_f = sp.Interval(range_f.inf, range_f.inf + (vert_box_count * box_size))

    print(range_f, vert_box_count)

    # Actual init:

    grid = find_xs_numeric(f, x, domain_f, range_f, vert_box_count, horizontal_box_count)

    #save_grid(grid, 'output_grid.png')
    return grid

def rescale(grid):
    rows = len(grid)
    cols = len(grid[0])
    new = [[False for _ in range(cols)] for _ in range(rows)]

    for r in range(rows//2):
        for c in range(cols//2):
            new[r][c] = (grid[2*r][2*c] or grid[2*r][2*c + 1] or grid[2*r + 1][2*c] or grid[2*r + 1][2*c + 1])
    return new

def create_grid_IMG():
    img_loc = input("Podaj lokalizację/nazwę pliku: ")
    image = Image.open(img_loc).convert("L")

    arr = asarray(image)
    bool_arr = [[False for _ in range(len(arr[0]))] for _ in range(len(arr))]

    # Rewrite as True/False array:
    treshold = 100
    for row in range(len(arr)):
        for col in range(len(arr[0])):
            L = arr[row][col]
            if(L<treshold):
                bool_arr[row][col] = True

    # Expand to fit 2^x edge size:
    extra_cols = find_next_two(len(bool_arr[0])) - len(bool_arr[0])
    extra_rows = find_next_two(len(bool_arr)) - len(bool_arr)
    for row in range(len(bool_arr)):
        bool_arr[row] = bool_arr[row] + [False for _ in range(extra_cols)]
    newlen = len(bool_arr[0])
    bool_arr += [[False for _ in range(newlen)] for _ in range(extra_rows)]

    return bool_arr

def count_boxes(boxes, rows, cols):

    for row in range(len(boxes)):
        for col in range(len(boxes[0])):
            if(boxes[row][col] == True):
                print('X', end = "")
            else:
                print(' ', end = "")
        print()

    N = 0
    for r in range(rows):
        for c in range(cols):
            if(boxes[r][c] == True):
                N += 1

    if(rows == 1 or cols == 1): # End recursion
        return [N]

    # Prepare new box grid (scaled to bigger boxes - smaller resolution)
    boxes = rescale(boxes) # Overwriting the array helps save some space (I think xD)
    return count_boxes(boxes, rows//2, cols//2) + [N]


def compute_dimension(b_num):
    """
    Important: We use log base = 2
    We also use scaling factor = 2, so log(s) = i
    """
    n = len(b_num)
    logs = np.empty([n])
    logN = np.empty([n])

    for i in range(n):
        logs[i] = i # log2(2^i) = i
        logN[i] = math.log2(b_num[i])
    a, b = np.polyfit(logs, logN, 1)
    return a

arr = create_grid_IMG()
Ns = count_boxes(arr, len(arr), len(arr[0]))
print(Ns)
print(compute_dimension(Ns))
"""
x = sp.symbols('x')

grid = (create_grid_x(sp.cos(1/x)))

for i in range(len(grid)):
    for j in grid[i]:
        print(j, end="")
    print()
"""