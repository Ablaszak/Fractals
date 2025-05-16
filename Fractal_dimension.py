import sympy as sp
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from numpy import asarray
from sympy import Basic
from PIL import Image
from skimage.morphology import skeletonize

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
    while(two < abs(num)):
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
    rows = len(grid)//2
    cols = len(grid[0])//2
    new = [[False for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            new[r][c] = (grid[2*r][2*c] or grid[2*r][2*c + 1] or grid[2*r + 1][2*c] or grid[2*r + 1][2*c + 1])
    return new

def prepare_IMG(img):
    img = img.convert("L")

    # Resize:
    hor, vert = img.size
    hor = find_next_two(hor)
    vert = find_next_two(vert)
    img = img.resize( (hor, vert) )

    img.save("Prepared_image.png")
    return img

def create_grid_IMG():
    # Open and prepare image:
    img_loc = input("Podaj lokalizację/nazwę pliku: ")
    image = Image.open(img_loc)
    image = prepare_IMG(image)


    arr = asarray(image)
    bool_arr = [[False for _ in range(len(arr[0]))] for _ in range(len(arr))]

    # Rewrite as True/False array:
    threshold = 100
    for row in range(len(arr)):
        for col in range(len(arr[0])):
            L = arr[row][col]
            if(L<threshold):
                bool_arr[row][col] = True

    """
    # Expand to fit 2^x edge size:
    extra_cols = find_next_two(len(bool_arr[0])) - len(bool_arr[0])
    extra_rows = find_next_two(len(bool_arr)) - len(bool_arr)
    for row in range(len(bool_arr)):
        bool_arr[row] = bool_arr[row] + [False for _ in range(extra_cols)]
    newlen = len(bool_arr[0])
    bool_arr += [[False for _ in range(newlen)] for _ in range(extra_rows)]
    """
    # Skeletonize:
    #skel = input("Czy chcesz wygładzić/zmniejszyć grubość linii? (y/n): ")
    #if(skel == 'y' or skel == 'Y'):
    #    bool_arr = skeletonize(np.array(bool_arr, dtype=bool))
    #debug:
    """
    for row in range(len(bool_arr)):
        for col in range(len(bool_arr[0])):
            if(bool_arr[row][col] == True):
                print('X', end = "")
            else:
                print(' ', end = "")
        print()
    """
    return bool_arr

def count_boxes(boxes, rows, cols):
    #debug:
    """
    for row in range(len(boxes)):
        for col in range(len(boxes[0])):
            if(boxes[row][col] == True):
                print('X', end = "")
            else:
                print(' ', end = "")
        print()
    """
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

    # Kill first results: (better or not?)
    b_num = b_num[len(b_num)//5 : ]
    print(b_num)

    n = len(b_num)
    logs = np.empty([n])
    logN = np.empty([n])

    for i in range(n):
        logs[i] = i # log2(2^i) = i
        logN[i] = log2(b_num[i])

    a, b = np.polyfit(logs, logN, 1)

    # Pisane przez chat xD:
    # 1. Scatter: zaznaczamy punkty
    plt.scatter(logs, logN, marker='o', label='punkty (log s, log N)')

    # 2. Linia regresji: y = a*x + b
    x_line = np.array([logs.min(), logs.max()])
    y_line = a * x_line + b
    plt.plot(x_line, y_line, linestyle='-', label=f'fit: y={a:.3f}x+{b:.3f}')

    # 3. Opisy osi i legenda
    plt.xlabel('log₂(skalowanie) [i]')
    plt.ylabel('log₂(liczba pudełek) [log N]')
    plt.title('Box‐counting: log–log i linia regresji')
    plt.legend()
    plt.grid(True)

    # 4. Zapisz wykres
    plt.savefig('wykres.png', dpi=300, bbox_inches='tight')
    plt.close()

    return a

arrout = create_grid_IMG()
Ns = count_boxes(arrout, len(arrout), len(arrout[0]))
print(compute_dimension(Ns))
"""
x = sp.symbols('x')

grid = (create_grid_x(sp.cos(1/x)))

for i in range(len(grid)):
    for j in grid[i]:
        print(j, end="")
    print()
"""