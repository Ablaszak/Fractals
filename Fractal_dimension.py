import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import Basic

horizontal_box_count = 1024

def make_callable(f, x):
    if isinstance(f, Basic):
        return sp.lambdify(x, f, 'numpy')
    elif callable(f):
        return f
    else:
        raise TypeError("Brzydko >:C", type(f))

def save_grid(grid, file_name='grid_image.png'): # Przyznaję, napisane przez GPT
    # Przekształcenie siatki na macierz 0 i 1
    grid_array = np.array([[1 if cell == 'X' else 0 for cell in row] for row in grid])

    # Tworzenie obrazu
    plt.imshow(grid_array, cmap='Greys', interpolation='nearest')

    # Usuwanie osi, żeby obraz wyglądał czysto
    plt.axis('on')

    # Zapis obrazu do pliku
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

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
        print("Czy chcesz przeanalizować fraktal tylko w wybranym zakresie (w pionie)? (y/n): ")
        y1 = input()
        if(y1 == 'Y' or y1 == 'y'):
            print("Podaj przedziały (wartości zostaną lekko zwiększone): ")
            y1 = float(input())
            y2 = float(input())
            range_f = sp.Interval(y1, y2)

    # Now we actually create the grid:

    # Init:
    # First, we have to find grid dimensions
    middle = (((domain_f.inf + domain_f.sup)/2) , ((range_f.inf + range_f.sup)/2))
    global horizontal_box_count
    box_size = ( (abs(domain_f.inf) + abs(domain_f.sup) ) / horizontal_box_count)
    # Now check how many boxes will fit vertically:
    vert_box_count = (abs(range_f.inf) + abs(range_f.sup) ) // box_size
    print("firstly we can fit ", vert_box_count, "boxes")
    # And expand it up to next_two() :
    range_f = sp.Interval(range_f.inf, range_f.sup + (find_next_two(vert_box_count) - vert_box_count) * box_size)
    vert_box_count = find_next_two(vert_box_count)

    print(range_f, vert_box_count)

    # Actual init:

    grid = find_xs_numeric(f, x, domain_f, range_f, vert_box_count, horizontal_box_count)

    #save_grid(grid, 'output_grid.png')
    return grid

def count_boxes(boxes, rows, cols):
    N = 0
    for r in range(rows):
        for c in range(cols):
            if(boxes[r][c] == True):
                N += 1
    # Prepare new box grid (scaled to bigger boxes - smaller resolution)
    scaled = [[False for _ in range(cols//2)] for _ in range(rows//2)]

    if(rows == 1 or cols == 1): # End recursion
        return [N]
    return [N] + count_boxes(scaled, rows//2, cols//2)

x = sp.symbols('x')

grid = (create_grid_x(sp.sin(1/x)))

for i in range(len(grid)):
    for j in grid[i]:
        print(j, end="")
    print()