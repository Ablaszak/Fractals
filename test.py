import sympy as sp
import matplotlib.pyplot as plt

horizontal_box_count = 32
def find_next_two(num):
    two = 2
    while(two <= abs(num)):
        two *= 2

    if(num < 0):
        return -two
    return two

def find_values(f, x, x1, x2, bottom, top):
    dom = sp.Interval(x1, x2)
    below_top = sp.solve_univariate_inequality(sp.Le(f, top), x, False, dom)
    above_bottom = sp.solve_univariate_inequality(sp.Ge(f, bottom), x, False, dom)

    return(sp.Intersection(below_top, above_bottom))

def is_in_box(bottom, top, value_set):
    box = sp.Interval(bottom, top)
    if(sp.Intersection(box, value_set) != sp.EmptySet):
        return 'X'
    return ' '

def create_grid_x(f): # For one variable functions

    print("Podaj dziedzinę: ")
    x1 = float(input())
    x2 = float(input())
    domain_f = sp.Interval(x1, x2)

    x = sp.symbols('x')
    range_f = sp.calculus.util.function_range(f, x, domain_f)

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
    # And expand it up to next_two() :
    range_f = sp.Interval(range_f. inf, range_f.sup + find_next_two(vert_box_count) - vert_box_count)
    vert_box_count = find_next_two(vert_box_count)

    # Actual init:
    grid = [[False for _ in range(horizontal_box_count)] for _ in range(vert_box_count)]

    # Now it is time to mark boxes

    # For every x range, we find sets of values of f(x)
    values = [None for _ in range(horizontal_box_count)]
    for i in range(horizontal_box_count):
        values[i] = find_values(f, x, i, i+box_size, domain_f.inf, domain_f.sup)
    # Mark boxes:
    for row in range(vert_box_count):
        for col in range(horizontal_box_count):
            bottom = range_f.sup - ((row+1) * box_size)
            grid[row][col] = is_in_box(bottom, bottom+box_size, values[col])
    return grid

def graph():
    x = sp.symbols('x')
    grid = create_grid_x(x**2)


    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.show()

x = sp.symbols('x')

grid = (create_grid_x(x**2))

for i in range(len(grid)):
    print(grid[i])