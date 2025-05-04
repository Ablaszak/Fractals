import sympy as sp

horizontal_box_count = 32
def find_next_two(num):
    two = 2
    while(two <= abs(num)):
        two *= 2

    if(num < 0):
        return -two
    return two

def create_grid_x(f): # For one variable functions

    print("Podaj dziedzinę: ")
    x1 = float(input())
    x2 = float(input())
    domain_f = sp.Interval(x1, x2)

    x = sp.symbols('x')
    range_f = sp.calculus.util.function_range(f, x, domain_f)

    # check if interval is open:
    if(range_f.is_unbounded):
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
    middle = (((domain_f.inf + domain_f.sup)/2) , ((range_f.inf + range_f.sup)/2))
    global horizontal_box_count
    box_size = ( (abs(domain_f.inf) + abs(domain_f.sup) ) / horizontal_box_count)
    # Now check how many boxes will fit vertically:
    vert_box_count = (abs(range_f.inf) + abs(range_f.sup) // box_size
    #