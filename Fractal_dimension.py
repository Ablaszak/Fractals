import sympy as sp

def create_grid(f):
    print("Podaj dziedzinę: ")
    x1 = input()
    x2 = input()
    domain = sp.Interval(x1, x2)

    x = sp.symbols('x')
    range_f = sp.calculus.util.function_range(f, x, domain)

    # check if interval is open:
    if(range_f.is_unbounded):
        print("funkcja dąży do nieskończoności, podaj przedziały do analizy: ")
        y1 = input()
        y2 = input()

    # Now we actually create the grid:

    # Init:
