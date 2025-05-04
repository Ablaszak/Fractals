import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import Basic

def make_callable(f, x):
    """
    Jeśli f jest obiektem Sympy (wierszem drzewa wyrażeń),
    zwróć lambdify(x, f), w przeciwnym razie zwróć f.
    """
    if isinstance(f, Basic):
        # f to np. sin(x) czy x**2 — czyli coś, co można przekonwertować
        return sp.lambdify(x, f, 'numpy')
    elif callable(f):
        # już callable Pythona
        return f
    else:
        raise TypeError(f"Oczekiwałem obiektu Sympy lub callable, dostałem {type(f)}")

def save_grid(grid, file_name='grid_image.png'): # Przyznaję, napisane przez GPT
    # Przekształcenie siatki na macierz 0 i 1
    grid_array = np.array([[1 if cell == 'X' else 0 for cell in row] for row in grid])

    # Tworzenie obrazu
    plt.imshow(grid_array, cmap='Greys', interpolation='nearest')

    # Usuwanie osi, żeby obraz wyglądał czysto
    plt.axis('off')

    # Zapis obrazu do pliku
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

# Finds sets of x values, where f(x) is in row range
def find_xs_sympy(f, x, domain, ran, count): # USE VERTICAL COUNT
    xs = [None for _ in range(count)]
    b_size = (abs(ran.inf) + abs(ran.sup)) / count

    for i in range(count):
        top = ran.sup - (i*b_size)
        bottom = top-b_size
        below_top = sp.solve_univariate_inequality(sp.Le(f, top), x, False, domain)
        above_bottom = sp.solve_univariate_inequality(sp.Ge(f, bottom), x, False, domain)
        xs[i] = sp.Intersection(below_top, above_bottom)
    return xs

def find_xs_numeric(f, x, domain, ran, count, res=10000):
    b_size = float((ran.sup - ran.inf) / count)
    x_step = (domain.sup - domain.inf) / res
    xs = [[' ' for _ in range(res)] for _ in range(count)]

    #N = sp.lambdify(x, f, 'numpy')
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

x = sp.symbols('x')
f = sp.sin(x)
p = (3.14)
grid = find_xs_numeric(f, x, sp.Interval(-p, p), sp.Interval(-1, 1), 1000, 4000)

#for i in range(len(grid)):
#    for j in grid[i]:
#        print(j, end = "")
#    print()

save_grid(grid, 'output_grid.png')
