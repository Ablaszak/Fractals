import sympy as sp

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
    b_size = (abs(ran.inf) + abs(ran.sup)) / count
    x_step = (abs(domain.inf) + abs(domain.sup)) / res
    xs = [[' ' for _ in range(res)] for _ in range(count)]

    N = sp.lambdify(x, f, 'numpy')

    # Check f(x) values for different x:
    current = domain.inf - (x_step / 2) # To get average value
    for i in range(res):
        current += x_step
        # Exclude dangerous values:
        try:
            y = N(current)
        except(TypeError, ValueError, ZeroDivisionError, FloatingPointError, RuntimeWarning):
            continue

        if(y < ran.inf or y > ran.sup):
            continue
        # Choosing a row in xs where the y is:
        row = int((ran.sup - y) / b_size) # Trust me, it works
        if(0 <= row <= count): # Additional protection
            xs[row][i] = 'X'

    return xs

x = sp.symbols('x')
grid = find_xs_numeric(x**2, x, sp.Interval(-5, 5), sp.Interval(-25, 25), 2048)

print()