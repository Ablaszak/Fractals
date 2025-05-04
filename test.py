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

def find_xs_numeric(f, x, domain, ran, count, res=1000):
    asd =0

x = sp.symbols('x')
print(find_xs_sympy(x**2, x, sp.Interval(-5, 5), sp.Interval(-25, 25), 2048))
