from sympy import *
x, y, a, r = symbols('x y a r')
fun = Eq( (x**2 + y**2)**2, 2* a**2 * (x**2 - y**2) )
cykloid = Eq(r, a*(1 + cos(x)))

ans = diff