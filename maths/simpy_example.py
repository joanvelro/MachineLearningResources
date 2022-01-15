import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import sys



def example_simpy():
    t = np.linspace(0, 20, 500)

    plt.plot(t, np.sin(t))
    plt.show()
    #plt.savefig("test.pdf", dpi=150)
    #plt.close()

    sp.init_printing(use_latex='mathjax')

    x, y, z = sp.symbols('x y z')
    print('Function to integrate respect to x:')
    print('sin(xy) + cos(yz)')
    f = sp.sin(x * y) + sp.cos(y * z)
    a = sp.integrate(f, x)
    print('result:')

    return a.as_expr()


if __name__ == '__main__':
    val = example_simpy()
    print(val)
