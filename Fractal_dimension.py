import sympy as sp
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from numpy import asarray
from sympy import Basic
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # :3

horizontal_box_count = 64

def make_callable(f, x):
    if isinstance(f, Basic):
        return sp.lambdify(x, f, 'numpy')
    elif callable(f):
        return f
    else:
        raise TypeError("Type error, buddy >:C")

def find_xs_numeric(f, x, domain, ran, count, res=10000):
    b_size = float((ran.sup - ran.inf) / count)
    x_step = (domain.sup - domain.inf) / res
    xs = [[False for _ in range(res)] for _ in range(count)]

    N = make_callable(f, x)
    current = float(domain.inf - (x_step / 2))  # Average value

    for i in range(res):
        current += x_step
        try:
            y = N(float(current))
        except (ZeroDivisionError, ValueError, TypeError):
            print("WTF", current)
            continue

        if y < ran.inf or y > ran.sup:
            continue

        row = round((float(ran.sup) - y) / b_size)
        if 0 <= row < count:
            xs[row][i] = True
    return xs

def find_next_two(num):
    two = 2
    while two < abs(num):
        two *= 2

    return -two if num < 0 else two

def get_info_x(f, x):
    manual_range = False
    print("Enter the domain:")
    x1 = float(input())
    x2 = float(input())
    domain_f = sp.Interval(x1, x2)
    print("Domain:", domain_f)
    try:
        print("If the process seems endless, press Ctrl+C")
        range_f = sp.calculus.util.function_range(f, x, domain_f)
    except KeyboardInterrupt:
        manual_range = True
        print("Function goes to infinity. Please enter vertical range:")
        y1 = float(input())
        y2 = float(input())
        range_f = sp.Interval(y1, y2)

    print("Value range:", range_f)

    if range_f.is_left_unbounded or range_f.is_right_unbounded:
        print("Function tends to infinity. Enter vertical range:")
        y1 = float(input())
        y2 = float(input())
        range_f = sp.Interval(y1, y2)
    elif not manual_range:
        y1 = input("Do you want to analyze a different vertical range? (y/n): ")
        if y1.lower() == 'y':
            print("Enter the range (values will be slightly expanded):")
            y1 = float(input())
            y2 = float(input())
            range_f = sp.Interval(y1, y2)
            manual_range = True
    return domain_f, range_f

def create_grid_x(f):
    x = sp.symbols('x')
    domain_f, range_f = get_info_x(f, x)

    middle = ((domain_f.sup - domain_f.inf) / 2, (range_f.sup - range_f.inf) / 2)
    global horizontal_box_count
    box_size = (domain_f.sup - domain_f.inf) / horizontal_box_count
    vert_box_count = (range_f.sup - range_f.inf) // box_size
    print("Initially, we can fit", vert_box_count, "boxes")

    vert_box_count = find_next_two(vert_box_count)
    range_f = sp.Interval(range_f.inf, range_f.inf + (vert_box_count * box_size))

    print("Adjusted vertical range and box count:", range_f, vert_box_count)

    grid = find_xs_numeric(f, x, domain_f, range_f, vert_box_count, horizontal_box_count)

    return grid

# IMAGE PROCESSING -------------------------

def rescale(grid):
    rows = len(grid) // 2
    cols = len(grid[0]) // 2
    new = [[False for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            new[r][c] = (grid[2 * r][2 * c] or grid[2 * r][2 * c + 1] or
                         grid[2 * r + 1][2 * c] or grid[2 * r + 1][2 * c + 1])
    return new

def prepare_IMG(img):
    img = img.convert("L")

    if img.width != img.height:
        size = max(img.width, img.height)
        canvas = Image.new("L", (size, size), 255)
        canvas.paste(img)
        img = canvas

    hor, vert = img.size
    hor = find_next_two(hor)
    vert = find_next_two(vert)

    img.save("Prepared_image.png")
    return img

def create_grid_IMG():
    img_loc = input("Enter the file path/name: ")
    image = Image.open(img_loc)
    image = prepare_IMG(image)

    arr = asarray(image)
    bool_arr = [[False for _ in range(len(arr[0]))] for _ in range(len(arr))]

    threshold = 100
    for row in range(len(arr)):
        for col in range(len(arr[0])):
            L = arr[row][col]
            if L < threshold:
                bool_arr[row][col] = True
    return bool_arr

def count_boxes(boxes, rows, cols):
    N = 0
    for r in range(rows):
        for c in range(cols):
            if boxes[r][c]:
                N += 1

    if rows == 1 or cols == 1:
        return [N]

    boxes = rescale(boxes)
    return count_boxes(boxes, rows // 2, cols // 2) + [N]

def compute_dimension(b_num):
    b_num = b_num[len(b_num) // 5:]
    print(b_num)

    n = len(b_num)
    logs = np.empty([n])
    logN = np.empty([n])

    for i in range(n):
        logs[i] = i
        logN[i] = log2(b_num[i])

    a, b = np.polyfit(logs, logN, 1)

    plt.scatter(logs, logN, marker='o', label='points (log s, log N)')
    x_line = np.array([logs.min(), logs.max()])
    y_line = a * x_line + b
    plt.plot(x_line, y_line, linestyle='-', label=f'fit: y={a:.3f}x+{b:.3f}')
    plt.xlabel('log₂(scale) [i]')
    plt.ylabel('log₂(box count) [log N]')
    plt.title('Box-counting: log–log plot and regression line')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return a

def run_img():
    arrout = create_grid_IMG()
    Ns = count_boxes(arrout, len(arrout), len(arrout[0]))
    print(compute_dimension(Ns))

# MAIN ----------------------------------

fun = input("Select input type:\nImage (1)\nFunction of one variable (2)\nFunction of two variables (3)\n")

if fun == "1":
    run_img()
elif fun == "2":
    asd = 1  # Placeholder
