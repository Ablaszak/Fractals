def find_next_two(num):
    two = 2
    while (two <= abs(num)):
        two *= 2

    if (num < 0):
        return -two
    return two

print(find_next_two(-11))