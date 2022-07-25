#!/usr/bin/env python

from functools import reduce

print("Hola", type(2), type(12.45), type(2 + 4j), type(False), type(None))

a = 2 + 2
b = 20 / a

print(a, b)

print(23 + 7 * 2 % 2 == 0)


def even_or_odd(number):
    return "eovdedn"[number % 2 == 1::2]


print(even_or_odd(11), even_or_odd(-10))

x = 54

if x % 3 == 0:
    result = x
elif x % 3 == 1:
    result = x**2
else:
    result = x**3

print(result)

x = 3.5
x_bound = x if 0 <= x <= 5 else 0.0

print(x_bound)

# least common multiple
m = 32
n = 12

d = min(m, n)

while m % d != 0 or n % d != 0:
    d -= 1  # Atención con esta instrucción!

print(d)


for color in ["blue", "red", "green", "yellow", "black", "pink"]:
    print(color)


def suma(*args):
    return reduce((lambda x, y: x + y), args)


print(suma(1, 2, 3, 4, 5))
