#%%

import sympy

# --- 1) Define symbolic variables ---

# The true (geometric) rates we want to find:
x12, x13, x23, x34 = sympy.symbols('x12 x13 x23 x34', nonnegative=True)
x123, x124, x134, x234, x1234 = sympy.symbols(
    'x123 x124 x134 x234 x1234', 
    nonnegative=True
)

# The measured (observed) coincidence rates (known numbers in practice):
R12, R13, R23, R34 = sympy.symbols('R12 R13 R23 R34', nonnegative=True)
R123, R124, R134, R234, R1234 = sympy.symbols(
    'R123 R124 R134 R234 R1234', 
    nonnegative=True
)

# The plane efficiencies (also known inputs):
e1, e2, e3, e4 = sympy.symbols('e1 e2 e3 e4', positive=True)


# --- 2) Write down the detection equations ---

eqs = []

# 4-plane detection
eqs.append(sympy.Eq(
    R1234,
    x1234*(e1*e2*e3*e4)
))

# 3-plane detections
eqs.append(sympy.Eq(
    R123,
    x123*(e1*e2*e3) + x1234*(e1*e2*e3)*(1 - e4)
))
eqs.append(sympy.Eq(
    R124,
    x124*(e1*e2*e4) + x1234*(e1*e2*e4)*(1 - e3)
))
eqs.append(sympy.Eq(
    R134,
    x134*(e1*e3*e4) + x1234*(e1*e3*e4)*(1 - e2)
))
eqs.append(sympy.Eq(
    R234,
    x234*(e2*e3*e4) + x1234*(e2*e3*e4)*(1 - e1)
))

# 2-plane detections
eqs.append(sympy.Eq(
    R12,
    x12*(e1*e2)
    + x123*(e1*e2)*(1 - e3)
    + x124*(e1*e2)*(1 - e4)
    + x1234*(e1*e2)*(1 - e3)*(1 - e4)
))
eqs.append(sympy.Eq(
    R13,
    x13*(e1*e3)
    + x123*(e1*e3)*(1 - e2)
    + x134*(e1*e3)*(1 - e4)
    + x1234*(e1*e3)*(1 - e2)*(1 - e4)
))
eqs.append(sympy.Eq(
    R23,
    x23*(e2*e3)
    + x123*(e2*e3)*(1 - e1)
    + x234*(e2*e3)*(1 - e4)
    + x1234*(e2*e3)*(1 - e1)*(1 - e4)
))
eqs.append(sympy.Eq(
    R34,
    x34*(e3*e4)
    + x234*(e3*e4)*(1 - e2)
    + x134*(e3*e4)*(1 - e1)
    + x1234*(e3*e4)*(1 - e1)*(1 - e2)
))


# --- 3) Solve the linear system symbolically ---
solution = sympy.solve(eqs, [
    x12, x13, x23, x34, x123, x124, x134, x234, x1234
], dict=True)

# solution will be a list (usually of length 1 if consistent) of dicts
# mapping each x_... to an expression in terms of R_..., e_1,..., e_4.
print("Symbolic solution:")
for sol in solution:
    for var in [x12, x13, x23, x34, x123, x124, x134, x234, x1234]:
        print(f"{var} = {sympy.simplify(sol[var])}")

#
# x1234 in the solution is your efficiency-corrected 4-plane crossing rate.
#


# --- 4) (Optional) Plug in numerical values to get a numeric solution ---
# Suppose you know your plane efficiencies and measured rates:
e1_val = 0.95
e2_val = 0.90
e3_val = 0.98
e4_val = 0.93
R12_val = 20
R13_val = 160.0
R23_val = 170.0
R34_val = 180.0
R123_val = 200.0
R124_val = 210.0
R134_val = 220.0
R234_val = 230.0
R1234_val = 240.0

# You can substitute into the symbolic solution:
numeric_sol = {}
for var in solution[0]:
    numeric_sol[var] = solution[0][var].subs({
        e1: e1_val, e2: e2_val, e3: e3_val, e4: e4_val,
        R12: R12_val, R13: R13_val, R23: R23_val, R34: R34_val,
        R123: R123_val, R124: R124_val, R134: R134_val, R234: R234_val,
        R1234: R1234_val
    })

print("\nNumeric solution:")
for var in numeric_sol:
    print(f"{var} = {numeric_sol[var].evalf()}")

# %%
