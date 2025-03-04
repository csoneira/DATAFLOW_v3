#%%

import numpy as np

def detection_prob(pattern, e1, e2, e3, e4, subset):
    """
    Return the probability that a muon from the 'subset' (one of the 6 physical subsets)
    ends up being detected in the given 'pattern' of fired planes.

    subset should be one of: '12', '23', '34', '123', '234', '1234'
    pattern can be any measured combo: '12','13','14','23','24','34',
                                      '123','124','134','234','1234'

    e1, e2, e3, e4 are the detection efficiencies of planes 1..4 respectively.

    We assume strictly layered geometry:
      - Any track from plane i to plane j with i<j must physically cross all planes in between.
      - So "13" detection means physically it's either subset '123' or '1234' (with plane2 failing).
      - "14" detection means physically only '1234' is possible (with planes2,3 failing).
      - etc.
    """

    # ---------- 4-plane pattern ----------
    if pattern == '1234':
        if subset == '1234':
            return e1 * e2 * e3 * e4
        else:
            return 0.0

    # ---------- 3-plane patterns ----------
    if pattern == '123':
        # can come from subset '123' or '1234' (plane4 fails if physically crossing it)
        if subset == '123':
            return e1 * e2 * e3
        elif subset == '1234':
            return e1 * e2 * e3 * (1 - e4)
        else:
            return 0.0

    if pattern == '234':
        # can come from '234' or '1234' (plane1 fails)
        if subset == '234':
            return e2 * e3 * e4
        elif subset == '1234':
            return (1 - e1) * e2 * e3 * e4
        else:
            return 0.0

    if pattern == '124':
        # physically includes planes 1,2,4 -> only subset '1234' can do that,
        # plane3 must fail detection
        if subset == '1234':
            return e1 * e2 * e4 * (1 - e3)
        else:
            return 0.0

    if pattern == '134':
        # physically includes 1,2,3,4 -> so subset '1234' only,
        # plane2 fails
        if subset == '1234':
            return e1 * e3 * e4 * (1 - e2)
        else:
            return 0.0

    # ---------- 2-plane patterns ----------

    # 12
    if pattern == '12':
        # subsets that physically include planes 1 & 2: '12','123','1234'
        if subset == '12':
            return e1 * e2
        elif subset == '123':
            return e1 * e2 * (1 - e3)
        elif subset == '1234':
            return e1 * e2 * (1 - e3) * (1 - e4)
        else:
            return 0.0

    # 23
    if pattern == '23':
        # can come from '23','123','234','1234'
        #   '23': e2*e3
        #   '123': (1-e1)*e2*e3
        #   '234': e2*e3*(1-e4)
        #   '1234': (1-e1)*e2*e3*(1-e4)
        if subset == '23':
            return e2 * e3
        elif subset == '123':
            return (1 - e1) * e2 * e3
        elif subset == '234':
            return e2 * e3 * (1 - e4)
        elif subset == '1234':
            return (1 - e1) * e2 * e3 * (1 - e4)
        else:
            return 0.0

    # 34
    if pattern == '34':
        # from '34','234','1234'
        #   '34': e3*e4
        #   '234': (1-e2)*e3*e4
        #   '1234': (1-e1)*(1-e2)*e3*e4
        if subset == '34':
            return e3 * e4
        elif subset == '234':
            return (1 - e2) * e3 * e4
        elif subset == '1234':
            return (1 - e1) * (1 - e2) * e3 * e4
        else:
            return 0.0

    # 13
    if pattern == '13':
        # physically, crossing planes 1 & 3 means also crossing plane2
        # so subsets '123' or '1234' with plane2 failing. plane4 fails if in '1234'
        if subset == '123':
            return e1 * (1 - e2) * e3
        elif subset == '1234':
            return e1 * (1 - e2) * e3 * (1 - e4)
        else:
            return 0.0

    # 14
    if pattern == '14':
        # physically must cross 2,3 => only '1234', with plane2 & 3 failing
        if subset == '1234':
            return e1 * e4 * (1 - e2) * (1 - e3)
        else:
            return 0.0

    # 24
    if pattern == '24':
        # physically must cross plane3 if it crosses 2->4 => subsets '234' or '1234'
        # '234': e2,e4 detect, plane3 fails => e2*(1-e3)*e4
        # '1234': plane1 fails, plane2 &4 ok, plane3 fails => (1-e1)*e2*(1-e3)*e4
        if subset == '234':
            return e2 * e4 * (1 - e3)
        elif subset == '1234':
            return (1 - e1) * e2 * (1 - e3) * e4
        else:
            return 0.0

    # If user measured something else, default 0:
    return 0.0



def build_matrix_and_vector(e1, e2, e3, e4, measured):
      """
      Build the A matrix and b vector for a set of measured patterns.

      measured : list of tuples like [('12', R12_value), ('13', R13_value), ...]
                  containing ALL patterns you want to include in the fit.

      We'll have 6 unknowns: [x12, x23, x34, x123, x234, x1234].
      Each measured pattern -> 1 row in A, 1 entry in b.
      """
      subsets = ['12','23','34','123','234','1234']  # The unknown x_... in this order
      A_rows = []
      b_vals = []

      for (pattern, Rval) in measured:
            # Build one row of length 6
            row_coeffs = []
            for subset in subsets:
                  p = detection_prob(pattern, e1, e2, e3, e4, subset)
                  row_coeffs.append(p)
            A_rows.append(row_coeffs)
            b_vals.append(Rval)

      A = np.array(A_rows, dtype=float)
      b = np.array(b_vals, dtype=float)
      return A, b, subsets

# -----------------------
# EXAMPLE USAGE FOR ONE ROW
# Suppose we have a row with these measured triggers:
my_measured_patterns = [
    ('12',   1000.0),
    ('13',    850.0),
    ('14',    100.0),
    ('23',   1200.0),
    ('24',    200.0),
    ('34',    950.0),
    ('123',   800.0),
    ('124',    80.0),
    ('134',    90.0),
    ('234',  1000.0),
    ('1234',  700.0),
]
# and plane efficiencies for this row:
e1_val, e2_val, e3_val, e4_val = 0.95, 0.92, 0.90, 0.88

# Build matrix system
A, b, subset_labels = build_matrix_and_vector(e1_val, e2_val, e3_val, e4_val, my_measured_patterns)

# If we have more patterns than 6, it's overdetermined. We'll do a least-squares solve:
x_best, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)

print("Subset labels (in order):", subset_labels)
print("Best-fit x-values:")
for label, val in zip(subset_labels, x_best):
    print(f"  x_{label} = {val:.3f}")

print("\nSum of squared residuals:", residuals)
print("Rank of A:", rank)

# %%



import numpy as np

def build_matrix_and_vector(e1, e2, e3, e4, measured):
    """
    measured : list of (pattern, R_value) like ('12', R12), ('13', R13), ...
    returns: (A, b, subset_list)
    """
    subset_list = ['12','23','34','123','234','1234']
    A_rows = []
    b_vals = []
    for (pattern, Rval) in measured:
        row = []
        for sub in subset_list:
            p = detection_prob(pattern, e1, e2, e3, e4, sub)
            row.append(p)
        A_rows.append(row)
        b_vals.append(Rval)
    A = np.array(A_rows, dtype=float)
    b = np.array(b_vals, dtype=float)
    return A, b, subset_list

def solve_for_subsets(row):
    # example: gather measured data for this row
    measured_patterns = [
        ('12',   row['type_12']),
        ('13',   row['type_13']),
        ('14',   row['type_14']),
        ('23',   row['type_23']),
        ('24',   row['type_24']),
        ('34',   row['type_34']),
        ('123',  row['type_123']),
        ('124',  row['type_124']),
        ('134',  row['type_134']),
        ('234',  row['type_234']),
        ('1234', row['type_1234']),
    ]

    # plane efficiencies in each row
    e1_val, e2_val, e3_val, e4_val = (
        row['final_eff_1'], 
        row['final_eff_2'], 
        row['final_eff_3'], 
        row['final_eff_4'],
    )

    # build system
    A, b, subs = build_matrix_and_vector(e1_val, e2_val, e3_val, e4_val, measured_patterns)

    # solve in least-squares sense
    x_best, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    # pack results
    out = {}
    for label, val in zip(subs, x_best):
        out[f"x_{label}"] = val
    out['residual_sum_sq'] = float(residuals[0]) if len(residuals) else 0.0
    out['rank'] = rank
    return pd.Series(out)

# Then apply over your DataFrame:
df_unfolded = df.apply(solve_for_subsets, axis=1)

# "df_unfolded" now has columns x_12, x_23, x_34, x_123, x_234, x_1234, plus residual info

