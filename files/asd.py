import warprnnt_numba

fastemit_lambda = 0.001  # any float >= 0.0

loss_pt = warprnnt_numba.RNNTLossNumba(
    blank=4, reduction="sum", fastemit_lambda=fastemit_lambda
)

print(len("she_had_your_dark_suit_in_greasy_wash_water_all_year"))
