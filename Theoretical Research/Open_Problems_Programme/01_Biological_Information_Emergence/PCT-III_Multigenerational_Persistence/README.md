# PCT-III: Multigenerational Persistence

Status: tested within PCT-I (test 2). No loss observed across three generations.

Question. Does sequence-dependent catalytic information persist across generations under stochastic replication, or is it lost?

Result. See [PCT-I RESULT_NOTE](../PCT-I_Hereditary_Catalytic_Capacity/RESULT_NOTE.md). Capacity decays gradually with copying error (for example, median C1 = 3.82, C2 = 3.30, C3 = 2.95 bits at p = 0.05, eps = 0.05). No RAF network fell below the 0.10-bit loss threshold within three generations in 18 cells.

Open. Longer horizons (g > 3) and a loss threshold that is not fixed to 0.10 bits have not been tested. A bound on the decay rate, if one exists, would be the next contribution.
