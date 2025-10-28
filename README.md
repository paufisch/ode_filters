# ODE Filters

## Testing

### Running Tests

To run the test suite, use pytest:

```bash
uv run pytest test/ -v
```

Or for specific test file:

```bash
uv run pytest test/test_gaussian_inference.py -v
```

### Test Suite for `marginalization` Function

The `test/test_gaussian_inference.py` file contains comprehensive tests for the `marginalization` function in `ode_filters/gaussian_inference.py`.

#### Test Coverage

1. **test_marginalization_basic**: Tests basic functionality with a simple 2D case, verifying exact numerical results.

2. **test_marginalization_shapes**: Validates that output shapes are correct across various input dimensions (1D→1D, 2D→1D, 3D→2D, 5D→3D).

3. **test_marginalization_zero_noise**: Tests the special case where observation noise Q = 0, verifying the simplified computation.

4. **test_marginalization_identity_transform**: Tests with identity transformation (A = I, b = 0) to verify the additive property of covariances.

5. **test_marginalization_with_offset**: Isolates testing of the offset parameter b to ensure it's correctly applied to the mean.

6. **test_marginalization_covariance_is_positive_semidefinite**: Verifies that output covariances maintain the positive semi-definite property across 5 random test cases.

7. **test_marginalization_dimensional_reduction**: Tests dimension reduction scenarios where we observe fewer dimensions than the state space.

8. **test_marginalization_linear_combination**: Tests observations that are linear combinations of state variables.

9. **test_marginalization_repeated_calls_consistency**: Ensures deterministic behavior - repeated calls with same inputs produce identical outputs.

10. **test_marginalization_numerical_stability_small_values**: Tests numerical stability with very small values (1e-5 to 1e-10 range).

11. **test_marginalization_numerical_stability_large_values**: Tests numerical stability with very large values (1e5 to 1e10 range).

#### Expected Behavior

The `marginalization` function computes the marginal distribution of `z = Ax + b` given:

- Prior: `p(x) ~ N(mu, Sigma)`
- Likelihood: `p(z|x) ~ N(Ax + b, Q)`

The function returns:

- `mu_z = A @ mu + b` (marginal mean)
- `Sigma_z = A @ Sigma @ A.T + Q` (marginal covariance)
