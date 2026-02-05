import torch
import pytest
import sys
import os

# Add src to path so we can import smrs
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from smrs.utils import compute_lambda, shrink_l1_lq


def test_compute_lambda_zeros():
    """Test that lambda is 0 for a zero matrix."""
    Y = torch.zeros((5, 10))
    lamb = compute_lambda(Y)
    assert torch.isclose(lamb, torch.tensor(0.0, dtype=torch.float64), atol=1e-6)


def test_compute_lambda_simple_case():
    """
    Test compute_lambda with a known simple case.
    Consider 2 points in 2D: p1=[1, 0], p2=[0, 1].
    Mean = [0.5, 0.5].
    Affine term (Mean - Y) = [[-0.5, 0.5], [0.5, -0.5]].
    T[0] = norm(p1 @ affine) = norm([1, 0] @ [[-0.5, 0.5], [0.5, -0.5]])
         = norm([-0.5, 0.5]) = sqrt(0.5) approx 0.707.
    """
    Y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    lamb = compute_lambda(Y)

    expected = torch.sqrt(torch.tensor(0.5, dtype=torch.float64))

    assert torch.isclose(lamb, expected, atol=1e-6)


def test_shrink_l1_lq_q1():
    """Test shrinkage with q=1 (Soft Thresholding)."""
    Z = torch.tensor([[0.5, -0.5], [0.1, -0.8]])
    lambda_param = 0.2

    # Expected: sign(x) * max(|x| - lambda, 0)
    # 0.5 -> 0.3
    # -0.5 -> -0.3
    # 0.1 -> 0.0 (shrunk to zero)
    # -0.8 -> -0.6
    expected = torch.tensor([[0.3, -0.3], [0.0, -0.6]], dtype=torch.float64)

    res = shrink_l1_lq(Z, lambda_param, q=1)
    assert torch.allclose(res, expected, atol=1e-6)


def test_shrink_l1_lq_q2():
    """
    Test shrinkage with q=2 (Row-wise shrinkage).
    Formula: row * max(1 - lambda/norm(row), 0)
    """
    # Row 1: [3, 4], norm=5. lambda=2. Factor = max(1 - 2/5, 0) = 0.6. Result: [1.8, 2.4]
    # Row 2: [0.1, 0.1], norm=sqrt(0.02)~0.14. lambda=2. Factor=0. Result: [0, 0]
    Z = torch.tensor([[3.0, 4.0], [0.1, 0.1]], dtype=torch.float64)
    lambda_param = 2.0

    res = shrink_l1_lq(Z, lambda_param, q=2)

    row1_expected = torch.tensor([1.8, 2.4], dtype=torch.float64)
    row2_expected = torch.tensor([0.0, 0.0], dtype=torch.float64)

    assert torch.allclose(res[0], row1_expected, atol=1e-6)
    assert torch.allclose(res[1], row2_expected, atol=1e-6)


def test_shrink_unsupported_q():
    """Test that invalid q raises ValueError."""
    Z = torch.randn(2, 2)
    with pytest.raises(ValueError):
        shrink_l1_lq(Z, 0.5, q=3)
