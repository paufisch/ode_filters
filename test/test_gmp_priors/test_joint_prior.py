"""Tests for JointPrior class."""

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import IWP, JointPrior, MaternPrior


class TestJointPriorConstruction:
    """Tests for JointPrior constructor."""

    def test_constructor_with_two_iwp_priors(self):
        """Test construction with two IWP priors."""
        prior_x = IWP(q=2, d=1)
        prior_u = IWP(q=3, d=1)
        joint = JointPrior(prior_x, prior_u)

        assert joint._prior_x is prior_x
        assert joint._prior_u is prior_u

    def test_constructor_with_different_priors(self):
        """Test construction with IWP and Matern priors."""
        prior_x = IWP(q=2, d=1)
        prior_u = MaternPrior(q=1, d=1, length_scale=1.0)
        joint = JointPrior(prior_x, prior_u)

        assert joint._prior_x is prior_x
        assert joint._prior_u is prior_u

    def test_rejects_non_prior_x(self):
        """Test that non-BasePrior for prior_x raises error."""
        prior_u = IWP(q=2, d=1)
        with pytest.raises(TypeError, match="prior_x must be BasePrior"):
            JointPrior("not a prior", prior_u)

    def test_rejects_non_prior_u(self):
        """Test that non-BasePrior for prior_u raises error."""
        prior_x = IWP(q=2, d=1)
        with pytest.raises(TypeError, match="prior_u must be BasePrior"):
            JointPrior(prior_x, [1, 2, 3])


class TestJointPriorMatrices:
    """Tests for JointPrior matrix methods."""

    @pytest.fixture
    def joint_prior(self):
        """Create a standard JointPrior for testing."""
        prior_x = IWP(q=2, d=1)  # D_x = 3
        prior_u = IWP(q=3, d=1)  # D_u = 4
        return JointPrior(prior_x, prior_u)

    def test_A_returns_block_diagonal(self, joint_prior):
        """Test that A(h) returns block-diagonal matrix."""
        h = 0.5
        A = joint_prior.A(h)

        # Expected shape: (D_x + D_u, D_x + D_u) = (7, 7)
        assert A.shape == (7, 7)

        # Check block structure
        A_x = joint_prior._prior_x.A(h)
        A_u = joint_prior._prior_u.A(h)

        # Top-left block should match A_x
        assert np.allclose(A[:3, :3], A_x)
        # Bottom-right block should match A_u
        assert np.allclose(A[3:, 3:], A_u)
        # Off-diagonal blocks should be zero
        assert np.allclose(A[:3, 3:], np.zeros((3, 4)))
        assert np.allclose(A[3:, :3], np.zeros((4, 3)))

    def test_Q_returns_block_diagonal(self, joint_prior):
        """Test that Q(h) returns block-diagonal matrix."""
        h = 0.5
        Q = joint_prior.Q(h)

        assert Q.shape == (7, 7)

        Q_x = joint_prior._prior_x.Q(h)
        Q_u = joint_prior._prior_u.Q(h)

        # Top-left block should match Q_x
        assert np.allclose(Q[:3, :3], Q_x)
        # Bottom-right block should match Q_u
        assert np.allclose(Q[3:, 3:], Q_u)
        # Off-diagonal blocks should be zero
        assert np.allclose(Q[:3, 3:], np.zeros((3, 4)))
        assert np.allclose(Q[3:, :3], np.zeros((4, 3)))

    def test_b_returns_concatenated_drifts(self, joint_prior):
        """Test that b(h) returns concatenated drift vectors."""
        h = 0.5
        b = joint_prior.b(h)

        assert b.shape == (7,)

        b_x = joint_prior._prior_x.b(h)
        b_u = joint_prior._prior_u.b(h)

        expected_b = np.concatenate([b_x, b_u])
        assert np.allclose(b, expected_b)


class TestJointPriorProjections:
    """Tests for JointPrior E0, E0_x, E0_hidden, and E1 properties."""

    def test_E0_extracts_both_states(self):
        """Test that E0 extracts both x and u from joint state."""
        prior_x = IWP(q=2, d=1)  # D_x = 3, d_x = 1
        prior_u = IWP(q=3, d=1)  # D_u = 4, d_u = 1
        joint = JointPrior(prior_x, prior_u)

        # E0 should extract [x, u] from [x, x', x'', u, u', u'', u''']
        E0 = joint.E0
        assert E0.shape == (2, 7)  # (d_x + d_u, D_x + D_u)

        # Test extraction
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        extracted = E0 @ state
        # Should get [x, u] = [1.0, 4.0]
        assert np.allclose(extracted, np.array([1.0, 4.0]))

    def test_E0_x_extracts_state_only(self):
        """Test that E0_x extracts only x from joint state."""
        prior_x = IWP(q=2, d=1)  # D_x = 3, d_x = 1
        prior_u = IWP(q=3, d=1)  # D_u = 4, d_u = 1
        joint = JointPrior(prior_x, prior_u)

        # E0_x should extract [x] from [x, x', x'', u, u', u'', u''']
        E0_x = joint.E0_x
        assert E0_x.shape == (1, 7)  # (d_x, D_x + D_u)

        # Test extraction
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        extracted = E0_x @ state
        # Should get [x] = [1.0]
        assert np.allclose(extracted, np.array([1.0]))

    def test_E0_hidden_extracts_hidden_only(self):
        """Test that E0_hidden extracts only u from joint state."""
        prior_x = IWP(q=2, d=1)  # D_x = 3, d_x = 1
        prior_u = IWP(q=3, d=1)  # D_u = 4, d_u = 1
        joint = JointPrior(prior_x, prior_u)

        # E0_hidden should extract [u] from [x, x', x'', u, u', u'', u''']
        E0_hidden = joint.E0_hidden
        assert E0_hidden.shape == (1, 7)  # (d_u, D_x + D_u)

        # Test extraction
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        extracted = E0_hidden @ state
        # Should get [u] = [4.0]
        assert np.allclose(extracted, np.array([4.0]))

    def test_E1_extracts_state_derivative(self):
        """Test that E1 extracts x' from joint state."""
        prior_x = IWP(q=2, d=1)  # D_x = 3, d_x = 1
        prior_u = IWP(q=3, d=1)  # D_u = 4, d_u = 1
        joint = JointPrior(prior_x, prior_u)

        # E1 should extract [x'] from joint state
        E1 = joint.E1
        assert E1.shape == (1, 7)  # (d_x, D_x + D_u)

        # Test extraction
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        extracted = E1 @ state
        # Should get [x'] = [2.0]
        assert np.allclose(extracted, np.array([2.0]))

    def test_E0_multidimensional(self):
        """Test E0 with multi-dimensional states."""
        prior_x = IWP(q=1, d=2)  # D_x = 4, d_x = 2
        prior_u = IWP(q=1, d=2)  # D_u = 4, d_u = 2
        joint = JointPrior(prior_x, prior_u)

        E0 = joint.E0
        assert E0.shape == (4, 8)  # (d_x + d_u, D_x + D_u)

    def test_E0_x_and_E0_hidden_multidimensional(self):
        """Test E0_x and E0_hidden with multi-dimensional states."""
        prior_x = IWP(q=1, d=2)  # D_x = 4, d_x = 2
        prior_u = IWP(q=1, d=3)  # D_u = 6, d_u = 3
        joint = JointPrior(prior_x, prior_u)

        E0_x = joint.E0_x
        E0_hidden = joint.E0_hidden
        assert E0_x.shape == (2, 10)  # (d_x, D_x + D_u)
        assert E0_hidden.shape == (3, 10)  # (d_u, D_x + D_u)

        # State: [x1, x2, x1', x2', u1, u2, u3, u1', u2', u3']
        state = np.arange(1.0, 11.0)
        x = E0_x @ state
        u = E0_hidden @ state

        # x should be [x1, x2] = [1, 2]
        assert np.allclose(x, np.array([1.0, 2.0]))
        # u should be [u1, u2, u3] = [5, 6, 7]
        assert np.allclose(u, np.array([5.0, 6.0, 7.0]))


class TestJointPriorIntegration:
    """Integration tests for JointPrior."""

    def test_joint_prior_preserves_individual_dynamics(self):
        """Test that joint prior preserves individual prior dynamics."""
        prior_x = IWP(q=1, d=1)
        prior_u = IWP(q=1, d=1)
        joint = JointPrior(prior_x, prior_u)

        h = 0.5

        # Individual predictions
        state_x = np.array([1.0, 0.5])
        state_u = np.array([2.0, -0.5])

        pred_x = prior_x.A(h) @ state_x
        pred_u = prior_u.A(h) @ state_u

        # Joint prediction
        joint_state = np.concatenate([state_x, state_u])
        joint_pred = joint.A(h) @ joint_state

        # Should match
        expected = np.concatenate([pred_x, pred_u])
        assert np.allclose(joint_pred, expected)
