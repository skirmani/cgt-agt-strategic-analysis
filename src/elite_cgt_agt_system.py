#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ELITE COMPUTATIONAL & ALGORITHMIC GAME THEORY ANALYSIS SYSTEM
================================================================================

A production-grade framework for analyzing complex strategic games in:
- US-China Grand Strategy (Trade, Technology, Chips, Commodities)
- Precious Metals Markets (Gold, Silver, De-dollarization)
- Geopolitical Resource Competition
- Multi-Actor Political Economy

Core Mathematical Foundations:
1. Nash Equilibrium Computation (N-player, Mixed Strategies)
2. Evolutionary Stable Strategies (Replicator Dynamics)
3. Mean Field Game Approximations (Large Population Limits)
4. Bayesian Belief Updating (Scenario Trees)
5. Mechanism Design & Contract Theory
6. Causal Graph Propagation (Nth-order Effects)

Reference: "The Strategic Architecture of Computational and Algorithmic Game Theory"

Author: Elite Quant Research Team
Version: 1.0.0
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from scipy import optimize
from scipy.linalg import eig
from scipy.special import softmax

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ELITE_CGT")


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

PayoffMatrix = np.ndarray
StrategyProfile = Dict[str, np.ndarray]
BeliefVector = np.ndarray
T = TypeVar("T")


# =============================================================================
# CORE ENUMS
# =============================================================================


class GameType(Enum):
    """Classification of strategic game types."""

    NORMAL_FORM = auto()          # Simultaneous moves, payoff matrices
    EXTENSIVE_FORM = auto()       # Sequential moves, game trees
    REPEATED = auto()             # Iterated games with memory
    STOCHASTIC = auto()           # Markov games with state transitions
    MEAN_FIELD = auto()           # Large population approximations
    SIGNALING = auto()            # Information asymmetry games
    MECHANISM_DESIGN = auto()     # Principal-agent contract design
    EVOLUTIONARY = auto()         # Population dynamics


class EquilibriumConcept(Enum):
    """Solution concepts for strategic games."""

    NASH = auto()                 # No unilateral deviation incentive
    SUBGAME_PERFECT = auto()      # Nash in every subgame
    BAYESIAN_NASH = auto()        # Nash with incomplete information
    SEQUENTIAL = auto()           # Consistent beliefs at information sets
    CORRELATED = auto()           # Public signal coordination
    EVOLUTIONARY_STABLE = auto()  # Invasion-resistant strategies
    MEAN_FIELD = auto()           # Fixed point of population distribution
    TREMBLING_HAND = auto()       # Robust to small perturbations


class PlayerType(Enum):
    """Categories of strategic actors."""

    SOVEREIGN_STATE = auto()      # Nation-states, central banks
    INSTITUTIONAL = auto()        # Hedge funds, pension funds, SWF
    ALGORITHMIC = auto()          # HFT, systematic traders
    RETAIL = auto()               # Individual investors
    MARKET_MAKER = auto()         # Liquidity providers
    REGULATOR = auto()            # Policy makers, central banks


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


@dataclass
class Player:
    """
    Represents a strategic actor in a game.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        player_type: Category of player
        action_set: Available actions/strategies
        payoff_function: Maps outcomes to utilities
        beliefs: Probability distribution over states/types
        sophistication: Strategic reasoning depth (0-1)
        capital: Economic weight/influence
        time_horizon: Investment/planning horizon in days
    """

    id: str
    name: str
    player_type: PlayerType
    action_set: List[str]
    payoff_function: Optional[Callable] = None
    beliefs: Optional[BeliefVector] = None
    sophistication: float = 0.5
    capital: float = 1.0
    time_horizon: int = 365
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_expected_utility(
        self,
        action: str,
        opponent_strategy: np.ndarray,
        payoff_matrix: PayoffMatrix
    ) -> float:
        """Compute expected utility given opponent's mixed strategy."""
        action_idx = self.action_set.index(action)
        return float(np.dot(payoff_matrix[action_idx], opponent_strategy))


@dataclass
class GameState:
    """
    Represents the current state of a strategic game.

    Attributes:
        state_id: Unique identifier
        players: List of active players
        history: Sequence of past actions
        information_sets: Player-specific observable histories
        public_signals: Commonly observed information
        payoffs_realized: Accumulated payoffs per player
    """

    state_id: str
    players: List[Player]
    history: List[Tuple[str, str]] = field(default_factory=list)  # (player_id, action)
    information_sets: Dict[str, List[str]] = field(default_factory=dict)
    public_signals: List[str] = field(default_factory=list)
    payoffs_realized: Dict[str, float] = field(default_factory=dict)

    def add_action(self, player_id: str, action: str, public: bool = True):
        """Record an action in the game history."""
        self.history.append((player_id, action))
        if public:
            self.public_signals.append(f"{player_id}:{action}")
        # Update information sets
        for pid in self.information_sets:
            if pid == player_id or public:
                self.information_sets[pid].append(f"{player_id}:{action}")


@dataclass
class ScenarioBranch:
    """
    Represents a scenario in a Bayesian scenario tree.

    Attributes:
        name: Scenario identifier
        description: Human-readable narrative
        prior: Prior probability
        posterior: Posterior after evidence
        horizon: Time horizon descriptor
        nth_order_effects: Causal propagation effects
        confirming_catalysts: Events that increase probability
        killer_catalysts: Events that decrease probability
        trade_implications: Actionable trade ideas
    """

    name: str
    description: str
    prior: float
    posterior: float = 0.0
    bayes_log_lr: float = 0.0
    horizon: str = "quarters"
    nth_order_effects: List[Dict[str, Any]] = field(default_factory=list)
    confirming_catalysts: List[str] = field(default_factory=list)
    killer_catalysts: List[str] = field(default_factory=list)
    trade_implications: List[str] = field(default_factory=list)

    def update_posterior(self, log_likelihood_ratio: float):
        """Bayesian update of posterior probability."""
        self.bayes_log_lr = log_likelihood_ratio
        # Soft Bayes: posterior propto prior * exp(log_lr)
        self.posterior = self.prior * math.exp(log_likelihood_ratio)


# =============================================================================
# NASH EQUILIBRIUM COMPUTATION
# =============================================================================


class NashEquilibriumSolver:
    """
    Computes Nash Equilibria for N-player normal form games.

    Implements multiple algorithms:
    1. Support Enumeration (exact, exponential)
    2. Lemke-Howson (2-player, polynomial)
    3. Replicator Dynamics (approximate, any N)
    4. Fictitious Play (approximate, any N)

    Reference: PPAD-completeness results (Daskalakis et al.)
    """

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 10000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve_two_player_bimatrix(
        self,
        payoff_A: PayoffMatrix,
        payoff_B: PayoffMatrix,
        method: str = "support_enumeration"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Nash equilibrium for 2-player game.

        Args:
            payoff_A: Row player's payoff matrix (m x n)
            payoff_B: Column player's payoff matrix (m x n)
            method: Algorithm selection

        Returns:
            Tuple of mixed strategy vectors (sigma_A, sigma_B)
        """
        if method == "support_enumeration":
            return self._support_enumeration(payoff_A, payoff_B)
        elif method == "replicator":
            return self._replicator_dynamics_2p(payoff_A, payoff_B)
        elif method == "fictitious_play":
            return self._fictitious_play_2p(payoff_A, payoff_B)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _support_enumeration(
        self,
        A: PayoffMatrix,
        B: PayoffMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact Nash equilibrium via support enumeration.

        For each possible support pair, solve linear indifference conditions.
        Exponential in worst case but exact.
        """
        m, n = A.shape

        best_nash = None
        best_welfare = -np.inf

        # Enumerate all possible support combinations
        from itertools import combinations

        for k in range(1, min(m, n) + 1):
            for support_A in combinations(range(m), k):
                for support_B in combinations(range(n), k):
                    try:
                        sigma_A, sigma_B = self._solve_support(
                            A, B, list(support_A), list(support_B)
                        )

                        if sigma_A is not None and self._is_valid_equilibrium(
                            A, B, sigma_A, sigma_B, list(support_A), list(support_B)
                        ):
                            welfare = (
                                np.dot(sigma_A, A @ sigma_B) +
                                np.dot(sigma_A, B @ sigma_B)
                            )
                            if welfare > best_welfare:
                                best_welfare = welfare
                                best_nash = (sigma_A, sigma_B)
                    except Exception:
                        continue

        if best_nash is None:
            # Fallback to uniform mixed
            logger.warning("Support enumeration failed, using uniform mixed")
            return np.ones(m) / m, np.ones(n) / n

        return best_nash

    def _solve_support(
        self,
        A: PayoffMatrix,
        B: PayoffMatrix,
        support_A: List[int],
        support_B: List[int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for mixed strategy given supports."""
        k = len(support_A)
        if len(support_B) != k:
            return None, None

        m, n = A.shape

        # Indifference conditions for player A
        # All actions in support must give equal expected utility
        # A[i] @ sigma_B = v_A for all i in support_A

        # Build system: [A_support; 1'] @ sigma_B = [v_A * 1; 1]
        A_sub = A[support_A][:, support_B]
        B_sub = B[support_A][:, support_B].T

        # Augmented system for player B's strategy
        if k > 1:
            # Use first k-1 indifference conditions + probability constraint
            constraints_B = A_sub[:-1] - A_sub[1:]  # Indifference
            rhs_B = np.zeros(k - 1)

            # Add probability constraint
            full_A = np.vstack([constraints_B, np.ones(k)])
            full_b = np.concatenate([rhs_B, [1.0]])

            try:
                sigma_B_support = np.linalg.solve(full_A, full_b)
            except np.linalg.LinAlgError:
                return None, None
        else:
            sigma_B_support = np.array([1.0])

        # Similarly for player A's strategy using B's indifference
        if k > 1:
            constraints_A = B_sub[:-1] - B_sub[1:]
            rhs_A = np.zeros(k - 1)
            full_A_A = np.vstack([constraints_A, np.ones(k)])
            full_b_A = np.concatenate([rhs_A, [1.0]])

            try:
                sigma_A_support = np.linalg.solve(full_A_A, full_b_A)
            except np.linalg.LinAlgError:
                return None, None
        else:
            sigma_A_support = np.array([1.0])

        # Check non-negativity
        if np.any(sigma_A_support < -self.tolerance) or np.any(sigma_B_support < -self.tolerance):
            return None, None

        # Clip small negatives
        sigma_A_support = np.maximum(sigma_A_support, 0)
        sigma_B_support = np.maximum(sigma_B_support, 0)

        # Normalize
        sigma_A_support /= sigma_A_support.sum() + 1e-12
        sigma_B_support /= sigma_B_support.sum() + 1e-12

        # Expand to full strategy space
        sigma_A = np.zeros(m)
        sigma_B = np.zeros(n)
        sigma_A[support_A] = sigma_A_support
        sigma_B[support_B] = sigma_B_support

        return sigma_A, sigma_B

    def _is_valid_equilibrium(
        self,
        A: PayoffMatrix,
        B: PayoffMatrix,
        sigma_A: np.ndarray,
        sigma_B: np.ndarray,
        support_A: List[int],
        support_B: List[int]
    ) -> bool:
        """Verify equilibrium conditions."""
        m, n = A.shape

        # Expected payoffs
        util_A = A @ sigma_B
        util_B = (sigma_A @ B)

        max_util_A = util_A.max()
        max_util_B = util_B.max()

        # All supported actions should be best responses
        for i in support_A:
            if util_A[i] < max_util_A - self.tolerance:
                return False

        for j in support_B:
            if util_B[j] < max_util_B - self.tolerance:
                return False

        return True

    def _replicator_dynamics_2p(
        self,
        A: PayoffMatrix,
        B: PayoffMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate Nash via continuous-time replicator dynamics.

        dx_i/dt = x_i * (u_i(x,y) - u_avg(x,y))

        Converges to Nash equilibrium in many games.
        """
        m, n = A.shape

        # Initialize with uniform + small perturbation
        x = np.ones(m) / m + np.random.randn(m) * 0.01
        y = np.ones(n) / n + np.random.randn(n) * 0.01
        x = np.maximum(x, 1e-10)
        y = np.maximum(y, 1e-10)
        x /= x.sum()
        y /= y.sum()

        dt = 0.01

        for _ in range(self.max_iterations):
            # Payoffs
            u_A = A @ y
            u_B = B.T @ x

            # Average payoffs
            u_avg_A = np.dot(x, u_A)
            u_avg_B = np.dot(y, u_B)

            # Replicator update
            dx = x * (u_A - u_avg_A)
            dy = y * (u_B - u_avg_B)

            x_new = x + dt * dx
            y_new = y + dt * dy

            # Project to simplex
            x_new = np.maximum(x_new, 1e-10)
            y_new = np.maximum(y_new, 1e-10)
            x_new /= x_new.sum()
            y_new /= y_new.sum()

            # Check convergence
            if np.max(np.abs(x_new - x)) < self.tolerance and \
               np.max(np.abs(y_new - y)) < self.tolerance:
                break

            x, y = x_new, y_new

        return x, y

    def _fictitious_play_2p(
        self,
        A: PayoffMatrix,
        B: PayoffMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate Nash via fictitious play.

        Each player best-responds to empirical distribution of opponent's play.
        """
        m, n = A.shape

        # Empirical action counts
        count_A = np.ones(m)  # Uniform prior
        count_B = np.ones(n)

        for t in range(1, self.max_iterations + 1):
            # Empirical frequencies
            freq_A = count_A / count_A.sum()
            freq_B = count_B / count_B.sum()

            # Best responses
            br_A = np.argmax(A @ freq_B)
            br_B = np.argmax(B.T @ freq_A)

            # Update counts
            count_A[br_A] += 1
            count_B[br_B] += 1

            # Check approximate convergence
            if t > 100:
                old_freq_A = (count_A - 1) / (count_A.sum() - m)
                old_freq_B = (count_B - 1) / (count_B.sum() - n)
                if np.max(np.abs(freq_A - old_freq_A)) < self.tolerance:
                    break

        return count_A / count_A.sum(), count_B / count_B.sum()

    def solve_n_player(
        self,
        payoffs: List[np.ndarray],
        players: List[Player]
    ) -> StrategyProfile:
        """
        Solve N-player normal form game via iterated replicator dynamics.

        Args:
            payoffs: List of N payoff tensors, each of shape (a1, a2, ..., aN)
            players: List of N players

        Returns:
            Strategy profile mapping player IDs to mixed strategies
        """
        n_players = len(players)
        action_sizes = [len(p.action_set) for p in players]

        # Initialize strategies
        strategies = {
            p.id: np.ones(action_sizes[i]) / action_sizes[i]
            for i, p in enumerate(players)
        }

        for _ in range(self.max_iterations):
            new_strategies = {}
            max_change = 0.0

            for i, player in enumerate(players):
                # Compute expected payoff for each action
                payoff_i = payoffs[i]

                # Marginalize over other players' strategies
                expected = np.zeros(action_sizes[i])

                for a_i in range(action_sizes[i]):
                    # Index into payoff tensor
                    idx = [slice(None)] * n_players
                    idx[i] = a_i

                    # Expected value over opponent strategies
                    sub_payoff = payoff_i[tuple(idx)]

                    # Compute expectation by iterating over opponent action profiles
                    exp_val = self._compute_expected_payoff(
                        sub_payoff, strategies, players, i
                    )
                    expected[a_i] = exp_val

                # Replicator update
                avg_payoff = np.dot(strategies[player.id], expected)
                delta = strategies[player.id] * (expected - avg_payoff)

                new_strat = strategies[player.id] + 0.1 * delta
                new_strat = np.maximum(new_strat, 1e-10)
                new_strat /= new_strat.sum()

                change = np.max(np.abs(new_strat - strategies[player.id]))
                max_change = max(max_change, change)
                new_strategies[player.id] = new_strat

            strategies = new_strategies

            if max_change < self.tolerance:
                break

        return strategies

    def _compute_expected_payoff(
        self,
        payoff_slice: np.ndarray,
        strategies: StrategyProfile,
        players: List[Player],
        exclude_idx: int
    ) -> float:
        """Compute expected payoff by averaging over opponent strategies."""
        # This is a simplified implementation
        # For full generality, use tensor operations

        other_players = [p for i, p in enumerate(players) if i != exclude_idx]

        if len(other_players) == 0:
            return float(payoff_slice)

        # Compute product of opponent strategies
        result = payoff_slice.copy()

        for j, player in enumerate(other_players):
            strat = strategies[player.id]
            # Contract along this player's dimension
            result = np.tensordot(result, strat, axes=([0], [0]))

        return float(result)


# =============================================================================
# EVOLUTIONARY GAME THEORY
# =============================================================================


class EvolutionaryDynamics:
    """
    Implements evolutionary game theory dynamics.

    Models strategy adaptation in populations via:
    1. Replicator Dynamics (imitation of successful strategies)
    2. Best Response Dynamics (rational updating)
    3. Logit Dynamics (noisy best response)
    4. Smith Dynamics (direct payoff comparison)

    Key concept: Evolutionarily Stable Strategy (ESS)
    - A strategy that cannot be invaded by mutants
    """

    def __init__(self, dt: float = 0.01, noise_level: float = 0.0):
        self.dt = dt
        self.noise_level = noise_level

    def replicator_dynamics(
        self,
        payoff_matrix: PayoffMatrix,
        initial_distribution: np.ndarray,
        time_steps: int = 1000
    ) -> List[np.ndarray]:
        """
        Simulate continuous-time replicator dynamics.

        dx_i/dt = x_i * (f_i(x) - phi(x))

        where f_i is fitness of strategy i and phi is average fitness.

        Args:
            payoff_matrix: Symmetric payoff matrix (A vs A)
            initial_distribution: Starting population shares
            time_steps: Number of simulation steps

        Returns:
            Trajectory of population distributions
        """
        x = initial_distribution.copy()
        trajectory = [x.copy()]

        for _ in range(time_steps):
            # Fitness vector
            fitness = payoff_matrix @ x

            # Average fitness
            avg_fitness = np.dot(x, fitness)

            # Replicator update
            dx = x * (fitness - avg_fitness)

            # Add mutation/noise
            if self.noise_level > 0:
                dx += self.noise_level * (np.ones_like(x) / len(x) - x)

            x = x + self.dt * dx

            # Project to simplex
            x = np.maximum(x, 0)
            x /= x.sum()

            trajectory.append(x.copy())

        return trajectory

    def find_ess(
        self,
        payoff_matrix: PayoffMatrix
    ) -> List[Tuple[int, float]]:
        """
        Find Evolutionarily Stable Strategies.

        A strategy i is an ESS if:
        1. A[i,i] >= A[j,i] for all j (Nash condition)
        2. If A[i,i] = A[j,i], then A[i,j] > A[j,j] (stability condition)

        Returns:
            List of (strategy_index, stability_score) tuples
        """
        n = payoff_matrix.shape[0]
        ess_candidates = []

        for i in range(n):
            is_ess = True
            stability_score = 1.0

            for j in range(n):
                if i == j:
                    continue

                # Nash condition
                if payoff_matrix[i, i] < payoff_matrix[j, i] - 1e-10:
                    is_ess = False
                    break

                # Stability condition (when Nash is equality)
                if abs(payoff_matrix[i, i] - payoff_matrix[j, i]) < 1e-10:
                    if payoff_matrix[i, j] <= payoff_matrix[j, j]:
                        is_ess = False
                        break
                    stability_score *= (payoff_matrix[i, j] - payoff_matrix[j, j]) / (
                        abs(payoff_matrix[i, j]) + abs(payoff_matrix[j, j]) + 1e-10
                    )

            if is_ess:
                ess_candidates.append((i, stability_score))

        return ess_candidates

    def invasion_analysis(
        self,
        payoff_matrix: PayoffMatrix,
        resident_strategy: int,
        mutant_strategy: int,
        initial_mutant_share: float = 0.01,
        time_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Analyze whether a mutant can invade a resident population.

        Returns:
            Dictionary with invasion dynamics metrics
        """
        n = payoff_matrix.shape[0]

        # Initial distribution: mostly resident, small mutant share
        x = np.zeros(n)
        x[resident_strategy] = 1.0 - initial_mutant_share
        x[mutant_strategy] = initial_mutant_share

        trajectory = self.replicator_dynamics(payoff_matrix, x, time_steps)

        final_dist = trajectory[-1]

        # Compute invasion fitness
        # When mutant is rare: f_mutant - f_resident
        invasion_fitness = (
            payoff_matrix[mutant_strategy, resident_strategy] -
            payoff_matrix[resident_strategy, resident_strategy]
        )

        return {
            "invasion_fitness": invasion_fitness,
            "mutant_can_invade": invasion_fitness > 0,
            "final_mutant_share": final_dist[mutant_strategy],
            "final_resident_share": final_dist[resident_strategy],
            "trajectory_length": len(trajectory),
            "converged": np.max(np.abs(trajectory[-1] - trajectory[-10])) < 1e-6
        }


# =============================================================================
# MEAN FIELD GAME FRAMEWORK
# =============================================================================


class MeanFieldGame:
    """
    Mean Field Game approximation for large population games.

    When N → ∞, individual interactions replaced by interaction with
    aggregate distribution. Solves coupled HJB-FPK system:

    1. Hamilton-Jacobi-Bellman (backward): Optimal control
    2. Fokker-Planck-Kolmogorov (forward): Population density evolution

    Applications:
    - Market microstructure with many traders
    - Crowd dynamics in financial markets
    - Optimal execution with market impact
    """

    def __init__(
        self,
        state_dim: int = 1,
        action_dim: int = 1,
        time_horizon: float = 1.0,
        n_time_steps: int = 100,
        n_state_points: int = 50
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = time_horizon
        self.n_t = n_time_steps
        self.n_x = n_state_points

        self.dt = time_horizon / n_time_steps

    def solve_linear_quadratic(
        self,
        Q: float,    # State cost
        R: float,    # Control cost
        sigma: float,  # Volatility
        interaction_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Solve linear-quadratic mean field game.

        Agent dynamics: dx = u dt + sigma dW
        Running cost: (x - m)^2 * Q/2 + u^2 * R/2

        where m is population mean (mean field).

        This has a closed-form solution via Riccati equations.
        """
        # Riccati equation coefficients
        # P' = -Q + P^2 / R (backward)
        # m' = -P * m / R (forward mean dynamics)

        t_grid = np.linspace(0, self.T, self.n_t + 1)

        # Solve Riccati backward
        P = np.zeros(self.n_t + 1)
        P[-1] = 0  # Terminal condition

        for i in range(self.n_t - 1, -1, -1):
            # Implicit Euler for stability
            P[i] = P[i + 1] + self.dt * (-Q + P[i + 1]**2 / R)

        # Solve mean dynamics forward
        m = np.zeros(self.n_t + 1)
        m[0] = 0  # Initial mean (centered)

        for i in range(self.n_t):
            m[i + 1] = m[i] - self.dt * P[i] * m[i] / R

        # Optimal control: u* = -P * (x - m) / R
        # Value function: V(x,t) = P(t) * (x - m(t))^2 / 2

        return {
            "riccati_solution": P,
            "mean_field_trajectory": m,
            "time_grid": t_grid,
            "optimal_control_gain": -P / R,
            "equilibrium_type": "Mean Field Nash Equilibrium"
        }

    def simulate_population(
        self,
        n_agents: int,
        initial_positions: np.ndarray,
        optimal_control_gain: np.ndarray,
        mean_trajectory: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        Simulate agent trajectories under MFG equilibrium controls.

        Returns:
            Array of shape (n_agents, n_time_steps + 1)
        """
        trajectories = np.zeros((n_agents, self.n_t + 1))
        trajectories[:, 0] = initial_positions

        for t in range(self.n_t):
            x = trajectories[:, t]
            m = mean_trajectory[t]

            # Optimal control
            u = optimal_control_gain[t] * (x - m)

            # Dynamics with noise
            dW = np.random.randn(n_agents) * np.sqrt(self.dt)
            trajectories[:, t + 1] = x + u * self.dt + sigma * dW

        return trajectories


# =============================================================================
# BAYESIAN SCENARIO ENGINE
# =============================================================================


class BayesianScenarioEngine:
    """
    Probabilistic scenario analysis with Bayesian updating.

    Features:
    1. Prior scenario tree construction
    2. Evidence-based posterior updates
    3. Nth-order causal propagation
    4. Market consistency checks
    """

    def __init__(self, scenarios: List[ScenarioBranch]):
        self.scenarios = scenarios
        self._normalize_priors()

    def _normalize_priors(self):
        """Ensure priors sum to 1."""
        total = sum(s.prior for s in self.scenarios)
        for s in self.scenarios:
            s.prior /= total
            s.posterior = s.prior  # Initialize posterior = prior

    def soft_bayes_update(
        self,
        log_likelihood_ratios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update posteriors using soft Bayes with log-likelihood ratios.

        P(scenario | evidence) ∝ P(scenario) * exp(log_LR)

        Args:
            log_likelihood_ratios: Map from scenario name to log-LR

        Returns:
            Updated posterior probabilities
        """
        posteriors_unnorm = []

        for scenario in self.scenarios:
            log_lr = log_likelihood_ratios.get(scenario.name, 0.0)
            scenario.bayes_log_lr = log_lr
            posteriors_unnorm.append(scenario.prior * math.exp(log_lr))

        # Normalize
        total = sum(posteriors_unnorm) + 1e-12
        posteriors = {}

        for scenario, p_unnorm in zip(self.scenarios, posteriors_unnorm):
            scenario.posterior = p_unnorm / total
            posteriors[scenario.name] = scenario.posterior

        return posteriors

    def compute_market_consistency_lr(
        self,
        scenario: ScenarioBranch,
        market_data: Dict[str, float],
        expected_directions: Dict[str, int]
    ) -> float:
        """
        Compute log-likelihood ratio based on market price consistency.

        If market prices move in direction predicted by scenario,
        increase likelihood; otherwise decrease.

        Args:
            scenario: The scenario to evaluate
            market_data: Current market returns (e.g., {"spx": -0.02, "oil": 0.05})
            expected_directions: Scenario's predicted directions (1=up, -1=down, 0=neutral)

        Returns:
            Log-likelihood ratio
        """
        log_lr = 0.0

        for asset, expected_dir in expected_directions.items():
            if expected_dir == 0 or asset not in market_data:
                continue

            actual_return = market_data[asset]

            # Agreement: same sign
            if expected_dir * actual_return > 0:
                magnitude = abs(actual_return)
                log_lr += 0.5 * magnitude * 10  # Scale factor
            elif expected_dir * actual_return < 0:
                magnitude = abs(actual_return)
                log_lr -= 0.3 * magnitude * 10

        return log_lr

    def get_probability_weighted_effects(
        self,
        causal_graph: "CausalGraph"
    ) -> Dict[str, float]:
        """
        Compute probability-weighted Nth-order effects across scenarios.

        Returns expected impact on each factor weighted by scenario probability.
        """
        aggregated_effects = {}

        for scenario in self.scenarios:
            for effect in scenario.nth_order_effects:
                factor = effect["factor"]
                score = effect["score"]
                direction = 1.0 if effect["direction"] == "UP" else -1.0

                weighted_impact = scenario.posterior * score * direction
                aggregated_effects[factor] = aggregated_effects.get(factor, 0.0) + weighted_impact

        return aggregated_effects


# =============================================================================
# CAUSAL GRAPH AND NTH-ORDER PROPAGATION
# =============================================================================


class CausalGraph:
    """
    Directed graph for modeling causal relationships between economic factors.

    Supports:
    1. Multi-hop propagation (Nth-order effects)
    2. Signed edge weights (positive/negative relationships)
    3. Decay factors across hops
    4. Cycle handling
    """

    # Default causal relationships for macro-financial analysis
    DEFAULT_EDGES = {
        "GeopoliticalRiskPremium": {
            "OilPrice": 0.70,
            "EquityVol": 0.65,
            "GoldPrice": 0.55,
            "USD": 0.40,
            "CreditSpreads": 0.35
        },
        "OilPrice": {
            "InflationExpectations": 0.55,
            "TransportationCosts": 0.50,
            "RefinerMargins": 0.35,
            "Equities": -0.25
        },
        "InflationExpectations": {
            "NominalRates": 0.45,
            "RealRates": -0.30,
            "GoldPrice": 0.35,
            "TIPS": 0.40
        },
        "NominalRates": {
            "GrowthEquities": -0.40,
            "Financials": 0.25,
            "RealEstate": -0.35,
            "USD": 0.30
        },
        "USD": {
            "EMFX": -0.50,
            "Commodities": -0.35,
            "EMEquities": -0.30,
            "GoldPrice": -0.25
        },
        "EquityVol": {
            "Equities": -0.50,
            "CreditSpreads": 0.40,
            "USD": 0.20,
            "Bonds": 0.15
        },
        "CreditSpreads": {
            "HighYield": -0.60,
            "Financials": -0.35,
            "RiskAssets": -0.40
        },
        "FedPolicy": {
            "NominalRates": 0.70,
            "USD": 0.45,
            "EquityVol": -0.20,
            "RiskAssets": 0.30
        },
        "ChipSupply": {
            "TechEquities": 0.50,
            "AIStocks": 0.60,
            "Semis": 0.70,
            "AutoSector": 0.30
        },
        "TradeWar": {
            "ChipSupply": -0.40,
            "GeopoliticalRiskPremium": 0.50,
            "GlobalTrade": -0.45,
            "SupplyChainCosts": 0.40
        }
    }

    def __init__(self, edges: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize causal graph.

        Args:
            edges: Dict mapping source -> {target: weight}
        """
        self.edges = edges if edges is not None else self.DEFAULT_EDGES.copy()
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency structures for efficient traversal."""
        self.nodes = set()
        self.outgoing = {}  # node -> [(neighbor, weight)]
        self.incoming = {}  # node -> [(neighbor, weight)]

        for source, targets in self.edges.items():
            self.nodes.add(source)
            if source not in self.outgoing:
                self.outgoing[source] = []

            for target, weight in targets.items():
                self.nodes.add(target)
                self.outgoing[source].append((target, weight))

                if target not in self.incoming:
                    self.incoming[target] = []
                self.incoming[target].append((source, weight))

    def propagate(
        self,
        root: str,
        direction: str,  # "UP" or "DOWN"
        max_depth: int = 3,
        decay_factor: float = 0.55,
        threshold: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Compute Nth-order effects from a root shock.

        Uses BFS with decay to compute transitive effects.

        Args:
            root: Starting node for shock
            direction: "UP" or "DOWN" for initial shock direction
            max_depth: Maximum propagation depth
            decay_factor: Multiplicative decay per hop
            threshold: Minimum effect magnitude to include

        Returns:
            List of effect dictionaries sorted by magnitude
        """
        sign = 1.0 if direction.upper() == "UP" else -1.0

        # BFS with decay
        effects: Dict[str, float] = {}
        frontier: List[Tuple[str, float, int]] = [(root, sign, 0)]
        visited_at_depth: Dict[str, int] = {root: 0}

        while frontier:
            node, effect_magnitude, depth = frontier.pop(0)

            if depth >= max_depth:
                continue

            # Propagate to neighbors
            for neighbor, edge_weight in self.outgoing.get(node, []):
                new_effect = effect_magnitude * edge_weight * decay_factor

                if abs(new_effect) < threshold:
                    continue

                # Accumulate effects (can receive from multiple paths)
                effects[neighbor] = effects.get(neighbor, 0.0) + new_effect

                # Add to frontier if not visited at lower depth
                if neighbor not in visited_at_depth or visited_at_depth[neighbor] > depth + 1:
                    visited_at_depth[neighbor] = depth + 1
                    frontier.append((neighbor, new_effect, depth + 1))

        # Format results
        results = []
        for factor, score in sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True):
            results.append({
                "factor": factor,
                "direction": "UP" if score > 0 else "DOWN",
                "score": abs(score),
                "raw_score": score
            })

        return results[:15]  # Top 15 effects

    def add_edge(self, source: str, target: str, weight: float):
        """Add or update an edge in the causal graph."""
        if source not in self.edges:
            self.edges[source] = {}
        self.edges[source][target] = weight
        self._build_adjacency()

    def get_upstream_factors(self, target: str, depth: int = 2) -> List[str]:
        """Find factors that causally influence target."""
        upstream = set()
        frontier = [target]

        for _ in range(depth):
            new_frontier = []
            for node in frontier:
                for source, _ in self.incoming.get(node, []):
                    if source not in upstream:
                        upstream.add(source)
                        new_frontier.append(source)
            frontier = new_frontier

        return list(upstream)


# =============================================================================
# STRATEGIC GAME TEMPLATES
# =============================================================================


class USChinaGrandStrategyGame:
    """
    Models US-China strategic competition across multiple domains.

    Domains:
    1. Technology/Chips (semiconductor supply chains)
    2. Trade (tariffs, exports)
    3. Finance (de-dollarization, capital controls)
    4. Geopolitics (Taiwan, South China Sea)
    5. Resources (rare earths, energy)

    Game structure: Multi-domain repeated game with linkages
    """

    def __init__(self):
        # Define players
        self.us = Player(
            id="USA",
            name="United States",
            player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["cooperate", "compete", "confront", "contain"],
            sophistication=0.85,
            capital=25e12,  # GDP
            time_horizon=1460  # 4 years (election cycle)
        )

        self.china = Player(
            id="CHN",
            name="People's Republic of China",
            player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["cooperate", "compete", "confront", "decouple"],
            sophistication=0.90,
            capital=18e12,  # GDP
            time_horizon=3650  # 10 years (5-year plans)
        )

        # Domain-specific payoff matrices (US payoff, China payoff)
        self._build_payoff_matrices()

        # Causal linkages between domains
        self.causal_graph = CausalGraph()
        self._add_domain_linkages()

    def _build_payoff_matrices(self):
        """
        Build payoff matrices for each strategic domain.

        These matrices are calibrated to reflect realistic strategic tensions:
        - US benefits from cooperation in tech (access to Chinese market/manufacturing)
        - China benefits from competition (forced technology transfer, catch-up)
        - Confrontation is costly for both but creates mixed incentives
        - Current geopolitical environment favors "compete" equilibrium
        """

        # Technology/Chips Domain
        # Rows: US actions, Cols: China actions
        # Payoffs: normalized to [-10, 10] scale
        #
        # Key dynamics:
        # - US has semiconductor design advantage
        # - China has manufacturing scale
        # - Decoupling is mutually costly but China has more to lose short-term
        self.tech_payoffs_us = np.array([
            # CHN: coop    compete  confront  decouple
            [  5.0,     2.0,    -3.0,    -5.0],   # US: cooperate - best if mutual, exploited if alone
            [  6.0,     3.0,     0.0,    -2.0],   # US: compete - dominant strategy currently
            [  4.0,     4.0,    -2.0,    -3.0],   # US: confront - costly but maintains leverage
            [  1.0,     2.0,     1.0,    -4.0],   # US: contain - painful but may be necessary
        ])

        self.tech_payoffs_china = np.array([
            # CHN: coop    compete  confront  decouple
            [  4.0,     6.0,     3.0,     0.0],   # US: cooperate - China exploits openness
            [  2.0,     4.0,     3.0,     1.0],   # US: compete - China can still compete
            [ -2.0,     2.0,     1.0,     2.0],   # US: confront - China pivots to self-reliance
            [ -5.0,    -1.0,     0.0,     3.0],   # US: contain - decoupling becomes attractive
        ])

        # Trade Domain
        # Key dynamics:
        # - Bilateral trade worth $700B+
        # - US consumer benefits from cheap imports
        # - China export sector depends on US market
        # - Tariff wars are lose-lose but politically expedient
        self.trade_payoffs_us = np.array([
            # CHN: coop    compete  confront  decouple
            [  5.0,     3.0,    -2.0,    -6.0],   # US: cooperate
            [  4.0,     2.0,    -1.0,    -4.0],   # US: compete - current equilibrium
            [  1.0,     1.0,    -3.0,    -5.0],   # US: confront - tariffs hurt consumers
            [ -2.0,    -1.0,    -4.0,    -7.0],   # US: contain - near-shoring costs
        ])

        self.trade_payoffs_china = np.array([
            # CHN: coop    compete  confront  decouple
            [  6.0,     4.0,     1.0,    -3.0],   # US: cooperate - China gains most from open trade
            [  4.0,     3.0,     1.0,    -2.0],   # US: compete
            [  2.0,     2.0,     0.0,     0.0],   # US: confront - China can absorb tariffs
            [ -1.0,     0.0,    -1.0,     1.0],   # US: contain - China pivots to ASEAN/EU
        ])

        # Finance/Currency Domain
        # Key dynamics:
        # - USD reserve currency status is US's key advantage
        # - China wants yuan internationalization
        # - De-dollarization is a long-term China goal
        # - US sanctions power depends on dollar dominance
        self.finance_payoffs_us = np.array([
            # CHN: coop    compete  confront  decouple
            [  7.0,     5.0,     2.0,    -2.0],   # US: cooperate - dollar dominance maintained
            [  6.0,     4.0,     2.0,    -1.0],   # US: compete - status quo favorable
            [  3.0,     3.0,     0.0,    -3.0],   # US: confront - sanctions have costs
            [  0.0,     1.0,    -1.0,    -5.0],   # US: contain - accelerates de-dollarization
        ])

        self.finance_payoffs_china = np.array([
            # CHN: coop    compete  confront  decouple
            [  2.0,     3.0,     4.0,     5.0],   # US: cooperate - China wants alternatives
            [  1.0,     2.0,     3.0,     4.0],   # US: compete - gradual de-dollarization
            [ -1.0,     1.0,     2.0,     4.0],   # US: confront - forces China to alternatives
            [ -3.0,     0.0,     2.0,     5.0],   # US: contain - decoupling serves China long-term
        ])

    def _add_domain_linkages(self):
        """Add causal links between strategic domains and market factors."""

        # Technology restrictions → Supply chain effects
        self.causal_graph.add_edge("TechWar", "ChipSupply", -0.60)
        self.causal_graph.add_edge("TechWar", "AIStocks", -0.40)
        self.causal_graph.add_edge("TechWar", "SupplyChainCosts", 0.50)

        # Trade conflict → Economic effects
        self.causal_graph.add_edge("TradeWar", "GlobalTrade", -0.55)
        self.causal_graph.add_edge("TradeWar", "InflationExpectations", 0.30)
        self.causal_graph.add_edge("TradeWar", "ConsumerGoods", -0.25)

        # Financial decoupling → Currency/flow effects
        self.causal_graph.add_edge("FinancialDecoupling", "USD", 0.20)
        self.causal_graph.add_edge("FinancialDecoupling", "CNY", -0.30)
        self.causal_graph.add_edge("FinancialDecoupling", "GoldPrice", 0.25)
        self.causal_graph.add_edge("FinancialDecoupling", "EMFlows", -0.35)

    def compute_equilibrium(
        self,
        domain: str = "tech",
        method: str = "replicator"
    ) -> Dict[str, Any]:
        """
        Compute Nash equilibrium for specified domain.

        Args:
            domain: One of "tech", "trade", "finance"
            method: Equilibrium computation method

        Returns:
            Equilibrium analysis results
        """
        if domain == "tech":
            payoff_us = self.tech_payoffs_us
            payoff_china = self.tech_payoffs_china
        elif domain == "trade":
            payoff_us = self.trade_payoffs_us
            payoff_china = self.trade_payoffs_china
        elif domain == "finance":
            payoff_us = self.finance_payoffs_us
            payoff_china = self.finance_payoffs_china
        else:
            raise ValueError(f"Unknown domain: {domain}")

        solver = NashEquilibriumSolver()
        sigma_us, sigma_china = solver.solve_two_player_bimatrix(
            payoff_us, payoff_china, method=method
        )

        # Compute expected payoffs
        exp_payoff_us = sigma_us @ payoff_us @ sigma_china
        exp_payoff_china = sigma_us @ payoff_china @ sigma_china

        # Most likely actions
        us_action = self.us.action_set[np.argmax(sigma_us)]
        china_action = self.china.action_set[np.argmax(sigma_china)]

        return {
            "domain": domain,
            "us_strategy": dict(zip(self.us.action_set, sigma_us)),
            "china_strategy": dict(zip(self.china.action_set, sigma_china)),
            "expected_payoff_us": float(exp_payoff_us),
            "expected_payoff_china": float(exp_payoff_china),
            "most_likely_us_action": us_action,
            "most_likely_china_action": china_action,
            "equilibrium_type": "Nash Equilibrium"
        }

    def analyze_scenario(
        self,
        us_action: str,
        china_action: str
    ) -> Dict[str, Any]:
        """
        Analyze market implications of a specific action profile.

        Returns Nth-order effects on asset classes.
        """
        # Map actions to shock nodes
        shock_mapping = {
            ("confront", "confront"): ("TradeWar", "UP"),
            ("confront", "decouple"): ("TechWar", "UP"),
            ("contain", "decouple"): ("FinancialDecoupling", "UP"),
            ("cooperate", "cooperate"): ("GlobalTrade", "UP"),
            ("compete", "compete"): ("GeopoliticalRiskPremium", "UP"),
        }

        key = (us_action, china_action)
        if key in shock_mapping:
            shock_node, direction = shock_mapping[key]
        else:
            shock_node, direction = "GeopoliticalRiskPremium", "UP"

        effects = self.causal_graph.propagate(shock_node, direction, max_depth=3)

        # Compute aggregate payoffs
        us_idx = self.us.action_set.index(us_action)
        china_idx = self.china.action_set.index(china_action)

        aggregate_us = (
            self.tech_payoffs_us[us_idx, china_idx] +
            self.trade_payoffs_us[us_idx, china_idx] +
            self.finance_payoffs_us[us_idx, china_idx]
        ) / 3

        aggregate_china = (
            self.tech_payoffs_china[us_idx, china_idx] +
            self.trade_payoffs_china[us_idx, china_idx] +
            self.finance_payoffs_china[us_idx, china_idx]
        ) / 3

        return {
            "us_action": us_action,
            "china_action": china_action,
            "us_aggregate_payoff": float(aggregate_us),
            "china_aggregate_payoff": float(aggregate_china),
            "primary_shock": shock_node,
            "shock_direction": direction,
            "nth_order_effects": effects,
            "winner": "US" if aggregate_us > aggregate_china else "China" if aggregate_china > aggregate_us else "Draw"
        }


class PreciousMetalsGame:
    """
    Models strategic dynamics in gold/silver markets.

    Players:
    1. Central Banks (PBOC, ECB, Fed, BOJ, RBI)
    2. Sovereign Wealth Funds
    3. Real Money (pensions, insurance)
    4. Algorithmic Traders
    5. Carry Traders

    Key dynamics:
    - De-dollarization coordination game
    - Real rate arbitrage
    - Session-based information asymmetry
    """

    def __init__(self):
        self._build_central_bank_game()
        self._build_trader_agents()
        self.causal_graph = CausalGraph()
        self._add_precious_metals_links()

    def _build_central_bank_game(self):
        """Build the N-player central bank coordination game."""

        # Central bank players
        self.pboc = Player(
            id="PBOC",
            name="People's Bank of China",
            player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["accelerate_gold", "maintain", "reduce"],
            sophistication=0.90,
            capital=3.5e12,  # Reserves
            time_horizon=3650,  # 10 years
            metadata={"gold_tonnes": 2200, "de_dollar_support": 0.85}
        )

        self.ecb = Player(
            id="ECB",
            name="European Central Bank",
            player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["coordinate_pboc", "neutral", "support_fed"],
            sophistication=0.80,
            capital=800e9,
            time_horizon=1825,  # 5 years
            metadata={"gold_tonnes": 500, "de_dollar_support": 0.25}
        )

        self.fed = Player(
            id="FED",
            name="Federal Reserve",
            player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["maintain_hawkish", "accommodate", "intervene"],
            sophistication=0.85,
            capital=8e12,
            time_horizon=730,  # 2 years
            metadata={"gold_tonnes": 8100, "de_dollar_support": 0.05}
        )

        # Coordination game payoff structure (Stag Hunt / Coordination Game)
        # Key dynamics:
        # - De-dollarization requires multi-party coordination
        # - Unilateral action is costly (sanctions risk)
        # - Status quo is safe but suboptimal for challengers
        # - ECB is a reluctant player (EU-US alliance)
        #
        # Current calibration reflects:
        # - PBOC has strong incentive to accumulate but needs cover
        # - ECB prefers neutrality but follows if coalition forms
        # - Coordination probability ~30-40% in current environment

        self.coordination_payoffs_pboc = np.array([
            # ECB: coord  neutral  support_fed
            [  8.0,    2.0,    -4.0],   # PBOC: accelerate - high risk/reward
            [  3.0,    4.0,     3.0],   # PBOC: maintain - safe middle ground
            [ -2.0,    1.0,     2.0],   # PBOC: reduce - capitulation
        ])

        self.coordination_payoffs_ecb = np.array([
            # ECB: coord  neutral  support_fed
            [  5.0,    3.0,    -2.0],   # PBOC: accelerate - ECB benefits if joins winning coalition
            [  2.0,    4.0,     4.0],   # PBOC: maintain - ECB prefers stability
            [ -1.0,    2.0,     5.0],   # PBOC: reduce - Fed dominance suits ECB-US relations
        ])

    def _build_trader_agents(self):
        """Build trading agent profiles."""

        self.sovereign_wealth = Player(
            id="SWF",
            name="Sovereign Wealth Funds",
            player_type=PlayerType.INSTITUTIONAL,
            action_set=["accumulate", "hold", "reduce"],
            sophistication=0.90,
            capital=500e9,
            time_horizon=1825
        )

        self.real_money = Player(
            id="REAL_MONEY",
            name="Pensions & Insurance",
            player_type=PlayerType.INSTITUTIONAL,
            action_set=["overweight_gold", "neutral", "underweight_gold"],
            sophistication=0.70,
            capital=250e9,
            time_horizon=365
        )

        self.algo_traders = Player(
            id="ALGO",
            name="Algorithmic Traders",
            player_type=PlayerType.ALGORITHMIC,
            action_set=["momentum_long", "neutral", "mean_revert_short"],
            sophistication=0.60,
            capital=50e9,
            time_horizon=1
        )

        self.carry_traders = Player(
            id="CARRY",
            name="Carry Traders",
            player_type=PlayerType.INSTITUTIONAL,
            action_set=["long_usd_short_gold", "neutral", "unwind"],
            sophistication=0.75,
            capital=150e9,
            time_horizon=30
        )

    def _add_precious_metals_links(self):
        """Add causal links specific to precious metals."""

        # De-dollarization effects
        self.causal_graph.add_edge("DeDollarization", "GoldPrice", 0.60)
        self.causal_graph.add_edge("DeDollarization", "USD", -0.50)
        self.causal_graph.add_edge("DeDollarization", "CNY", 0.30)

        # Real rate effects
        self.causal_graph.add_edge("RealRates", "GoldPrice", -0.70)
        self.causal_graph.add_edge("RealRates", "SilverPrice", -0.65)
        self.causal_graph.add_edge("RealRates", "USD", 0.40)

        # Risk premium effects
        self.causal_graph.add_edge("GeopoliticalRiskPremium", "GoldPrice", 0.55)
        self.causal_graph.add_edge("GeopoliticalRiskPremium", "SilverPrice", 0.45)

    def compute_cb_equilibrium(self) -> Dict[str, Any]:
        """Compute equilibrium in the central bank coordination game."""

        solver = NashEquilibriumSolver()

        sigma_pboc, sigma_ecb = solver.solve_two_player_bimatrix(
            self.coordination_payoffs_pboc,
            self.coordination_payoffs_ecb,
            method="replicator"
        )

        # Expected payoffs
        exp_pboc = sigma_pboc @ self.coordination_payoffs_pboc @ sigma_ecb
        exp_ecb = sigma_pboc @ self.coordination_payoffs_ecb @ sigma_ecb

        # Coordination probability
        coord_prob = sigma_pboc[0] * sigma_ecb[0]  # Both choose coordinating action

        return {
            "pboc_strategy": dict(zip(self.pboc.action_set, sigma_pboc)),
            "ecb_strategy": dict(zip(self.ecb.action_set, sigma_ecb)),
            "expected_payoff_pboc": float(exp_pboc),
            "expected_payoff_ecb": float(exp_ecb),
            "de_dollarization_coordination_prob": float(coord_prob),
            "equilibrium_assessment": self._assess_equilibrium(coord_prob)
        }

    def _assess_equilibrium(self, coord_prob: float) -> str:
        """Provide qualitative assessment of equilibrium state."""

        if coord_prob > 0.5:
            return "HIGH: De-dollarization coordination likely. Bullish gold long-term."
        elif coord_prob > 0.25:
            return "MEDIUM: Mixed signals. Gold faces two-way risk."
        else:
            return "LOW: USD dominance persists. Gold faces headwinds from real rates."

    def session_dynamics(
        self,
        session: str,  # "ASIA", "EURO", "US"
        cb_equilibrium: Dict[str, Any],
        real_rate_differential: float = 0.015,  # 1.5%
    ) -> Dict[str, Any]:
        """
        Model expected price dynamics for a trading session.

        Args:
            session: Trading session
            cb_equilibrium: Output from compute_cb_equilibrium()
            real_rate_differential: USD real rate advantage

        Returns:
            Session dynamics analysis
        """
        coord_prob = cb_equilibrium["de_dollarization_coordination_prob"]

        # Session-specific dynamics
        session_profiles = {
            "ASIA": {
                "liquidity": 0.3,
                "political_weight": 0.7,  # CB accumulation more visible
                "economic_weight": 0.3,
                "dominant_players": ["PBOC", "SWF"],
                "typical_flow": "accumulate"
            },
            "EURO": {
                "liquidity": 0.5,
                "political_weight": 0.5,
                "economic_weight": 0.5,
                "dominant_players": ["ECB", "REAL_MONEY"],
                "typical_flow": "mixed"
            },
            "US": {
                "liquidity": 0.8,
                "political_weight": 0.3,
                "economic_weight": 0.7,  # Real rates dominate
                "dominant_players": ["FED", "ALGO", "CARRY"],
                "typical_flow": "sell"
            }
        }

        profile = session_profiles.get(session, session_profiles["US"])

        # Expected price change formula
        political_impulse = coord_prob * profile["political_weight"] * 0.01  # Up to +1%
        economic_impulse = -real_rate_differential * profile["economic_weight"] * 0.5  # Rate drag

        # Liquidity impact (lower liquidity = higher impact)
        liquidity_multiplier = 1 / (profile["liquidity"] + 0.2)

        expected_return = (political_impulse + economic_impulse) * liquidity_multiplier

        return {
            "session": session,
            "expected_return": float(expected_return),
            "political_impulse": float(political_impulse),
            "economic_impulse": float(economic_impulse),
            "dominant_force": "political" if abs(political_impulse) > abs(economic_impulse) else "economic",
            "profile": profile
        }

    def generate_gold_price_scenarios(
        self,
        current_price: float = 2050.0,
        horizon_days: int = 90
    ) -> List[ScenarioBranch]:
        """
        Generate price scenarios for gold based on game-theoretic analysis.

        Returns Bayesian scenario branches with trade implications.
        """
        cb_eq = self.compute_cb_equilibrium()
        coord_prob = cb_eq["de_dollarization_coordination_prob"]

        scenarios = [
            ScenarioBranch(
                name="DE_DOLLAR_ACCELERATION",
                description="PBOC accelerates gold accumulation, ECB coordinates. Fed forced to accommodate.",
                prior=coord_prob * 0.6,
                horizon="quarters",
                confirming_catalysts=[
                    "PBOC gold purchase announcements > 50 tonnes/month",
                    "BRICS summit currency agreements",
                    "USD reserve share drops below 55%"
                ],
                killer_catalysts=[
                    "Fed maintains hawkish stance with 5%+ rates",
                    "China capital flight accelerates",
                    "EU-US security alignment tightens"
                ],
                trade_implications=[
                    "Long GLD/IAU with 6-month horizon",
                    "Long EMFX basket vs USD",
                    "Overweight gold miners (GDX)"
                ]
            ),
            ScenarioBranch(
                name="RATE_ECONOMICS_DOMINANCE",
                description="Real rates remain elevated. Carry trade favors USD. Gold faces opportunity cost.",
                prior=(1 - coord_prob) * 0.7,
                horizon="quarters",
                confirming_catalysts=[
                    "Fed funds rate stays above 4%",
                    "US breakevens decline (disinflation)",
                    "Strong USD trade-weighted"
                ],
                killer_catalysts=[
                    "Inflation re-acceleration",
                    "Fed pivot to cuts",
                    "US fiscal crisis"
                ],
                trade_implications=[
                    "Underweight gold vs bonds",
                    "Long UUP (USD)",
                    "Short gold on rallies"
                ]
            ),
            ScenarioBranch(
                name="CRISIS_SAFE_HAVEN",
                description="Geopolitical or financial crisis triggers flight to safety.",
                prior=0.15,
                horizon="weeks-months",
                confirming_catalysts=[
                    "Major geopolitical escalation (Taiwan, Middle East)",
                    "Banking stress or credit event",
                    "Equity market correction > 20%"
                ],
                killer_catalysts=[
                    "Geopolitical de-escalation",
                    "Central bank intervention succeeds",
                    "Risk-on recovery"
                ],
                trade_implications=[
                    "Long gold + vol tail hedges",
                    "Long GLD calls",
                    "Long TLT + GLD barbell"
                ]
            )
        ]

        # Normalize priors
        total_prior = sum(s.prior for s in scenarios)
        for s in scenarios:
            s.prior /= total_prior
            s.posterior = s.prior

        # Add price targets
        for scenario in scenarios:
            if scenario.name == "DE_DOLLAR_ACCELERATION":
                scenario.metadata = {
                    "price_target": current_price * 1.15,
                    "price_range": (current_price * 1.05, current_price * 1.25)
                }
            elif scenario.name == "RATE_ECONOMICS_DOMINANCE":
                scenario.metadata = {
                    "price_target": current_price * 0.92,
                    "price_range": (current_price * 0.85, current_price * 1.02)
                }
            elif scenario.name == "CRISIS_SAFE_HAVEN":
                scenario.metadata = {
                    "price_target": current_price * 1.25,
                    "price_range": (current_price * 1.10, current_price * 1.40)
                }

        return scenarios


# =============================================================================
# AGENT-BASED SIMULATION
# =============================================================================


class StrategicAgent(ABC):
    """Abstract base class for strategic agents in simulation."""

    def __init__(
        self,
        agent_id: str,
        player: Player,
        initial_position: float = 0.0
    ):
        self.id = agent_id
        self.player = player
        self.position = initial_position
        self.capital = player.capital
        self.history: List[Tuple[str, float]] = []

    @abstractmethod
    def decide(
        self,
        market_state: Dict[str, float],
        game_state: Optional[GameState]
    ) -> Tuple[str, float]:
        """
        Make a strategic decision.

        Returns:
            Tuple of (action, size)
        """
        pass

    def update(self, action: str, size: float, realized_pnl: float):
        """Update agent state after action."""
        self.position += size if action == "BUY" else -size if action == "SELL" else 0
        self.capital += realized_pnl
        self.history.append((action, size))


class CentralBankAgent(StrategicAgent):
    """Central bank agent with long-term strategic objectives."""

    def __init__(
        self,
        agent_id: str,
        player: Player,
        accumulation_target: float,  # Target gold holdings
        de_dollar_support: float      # Degree of de-dollarization support
    ):
        super().__init__(agent_id, player)
        self.accumulation_target = accumulation_target
        self.de_dollar_support = de_dollar_support

    def decide(
        self,
        market_state: Dict[str, float],
        game_state: Optional[GameState]
    ) -> Tuple[str, float]:
        """
        CB decision logic: Accumulate gold quietly in non-US hours.
        """
        session = market_state.get("session", "US")
        current_gold = market_state.get("gold_holdings", 0)

        # Gap to target
        gap = self.accumulation_target - current_gold

        if gap <= 0:
            return ("HOLD", 0)

        # Accumulate more in Asia/Euro sessions (stealth)
        if session in ["ASIA", "EURO"]:
            # Higher accumulation when de-dollar support is high
            size = min(gap * 0.01 * self.de_dollar_support, self.capital * 0.0001)
            return ("BUY", size)
        else:
            # Minimal accumulation in US session
            size = min(gap * 0.001 * self.de_dollar_support, self.capital * 0.00001)
            return ("BUY", size)


class AlgorithmicTrader(StrategicAgent):
    """Algorithmic trader exploiting mean reversion and momentum."""

    def __init__(
        self,
        agent_id: str,
        player: Player,
        lookback: int = 5,
        mean_revert_threshold: float = 0.01
    ):
        super().__init__(agent_id, player)
        self.lookback = lookback
        self.mean_revert_threshold = mean_revert_threshold
        self.price_history: List[float] = []

    def decide(
        self,
        market_state: Dict[str, float],
        game_state: Optional[GameState]
    ) -> Tuple[str, float]:
        """
        Algo decision: Mean revert overnight gains into US session.
        """
        price = market_state.get("price", 100)
        session = market_state.get("session", "US")

        self.price_history.append(price)

        if len(self.price_history) < self.lookback:
            return ("HOLD", 0)

        # Compute recent return
        recent_return = (price - self.price_history[-self.lookback]) / self.price_history[-self.lookback]

        # Mean reversion strategy
        if session == "US" and recent_return > self.mean_revert_threshold:
            # Overnight gains → sell into US open
            size = self.capital * 0.001 * abs(recent_return) / self.mean_revert_threshold
            return ("SELL", min(size, self.capital * 0.01))

        elif session == "US" and recent_return < -self.mean_revert_threshold:
            # Overnight losses → buy into US open
            size = self.capital * 0.001 * abs(recent_return) / self.mean_revert_threshold
            return ("BUY", min(size, self.capital * 0.01))

        return ("HOLD", 0)


class MarketSimulator:
    """
    Simulates market dynamics with multiple strategic agents.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        liquidity: float = 1e12
    ):
        self.price = initial_price
        self.volatility = volatility
        self.liquidity = liquidity
        self.agents: List[StrategicAgent] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []

    def add_agent(self, agent: StrategicAgent):
        """Add an agent to the simulation."""
        self.agents.append(agent)

    def step(
        self,
        session: str,
        exogenous_shock: float = 0.0
    ) -> Dict[str, float]:
        """
        Simulate one time step.

        Args:
            session: Current trading session
            exogenous_shock: External price shock

        Returns:
            Market state after step
        """
        market_state = {
            "price": self.price,
            "session": session,
            "volatility": self.volatility
        }

        # Collect agent decisions
        total_demand = 0.0
        total_volume = 0.0

        for agent in self.agents:
            action, size = agent.decide(market_state, None)

            if action == "BUY":
                total_demand += size
            elif action == "SELL":
                total_demand -= size

            total_volume += abs(size)

        # Price impact
        impact = total_demand / self.liquidity

        # Random noise
        noise = np.random.normal(0, self.volatility)

        # Price update
        self.price *= (1 + impact + noise + exogenous_shock)

        self.price_history.append(self.price)
        self.volume_history.append(total_volume)

        return {
            "price": self.price,
            "price_change": impact + noise + exogenous_shock,
            "volume": total_volume,
            "net_demand": total_demand
        }

    def simulate_day(
        self,
        sessions: List[str] = ["ASIA", "EURO", "US"],
        steps_per_session: int = 8
    ) -> Dict[str, Any]:
        """
        Simulate a full trading day across sessions.
        """
        day_results = {
            "session_results": [],
            "open_price": self.price,
            "close_price": None
        }

        for session in sessions:
            session_open = self.price

            for _ in range(steps_per_session):
                self.step(session)

            session_close = self.price
            session_return = (session_close - session_open) / session_open

            day_results["session_results"].append({
                "session": session,
                "open": session_open,
                "close": session_close,
                "return": session_return
            })

        day_results["close_price"] = self.price
        day_results["daily_return"] = (self.price - day_results["open_price"]) / day_results["open_price"]

        return day_results


# =============================================================================
# INSIGHT GENERATION
# =============================================================================


class StrategicInsightEngine:
    """
    Generates actionable insights from game-theoretic analysis.

    Synthesizes:
    1. Equilibrium analysis
    2. Scenario probabilities
    3. Nth-order effects
    4. Trade recommendations
    """

    def __init__(self):
        self.causal_graph = CausalGraph()

    def generate_us_china_insights(
        self,
        game: USChinaGrandStrategyGame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights for US-China competition.
        """
        # Compute equilibria across domains
        tech_eq = game.compute_equilibrium("tech")
        trade_eq = game.compute_equilibrium("trade")
        finance_eq = game.compute_equilibrium("finance")

        # Most likely scenario
        us_likely = max(
            tech_eq["us_strategy"].items(),
            key=lambda x: x[1]
        )[0]
        china_likely = max(
            tech_eq["china_strategy"].items(),
            key=lambda x: x[1]
        )[0]

        scenario_analysis = game.analyze_scenario(us_likely, china_likely)

        # Build insights
        insights = {
            "summary": self._generate_summary(tech_eq, trade_eq, finance_eq),
            "equilibria": {
                "technology": tech_eq,
                "trade": trade_eq,
                "finance": finance_eq
            },
            "most_likely_scenario": scenario_analysis,
            "market_implications": self._derive_market_implications(
                scenario_analysis["nth_order_effects"]
            ),
            "trade_recommendations": self._generate_trade_recommendations(
                scenario_analysis, tech_eq
            ),
            "risk_assessment": self._assess_risks(tech_eq, trade_eq)
        }

        return insights

    def _generate_summary(
        self,
        tech_eq: Dict,
        trade_eq: Dict,
        finance_eq: Dict
    ) -> str:
        """Generate executive summary."""

        us_posture = "competitive" if tech_eq["most_likely_us_action"] in ["compete", "confront"] else "cooperative"
        china_posture = "decoupling" if tech_eq["most_likely_china_action"] in ["decouple", "confront"] else "engaging"

        return f"""
US-CHINA STRATEGIC ASSESSMENT

Current Equilibrium State:
- US posture: {us_posture.upper()} (likely action: {tech_eq['most_likely_us_action']})
- China posture: {china_posture.upper()} (likely action: {tech_eq['most_likely_china_action']})

Domain Analysis:
- Technology: US payoff={tech_eq['expected_payoff_us']:.1f}, China payoff={tech_eq['expected_payoff_china']:.1f}
- Trade: US payoff={trade_eq['expected_payoff_us']:.1f}, China payoff={trade_eq['expected_payoff_china']:.1f}
- Finance: US payoff={finance_eq['expected_payoff_us']:.1f}, China payoff={finance_eq['expected_payoff_china']:.1f}

Equilibrium Assessment:
The current strategic configuration represents a {self._classify_equilibrium(tech_eq)}.
"""

    def _classify_equilibrium(self, eq: Dict) -> str:
        """Classify the type of equilibrium."""
        us_action = eq["most_likely_us_action"]
        china_action = eq["most_likely_china_action"]

        if us_action == "cooperate" and china_action == "cooperate":
            return "COOPERATIVE EQUILIBRIUM (mutually beneficial)"
        elif us_action in ["confront", "contain"] and china_action in ["confront", "decouple"]:
            return "CONFLICT EQUILIBRIUM (destructive standoff)"
        elif us_action == "compete" and china_action == "compete":
            return "COMPETITIVE EQUILIBRIUM (managed rivalry)"
        else:
            return "MIXED EQUILIBRIUM (asymmetric strategies)"

    def _derive_market_implications(
        self,
        nth_order_effects: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Convert nth-order effects to market implications."""

        implications = []

        asset_mapping = {
            "ChipSupply": {"assets": ["SMH", "SOXX", "TSM"], "sector": "Semiconductors"},
            "TechEquities": {"assets": ["QQQ", "XLK"], "sector": "Technology"},
            "AIStocks": {"assets": ["NVDA", "MSFT", "GOOGL"], "sector": "AI/ML"},
            "GlobalTrade": {"assets": ["EEM", "FXI"], "sector": "Emerging Markets"},
            "USD": {"assets": ["UUP", "DXY"], "sector": "Currency"},
            "GoldPrice": {"assets": ["GLD", "IAU"], "sector": "Precious Metals"},
            "EquityVol": {"assets": ["VIX", "VIXY"], "sector": "Volatility"},
            "InflationExpectations": {"assets": ["TIP", "SCHP"], "sector": "TIPS"}
        }

        for effect in nth_order_effects[:8]:
            factor = effect["factor"]
            direction = effect["direction"]
            score = effect["score"]

            if factor in asset_mapping:
                mapping = asset_mapping[factor]
                implications.append({
                    "factor": factor,
                    "direction": direction,
                    "magnitude": score,
                    "assets": mapping["assets"],
                    "sector": mapping["sector"],
                    "recommendation": f"{'LONG' if direction == 'UP' else 'SHORT'} {mapping['assets'][0]} (conviction: {score:.2f})"
                })

        return implications

    def _generate_trade_recommendations(
        self,
        scenario: Dict,
        tech_eq: Dict
    ) -> List[Dict[str, Any]]:
        """Generate specific trade recommendations."""

        recommendations = []

        # Based on scenario
        if scenario["primary_shock"] == "TechWar":
            recommendations.append({
                "trade": "Underweight Semiconductors",
                "expression": "Short SMH vs Long XLK",
                "rationale": "Tech war disrupts chip supply chains disproportionately",
                "conviction": "HIGH" if scenario["shock_direction"] == "UP" else "MEDIUM",
                "horizon": "3-6 months"
            })
            recommendations.append({
                "trade": "Long Domestic Tech",
                "expression": "Long US software vs hardware",
                "rationale": "Software less exposed to supply chain disruption",
                "conviction": "MEDIUM",
                "horizon": "6-12 months"
            })

        if scenario["primary_shock"] == "TradeWar":
            recommendations.append({
                "trade": "Underweight EM, Overweight US",
                "expression": "Long SPY vs Short EEM",
                "rationale": "Trade war hurts EM exporters disproportionately",
                "conviction": "HIGH",
                "horizon": "3-6 months"
            })

        if scenario["primary_shock"] == "FinancialDecoupling":
            recommendations.append({
                "trade": "Long Gold",
                "expression": "Long GLD",
                "rationale": "De-dollarization supports gold as alternative reserve",
                "conviction": "MEDIUM",
                "horizon": "12-24 months"
            })

        return recommendations

    def _assess_risks(
        self,
        tech_eq: Dict,
        trade_eq: Dict
    ) -> Dict[str, Any]:
        """Assess key risks and uncertainties."""

        return {
            "equilibrium_stability": "UNSTABLE" if tech_eq["most_likely_us_action"] == "confront" else "STABLE",
            "key_uncertainties": [
                "Taiwan scenario escalation",
                "US election cycle policy shifts",
                "China economic slowdown trajectory",
                "Technology leapfrog possibilities"
            ],
            "black_swan_scenarios": [
                {"event": "Taiwan military conflict", "probability": 0.05, "impact": "SEVERE"},
                {"event": "Major US-China financial decoupling", "probability": 0.15, "impact": "HIGH"},
                {"event": "Breakthrough cooperation agreement", "probability": 0.10, "impact": "POSITIVE"}
            ]
        }

    def generate_precious_metals_insights(
        self,
        game: PreciousMetalsGame
    ) -> Dict[str, Any]:
        """
        Generate insights for precious metals markets.
        """
        cb_eq = game.compute_cb_equilibrium()
        scenarios = game.generate_gold_price_scenarios()

        # Session analysis
        asia_dynamics = game.session_dynamics("ASIA", cb_eq)
        euro_dynamics = game.session_dynamics("EURO", cb_eq)
        us_dynamics = game.session_dynamics("US", cb_eq)

        # Build scenario engine
        scenario_engine = BayesianScenarioEngine(scenarios)

        insights = {
            "summary": self._generate_gold_summary(cb_eq, scenarios),
            "central_bank_equilibrium": cb_eq,
            "scenarios": [
                {
                    "name": s.name,
                    "description": s.description,
                    "probability": s.posterior,
                    "price_target": s.metadata.get("price_target") if hasattr(s, "metadata") else None,
                    "confirming_catalysts": s.confirming_catalysts,
                    "trade_implications": s.trade_implications
                }
                for s in scenarios
            ],
            "session_dynamics": {
                "asia": asia_dynamics,
                "euro": euro_dynamics,
                "us": us_dynamics
            },
            "intraday_pattern": self._analyze_intraday_pattern(
                asia_dynamics, euro_dynamics, us_dynamics
            ),
            "trade_recommendations": self._generate_gold_trades(scenarios, cb_eq)
        }

        return insights

    def _generate_gold_summary(
        self,
        cb_eq: Dict,
        scenarios: List[ScenarioBranch]
    ) -> str:
        """Generate summary for gold market analysis."""

        coord_prob = cb_eq["de_dollarization_coordination_prob"]
        top_scenario = max(scenarios, key=lambda s: s.posterior)

        return f"""
PRECIOUS METALS STRATEGIC ASSESSMENT

De-dollarization Coordination Probability: {coord_prob:.1%}
Assessment: {cb_eq['equilibrium_assessment']}

Central Bank Strategies:
- PBOC: {max(cb_eq['pboc_strategy'].items(), key=lambda x: x[1])[0]} ({max(cb_eq['pboc_strategy'].values()):.1%})
- ECB: {max(cb_eq['ecb_strategy'].items(), key=lambda x: x[1])[0]} ({max(cb_eq['ecb_strategy'].values()):.1%})

Most Likely Scenario: {top_scenario.name} ({top_scenario.posterior:.1%})
{top_scenario.description}

Key Insight: The market is currently in a state where {'political forces (de-dollarization) dominate' if coord_prob > 0.4 else 'economic forces (real rates) dominate'}.
"""

    def _analyze_intraday_pattern(
        self,
        asia: Dict,
        euro: Dict,
        us: Dict
    ) -> Dict[str, Any]:
        """Analyze the classic Asia-up, US-down intraday pattern."""

        pattern_strength = asia["expected_return"] - us["expected_return"]

        return {
            "pattern_exists": pattern_strength > 0.001,
            "pattern_strength": pattern_strength,
            "asia_expected_return": asia["expected_return"],
            "euro_expected_return": euro["expected_return"],
            "us_expected_return": us["expected_return"],
            "explanation": (
                "Political accumulation in Asia sessions creates upward pressure. "
                "Economic forces (real rates, carry trade) dominate in US session, "
                "creating selling pressure."
            ) if pattern_strength > 0 else (
                "Pattern currently weak. Economic forces dominating across sessions."
            ),
            "trading_opportunity": {
                "strategy": "Long gold in Asia, short into US open",
                "expected_edge": pattern_strength,
                "confidence": "MEDIUM" if 0.001 < pattern_strength < 0.005 else "HIGH" if pattern_strength >= 0.005 else "LOW"
            }
        }

    def _generate_gold_trades(
        self,
        scenarios: List[ScenarioBranch],
        cb_eq: Dict
    ) -> List[Dict[str, Any]]:
        """Generate gold trading recommendations."""

        recommendations = []
        coord_prob = cb_eq["de_dollarization_coordination_prob"]

        if coord_prob > 0.4:
            recommendations.append({
                "trade": "Strategic Long Gold",
                "expression": "Long GLD/IAU",
                "rationale": "De-dollarization coordination probability elevated",
                "conviction": "MEDIUM-HIGH",
                "horizon": "6-12 months",
                "sizing": "5-8% of portfolio"
            })

        recommendations.append({
            "trade": "Intraday Pattern Arbitrage",
            "expression": "Long gold futures Asia, short into US open",
            "rationale": "Exploit session-based belief divergence",
            "conviction": "MEDIUM",
            "horizon": "Intraday",
            "sizing": "Tactical, risk-controlled"
        })

        recommendations.append({
            "trade": "Gold Convexity (Tail Hedge)",
            "expression": "Long GLD calls (3-6 month)",
            "rationale": "Asymmetric payoff in crisis scenarios",
            "conviction": "MEDIUM",
            "horizon": "3-6 months",
            "sizing": "1-2% of portfolio"
        })

        return recommendations


# =============================================================================
# MAIN INTERFACE
# =============================================================================


class EliteCGTSystem:
    """
    Main interface for the Elite CGT/AGT Analysis System.

    Provides unified access to:
    1. US-China grand strategy analysis
    2. Precious metals market analysis
    3. Custom game construction and solution
    4. Scenario-based planning
    """

    def __init__(self):
        self.insight_engine = StrategicInsightEngine()
        self.causal_graph = CausalGraph()
        self.nash_solver = NashEquilibriumSolver()
        self.evo_dynamics = EvolutionaryDynamics()

        logger.info("Elite CGT/AGT System initialized")

    def analyze_us_china(self) -> Dict[str, Any]:
        """Run comprehensive US-China strategic analysis."""

        logger.info("Running US-China grand strategy analysis...")
        game = USChinaGrandStrategyGame()
        insights = self.insight_engine.generate_us_china_insights(game)
        logger.info("US-China analysis complete")

        return insights

    def analyze_precious_metals(self) -> Dict[str, Any]:
        """Run comprehensive precious metals analysis."""

        logger.info("Running precious metals analysis...")
        game = PreciousMetalsGame()
        insights = self.insight_engine.generate_precious_metals_insights(game)
        logger.info("Precious metals analysis complete")

        return insights

    def analyze_custom_game(
        self,
        payoff_matrix_a: np.ndarray,
        payoff_matrix_b: np.ndarray,
        player_a_actions: List[str],
        player_b_actions: List[str],
        shock_mappings: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a custom 2-player game.

        Args:
            payoff_matrix_a: Payoff matrix for player A
            payoff_matrix_b: Payoff matrix for player B
            player_a_actions: Action labels for player A
            player_b_actions: Action labels for player B
            shock_mappings: Map from action pairs to (shock_node, direction)

        Returns:
            Analysis results
        """
        logger.info("Analyzing custom game...")

        # Compute equilibrium
        sigma_a, sigma_b = self.nash_solver.solve_two_player_bimatrix(
            payoff_matrix_a, payoff_matrix_b
        )

        # Expected payoffs
        exp_a = sigma_a @ payoff_matrix_a @ sigma_b
        exp_b = sigma_a @ payoff_matrix_b @ sigma_b

        # Most likely actions
        a_action = player_a_actions[np.argmax(sigma_a)]
        b_action = player_b_actions[np.argmax(sigma_b)]

        # Nth-order effects if shock mappings provided
        effects = []
        if shock_mappings and (a_action, b_action) in shock_mappings:
            shock_node, direction = shock_mappings[(a_action, b_action)]
            effects = self.causal_graph.propagate(shock_node, direction)

        return {
            "player_a_strategy": dict(zip(player_a_actions, sigma_a)),
            "player_b_strategy": dict(zip(player_b_actions, sigma_b)),
            "expected_payoff_a": float(exp_a),
            "expected_payoff_b": float(exp_b),
            "most_likely_a_action": a_action,
            "most_likely_b_action": b_action,
            "nth_order_effects": effects
        }

    def run_scenario_analysis(
        self,
        scenarios: List[Dict[str, Any]],
        evidence: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run Bayesian scenario analysis with evidence updates.

        Args:
            scenarios: List of scenario dicts with name, description, prior
            evidence: Dict of scenario names to log-likelihood ratios

        Returns:
            Updated scenario analysis
        """
        branches = [
            ScenarioBranch(
                name=s["name"],
                description=s.get("description", ""),
                prior=s["prior"],
                horizon=s.get("horizon", "quarters"),
                confirming_catalysts=s.get("confirming_catalysts", []),
                killer_catalysts=s.get("killer_catalysts", []),
                trade_implications=s.get("trade_implications", [])
            )
            for s in scenarios
        ]

        engine = BayesianScenarioEngine(branches)
        posteriors = engine.soft_bayes_update(evidence)

        return {
            "posteriors": posteriors,
            "top_scenario": max(posteriors.items(), key=lambda x: x[1]),
            "scenarios": [
                {
                    "name": b.name,
                    "prior": b.prior,
                    "posterior": b.posterior,
                    "log_lr": b.bayes_log_lr,
                    "description": b.description
                }
                for b in engine.scenarios
            ]
        }

    def simulate_market(
        self,
        n_days: int = 30,
        initial_price: float = 2050.0
    ) -> Dict[str, Any]:
        """
        Run agent-based market simulation.

        Returns simulation results with price trajectories.
        """
        logger.info(f"Running {n_days}-day market simulation...")

        simulator = MarketSimulator(initial_price=initial_price)

        # Add agents
        pboc_player = Player(
            id="PBOC", name="PBOC", player_type=PlayerType.SOVEREIGN_STATE,
            action_set=["BUY", "HOLD", "SELL"], capital=500e9
        )
        algo_player = Player(
            id="ALGO", name="Algorithmic Traders", player_type=PlayerType.ALGORITHMIC,
            action_set=["BUY", "HOLD", "SELL"], capital=50e9
        )

        simulator.add_agent(CentralBankAgent(
            "PBOC_agent", pboc_player,
            accumulation_target=3000,  # tonnes
            de_dollar_support=0.85
        ))
        simulator.add_agent(AlgorithmicTrader(
            "ALGO_agent", algo_player,
            lookback=5, mean_revert_threshold=0.008
        ))

        # Run simulation
        daily_results = []
        for day in range(n_days):
            result = simulator.simulate_day()
            daily_results.append(result)

        logger.info("Market simulation complete")

        return {
            "n_days": n_days,
            "initial_price": initial_price,
            "final_price": simulator.price,
            "total_return": (simulator.price - initial_price) / initial_price,
            "price_history": simulator.price_history,
            "daily_results": daily_results
        }

    def generate_report(
        self,
        analysis_type: str = "all"
    ) -> str:
        """
        Generate a comprehensive strategic report.

        Args:
            analysis_type: One of "us_china", "precious_metals", "all"

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "ELITE CGT/AGT STRATEGIC INTELLIGENCE REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 80,
            ""
        ]

        if analysis_type in ["us_china", "all"]:
            us_china = self.analyze_us_china()
            report_lines.append(us_china["summary"])
            report_lines.append("")
            report_lines.append("MARKET IMPLICATIONS:")
            for impl in us_china["market_implications"][:5]:
                report_lines.append(f"  - {impl['recommendation']}")
            report_lines.append("")
            report_lines.append("TRADE RECOMMENDATIONS:")
            for rec in us_china["trade_recommendations"][:3]:
                report_lines.append(f"  - {rec['trade']}: {rec['expression']} ({rec['conviction']})")
            report_lines.append("")
            report_lines.append("-" * 80)
            report_lines.append("")

        if analysis_type in ["precious_metals", "all"]:
            pm = self.analyze_precious_metals()
            report_lines.append(pm["summary"])
            report_lines.append("")
            report_lines.append("SCENARIO PROBABILITIES:")
            for scenario in pm["scenarios"]:
                report_lines.append(f"  - {scenario['name']}: {scenario['probability']:.1%}")
            report_lines.append("")
            report_lines.append("INTRADAY PATTERN:")
            pattern = pm["intraday_pattern"]
            report_lines.append(f"  Pattern exists: {pattern['pattern_exists']}")
            report_lines.append(f"  Expected edge: {pattern['pattern_strength']:.4f}")
            report_lines.append(f"  Confidence: {pattern['trading_opportunity']['confidence']}")
            report_lines.append("")
            report_lines.append("TRADE RECOMMENDATIONS:")
            for rec in pm["trade_recommendations"][:3]:
                report_lines.append(f"  - {rec['trade']}: {rec['expression']} ({rec['conviction']})")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Main entry point for CLI usage."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Elite CGT/AGT Strategic Analysis System"
    )
    parser.add_argument(
        "--analysis",
        choices=["us_china", "precious_metals", "all"],
        default="all",
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run market simulation"
    )
    parser.add_argument(
        "--sim-days",
        type=int,
        default=30,
        help="Number of days to simulate"
    )

    args = parser.parse_args()

    # Initialize system
    system = EliteCGTSystem()

    # Generate report
    report = system.generate_report(args.analysis)

    # Optional simulation
    if args.simulate:
        sim_results = system.simulate_market(n_days=args.sim_days)
        report += f"\n\nSIMULATION RESULTS:\n"
        report += f"Days simulated: {sim_results['n_days']}\n"
        report += f"Final price: ${sim_results['final_price']:.2f}\n"
        report += f"Total return: {sim_results['total_return']:.2%}\n"

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
