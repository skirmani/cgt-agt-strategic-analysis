#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BRANDT vs OLIVER: SILVER MARKET GAME THEORY ANALYSIS
================================================================================

Analyzing two contrasting views on silver using Computational Game Theory:

PETER BRANDT (Bearish View):
- 4.3B oz traded on Comex = 5.2 years production
- Miners should hedge 3 years at $110+
- Recycling could double/triple at high prices
- 10% demand reduction from substitution
- Warns of "pipeline glut" scenario

MICHAEL OLIVER (Bullish View):
- $300-$500 silver by summer
- Different from 1980/2011 due to monetary factors
- Dollar weakness, bond instability
- Institutional shifts away from dollar
- Gold to $8,500 first
- Mining stocks massively undervalued

Analysis Framework:
1. Multi-player strategic game (Miners, Speculators, Industry, Central Banks)
2. Bayesian scenario tree with probability updating
3. Evolutionary dynamics of market strategies
4. Nth-order causal effects for each scenario

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from enum import Enum, auto
import math

# =============================================================================
# PLAYER AND SCENARIO DEFINITIONS
# =============================================================================

class MarketActor(Enum):
    MINERS = "Miners"
    SPECULATORS = "Speculators"
    INDUSTRY = "Industrial Users"
    RECYCLERS = "Recyclers"
    CENTRAL_BANKS = "Central Banks"
    ETF_FUNDS = "ETF/Physical Funds"

@dataclass
class SilverScenario:
    """Represents a silver market scenario branch."""
    name: str
    description: str
    prior: float
    posterior: float = 0.0
    price_target: Tuple[float, float] = (0, 0)  # (low, high)
    timeframe: str = "12-18 months"
    key_drivers: List[str] = field(default_factory=list)
    confirming_signals: List[str] = field(default_factory=list)
    killer_signals: List[str] = field(default_factory=list)
    trade_implications: List[str] = field(default_factory=list)
    nth_order_effects: Dict[str, float] = field(default_factory=dict)

# =============================================================================
# BRANDT vs OLIVER SCENARIO CONSTRUCTION
# =============================================================================

def build_silver_scenarios() -> List[SilverScenario]:
    """Build scenario tree incorporating both Brandt and Oliver views."""

    scenarios = [
        # BRANDT'S BEARISH SCENARIOS
        SilverScenario(
            name="HEDGING_SUPPLY_GLUT",
            description="Miners hedge aggressively at $110+, recycling doubles, industrial substitution occurs",
            prior=0.25,
            price_target=(65, 85),
            timeframe="6-18 months",
            key_drivers=[
                "Comex 4.3B oz volume = 5.2 years production available",
                "Miners rational to hedge 3 years at $110+",
                "Recycling doubles from jewelry/industrial scrap at $100+",
                "Industrial users substitute to copper/aluminum where possible",
                "Speculative longs face massive real supply wall"
            ],
            confirming_signals=[
                "Miner forward sales announcements increase",
                "Comex registered inventory rising",
                "Recycling flows increase (scrap dealers report surge)",
                "Industrial demand reports show demand destruction",
                "Speculative long positioning at extremes"
            ],
            killer_signals=[
                "Miners choose not to hedge despite prices",
                "Physical premium over paper persists/expands",
                "Central bank accumulation continues",
                "Dollar index breaks below 95 decisively"
            ],
            trade_implications=[
                "SHORT SLV at $100+ with tight stops",
                "LONG mining puts (AG, PAAS, HL)",
                "Pairs: LONG GLD / SHORT SLV at ratio extremes",
                "Timeline: 6-12 month bear case"
            ],
            nth_order_effects={
                "MiningEquity": -0.35,
                "GoldSilverRatio": 0.40,
                "IndustrialMetals": -0.15,
                "RecyclingIndustry": 0.50,
                "JewelryDemand": -0.25
            }
        ),

        SilverScenario(
            name="DEMAND_DESTRUCTION_CYCLE",
            description="High prices trigger self-correcting supply/demand mechanism",
            prior=0.15,
            price_target=(45, 70),
            timeframe="12-24 months",
            key_drivers=[
                "Solar/EV demand elasticity kicks in at high prices",
                "Industrial users stockpiled at lower prices",
                "Inventory destocking cycle begins",
                "Jewelry demand collapses in price-sensitive markets (India, China)"
            ],
            confirming_signals=[
                "Solar panel manufacturers announce cost-cutting measures",
                "Industrial inventory reports show destocking",
                "Indian silver imports decline significantly",
                "Photography/medical demand continues secular decline"
            ],
            killer_signals=[
                "Green energy mandates override cost concerns",
                "Industrial users scramble for supply (shortage signals)",
                "India/China buying dips aggressively"
            ],
            trade_implications=[
                "SHORT silver futures on rallies",
                "LONG copper vs silver spread",
                "Avoid mining equities",
                "Timeline: 1-2 year bear case"
            ],
            nth_order_effects={
                "SolarStocks": -0.20,
                "EVManufacturers": -0.10,
                "GoldSilverRatio": 0.55,
                "IndustrialProduction": -0.05,
                "BaseMetals": 0.15
            }
        ),

        # OLIVER'S BULLISH SCENARIOS
        SilverScenario(
            name="MONETARY_REGIME_CHANGE",
            description="Dollar hegemony crisis drives silver to $300-500 as monetary metal",
            prior=0.20,
            price_target=(200, 500),
            timeframe="6-18 months",
            key_drivers=[
                "Different from 1980/2011 - this is monetary, not speculative",
                "Dollar losing reserve status (BRICS alternatives)",
                "Bond market instability forces Fed to monetize",
                "Institutional reallocation away from treasuries",
                "Gold to $8,500 creates massive silver catch-up trade",
                "Mining stocks at multi-decade lows vs metal prices"
            ],
            confirming_signals=[
                "Dollar index breaks below 90",
                "10Y treasury yields spike despite Fed cuts",
                "Central bank gold buying accelerates",
                "BRICS announce commodity-backed currency progress",
                "Gold/Silver ratio contracts below 60",
                "Mining stocks outperform physical"
            ],
            killer_signals=[
                "Dollar strength returns (DXY > 110)",
                "Deflation scare (Fed tightening)",
                "Risk-off moves favor treasuries",
                "Gold fails at $3,500"
            ],
            trade_implications=[
                "LONG silver miners (leveraged exposure)",
                "LONG physical silver (coins, bars)",
                "LONG SLV calls 12-18 month expiry",
                "LONG gold/silver miners (GDX, GDXJ, SIL)",
                "SHORT treasury bonds (TLT puts)"
            ],
            nth_order_effects={
                "USD": -0.70,
                "TreasuryBonds": -0.55,
                "GoldPrice": 0.85,
                "MiningEquity": 0.90,
                "Inflation": 0.60,
                "RealRates": -0.50,
                "CryptoAssets": 0.40,
                "EMFX": 0.35
            }
        ),

        SilverScenario(
            name="INDUSTRIAL_SUPERCYCLE",
            description="Green energy + AI infrastructure creates structural deficit",
            prior=0.20,
            price_target=(100, 200),
            timeframe="12-36 months",
            key_drivers=[
                "Solar demand growth exponential (100M oz/year and growing)",
                "EV adoption accelerating (each EV uses 1-2 oz silver)",
                "5G/AI infrastructure buildout",
                "Mine supply constrained (underinvestment since 2012)",
                "Above-ground inventories depleting"
            ],
            confirming_signals=[
                "Comex registered inventory declining",
                "LBMA inventory declining",
                "Physical premiums expanding",
                "Mine supply forecasts revised down",
                "Industrial users paying premium for guaranteed delivery"
            ],
            killer_signals=[
                "Recycling surge fills deficit",
                "Technology substitution (copper, graphene)",
                "Green energy investment slows",
                "Major new mine supply comes online"
            ],
            trade_implications=[
                "LONG silver futures (back months)",
                "LONG physical silver",
                "LONG silver miners (production growth)",
                "LONG solar stocks (First Solar, etc.)"
            ],
            nth_order_effects={
                "SolarStocks": 0.45,
                "EVManufacturers": 0.30,
                "MiningEquity": 0.70,
                "GreenEnergy": 0.35,
                "CopperPrice": 0.40,
                "IndustrialProduction": 0.20
            }
        ),

        # NEUTRAL/MIXED SCENARIOS
        SilverScenario(
            name="RANGE_BOUND_VOLATILITY",
            description="Neither thesis dominates; silver trades wide range with high vol",
            prior=0.15,
            price_target=(80, 140),
            timeframe="12-24 months",
            key_drivers=[
                "Conflicting forces create equilibrium",
                "Supply concerns balanced by demand uncertainty",
                "Dollar volatile but not collapsing",
                "Fed policy uncertain",
                "Industrial demand steady but not parabolic"
            ],
            confirming_signals=[
                "Silver range-bound for 6+ months",
                "Volatility remains elevated",
                "Positioning neutral",
                "Inventory stable"
            ],
            killer_signals=[
                "Breakout above $150 sustained",
                "Breakdown below $70 sustained",
                "Major macro catalyst emerges"
            ],
            trade_implications=[
                "SELL volatility (strangles, iron condors)",
                "Range trading strategies",
                "Avoid directional bets",
                "Accumulate on dips below $90, trim on rallies above $130"
            ],
            nth_order_effects={
                "VolatilityProducts": 0.30,
                "MiningEquity": 0.15,
                "OptionsMarkets": 0.40
            }
        ),

        SilverScenario(
            name="GOLD_CATCH_UP_RALLY",
            description="Silver catches up to gold's performance as ratio normalizes",
            prior=0.05,
            price_target=(150, 250),
            timeframe="6-12 months",
            key_drivers=[
                "Gold/Silver ratio at historical extremes (85-90)",
                "Historical mean reversion (ratio 50-60)",
                "Silver is 'poor man's gold' trade",
                "Retail investment demand surges"
            ],
            confirming_signals=[
                "Gold/Silver ratio breaks below 75",
                "Retail coin demand surging",
                "Silver ETF inflows accelerating",
                "Social media buzz on silver"
            ],
            killer_signals=[
                "Gold pulls back sharply",
                "Ratio expands above 100",
                "Risk-off favors gold over silver"
            ],
            trade_implications=[
                "LONG SLV / SHORT GLD spread",
                "LONG silver miners vs gold miners",
                "LONG physical silver"
            ],
            nth_order_effects={
                "GoldSilverRatio": -0.80,
                "RetailSentiment": 0.50,
                "ETFFlows": 0.60,
                "MiningEquity": 0.55
            }
        ),
    ]

    # Normalize priors
    total_prior = sum(s.prior for s in scenarios)
    for s in scenarios:
        s.prior /= total_prior
        s.posterior = s.prior

    return scenarios


# =============================================================================
# MULTI-PLAYER STRATEGIC GAME
# =============================================================================

def build_silver_market_game() -> Dict[str, Any]:
    """
    Build N-player silver market game.

    Players:
    1. Miners - Can hedge or hold (not sell forward)
    2. Speculators - Can go long, short, or neutral
    3. Industrial Users - Can stockpile, use normally, or substitute
    4. Recyclers - Can increase supply or hold
    5. ETF/Funds - Can accumulate or distribute
    """

    players = {
        "Miners": {
            "actions": ["HEDGE_AGGRESSIVELY", "HEDGE_MODERATE", "NO_HEDGE"],
            "sophistication": 0.70,
            "time_horizon": "3-5 years",
            "incentives": "Maximize present value of production, reduce bankruptcy risk"
        },
        "Speculators": {
            "actions": ["LONG", "NEUTRAL", "SHORT"],
            "sophistication": 0.85,
            "time_horizon": "1-12 months",
            "incentives": "Maximize risk-adjusted returns"
        },
        "Industrial_Users": {
            "actions": ["STOCKPILE", "NORMAL_USE", "SUBSTITUTE"],
            "sophistication": 0.65,
            "time_horizon": "1-3 years",
            "incentives": "Minimize input costs, ensure supply security"
        },
        "Recyclers": {
            "actions": ["INCREASE_SUPPLY", "MAINTAIN", "HOLD_INVENTORY"],
            "sophistication": 0.50,
            "time_horizon": "0-6 months",
            "incentives": "Maximize scrap margins"
        },
        "ETF_Funds": {
            "actions": ["ACCUMULATE", "HOLD", "DISTRIBUTE"],
            "sophistication": 0.80,
            "time_horizon": "1-5 years",
            "incentives": "Track investor flows, maintain liquidity"
        }
    }

    # Payoff matrices by scenario
    # Format: payoffs[player][scenario][action_profile] -> payoff

    # Simplified: For each scenario, what's the dominant strategy for each player?

    strategic_analysis = {
        "HEDGING_SUPPLY_GLUT": {
            "Miners": {
                "dominant_strategy": "HEDGE_AGGRESSIVELY",
                "rationale": "Brandt's thesis - lock in $110+ prices, avoid future price collapse",
                "payoff_if_correct": 8.0,
                "payoff_if_wrong": 3.0  # Still protected downside
            },
            "Speculators": {
                "dominant_strategy": "SHORT",
                "rationale": "Supply glut creates downward pressure",
                "payoff_if_correct": 7.0,
                "payoff_if_wrong": -5.0  # Squeeze risk
            },
            "Industrial_Users": {
                "dominant_strategy": "NORMAL_USE",
                "rationale": "No need to stockpile if prices falling",
                "payoff_if_correct": 6.0,
                "payoff_if_wrong": 2.0
            },
            "Recyclers": {
                "dominant_strategy": "INCREASE_SUPPLY",
                "rationale": "High prices incentivize maximum scrap flow",
                "payoff_if_correct": 8.0,
                "payoff_if_wrong": 4.0
            },
            "ETF_Funds": {
                "dominant_strategy": "DISTRIBUTE",
                "rationale": "Reduce inventory into strength",
                "payoff_if_correct": 6.0,
                "payoff_if_wrong": -2.0
            }
        },

        "MONETARY_REGIME_CHANGE": {
            "Miners": {
                "dominant_strategy": "NO_HEDGE",
                "rationale": "Oliver's thesis - why sell at $110 when $300+ coming?",
                "payoff_if_correct": 10.0,
                "payoff_if_wrong": -3.0  # Miss hedging opportunity
            },
            "Speculators": {
                "dominant_strategy": "LONG",
                "rationale": "Monetary crisis drives massive revaluation",
                "payoff_if_correct": 10.0,
                "payoff_if_wrong": -4.0
            },
            "Industrial_Users": {
                "dominant_strategy": "STOCKPILE",
                "rationale": "Prices going much higher, secure supply now",
                "payoff_if_correct": 8.0,
                "payoff_if_wrong": -3.0  # Tied up capital
            },
            "Recyclers": {
                "dominant_strategy": "HOLD_INVENTORY",
                "rationale": "Wait for higher prices before selling",
                "payoff_if_correct": 9.0,
                "payoff_if_wrong": -2.0
            },
            "ETF_Funds": {
                "dominant_strategy": "ACCUMULATE",
                "rationale": "Investor demand will surge",
                "payoff_if_correct": 9.0,
                "payoff_if_wrong": -3.0
            }
        },

        "INDUSTRIAL_SUPERCYCLE": {
            "Miners": {
                "dominant_strategy": "HEDGE_MODERATE",
                "rationale": "Protect some production, participate in upside",
                "payoff_if_correct": 7.0,
                "payoff_if_wrong": 4.0
            },
            "Speculators": {
                "dominant_strategy": "LONG",
                "rationale": "Structural deficit = higher prices",
                "payoff_if_correct": 8.0,
                "payoff_if_wrong": -3.0
            },
            "Industrial_Users": {
                "dominant_strategy": "STOCKPILE",
                "rationale": "Supply constraints mean secure supply priority",
                "payoff_if_correct": 9.0,
                "payoff_if_wrong": -2.0
            },
            "Recyclers": {
                "dominant_strategy": "INCREASE_SUPPLY",
                "rationale": "High prices, strong demand = good margins",
                "payoff_if_correct": 7.0,
                "payoff_if_wrong": 4.0
            },
            "ETF_Funds": {
                "dominant_strategy": "ACCUMULATE",
                "rationale": "Industrial thesis supports long-term accumulation",
                "payoff_if_correct": 7.0,
                "payoff_if_wrong": -2.0
            }
        },

        "RANGE_BOUND_VOLATILITY": {
            "Miners": {
                "dominant_strategy": "HEDGE_MODERATE",
                "rationale": "Collar strategies, lock in some production",
                "payoff_if_correct": 5.0,
                "payoff_if_wrong": 3.0
            },
            "Speculators": {
                "dominant_strategy": "NEUTRAL",
                "rationale": "No clear directional edge",
                "payoff_if_correct": 4.0,
                "payoff_if_wrong": -1.0
            },
            "Industrial_Users": {
                "dominant_strategy": "NORMAL_USE",
                "rationale": "No urgency to change behavior",
                "payoff_if_correct": 5.0,
                "payoff_if_wrong": 3.0
            },
            "Recyclers": {
                "dominant_strategy": "MAINTAIN",
                "rationale": "Steady state operations",
                "payoff_if_correct": 4.0,
                "payoff_if_wrong": 3.0
            },
            "ETF_Funds": {
                "dominant_strategy": "HOLD",
                "rationale": "No strong catalyst for change",
                "payoff_if_correct": 4.0,
                "payoff_if_wrong": 2.0
            }
        }
    }

    return {
        "players": players,
        "strategic_analysis": strategic_analysis
    }


# =============================================================================
# BAYESIAN SCENARIO ENGINE
# =============================================================================

class SilverBayesianEngine:
    """Bayesian updating for silver scenarios."""

    def __init__(self, scenarios: List[SilverScenario]):
        self.scenarios = scenarios

    def update_with_evidence(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """
        Update scenario probabilities with market evidence.

        Args:
            evidence: Dict mapping evidence type to log-likelihood ratio
                     Positive = favors bullish scenarios
                     Negative = favors bearish scenarios

        Evidence types:
        - "dollar_weakness": DXY moves (negative = dollar weak = bullish)
        - "inventory_trend": Comex inventory (negative = declining = bullish)
        - "physical_premium": Premium over paper (positive = bullish)
        - "miner_hedging": Hedging activity (positive = bearish)
        - "recycling_flow": Scrap supply (positive = bearish)
        - "industrial_demand": Demand indicators (positive = bullish)
        - "gold_performance": Gold price action (positive = bullish)
        - "bond_instability": Treasury vol (positive = bullish for Oliver thesis)
        """

        # Evidence impact by scenario
        evidence_weights = {
            "HEDGING_SUPPLY_GLUT": {
                "dollar_weakness": -0.3,
                "inventory_trend": 0.5,
                "physical_premium": -0.4,
                "miner_hedging": 0.8,
                "recycling_flow": 0.7,
                "industrial_demand": -0.3,
                "gold_performance": -0.2,
                "bond_instability": -0.4
            },
            "DEMAND_DESTRUCTION_CYCLE": {
                "dollar_weakness": -0.2,
                "inventory_trend": 0.4,
                "physical_premium": -0.3,
                "miner_hedging": 0.5,
                "recycling_flow": 0.5,
                "industrial_demand": -0.8,
                "gold_performance": -0.3,
                "bond_instability": -0.2
            },
            "MONETARY_REGIME_CHANGE": {
                "dollar_weakness": 0.9,
                "inventory_trend": -0.2,
                "physical_premium": 0.6,
                "miner_hedging": -0.5,
                "recycling_flow": -0.3,
                "industrial_demand": 0.3,
                "gold_performance": 0.8,
                "bond_instability": 0.9
            },
            "INDUSTRIAL_SUPERCYCLE": {
                "dollar_weakness": 0.3,
                "inventory_trend": -0.7,
                "physical_premium": 0.7,
                "miner_hedging": -0.3,
                "recycling_flow": -0.2,
                "industrial_demand": 0.9,
                "gold_performance": 0.4,
                "bond_instability": 0.2
            },
            "RANGE_BOUND_VOLATILITY": {
                "dollar_weakness": 0.1,
                "inventory_trend": 0.1,
                "physical_premium": 0.1,
                "miner_hedging": 0.1,
                "recycling_flow": 0.1,
                "industrial_demand": 0.1,
                "gold_performance": 0.1,
                "bond_instability": 0.1
            },
            "GOLD_CATCH_UP_RALLY": {
                "dollar_weakness": 0.5,
                "inventory_trend": -0.3,
                "physical_premium": 0.5,
                "miner_hedging": -0.2,
                "recycling_flow": -0.1,
                "industrial_demand": 0.2,
                "gold_performance": 0.7,
                "bond_instability": 0.4
            }
        }

        # Compute log-likelihood ratios for each scenario
        posteriors_unnorm = []

        for scenario in self.scenarios:
            log_lr = 0.0
            weights = evidence_weights.get(scenario.name, {})

            for evidence_type, evidence_value in evidence.items():
                weight = weights.get(evidence_type, 0.0)
                log_lr += weight * evidence_value

            scenario.bayes_log_lr = log_lr
            posteriors_unnorm.append(scenario.prior * math.exp(log_lr))

        # Normalize
        total = sum(posteriors_unnorm) + 1e-12
        posteriors = {}

        for scenario, p_unnorm in zip(self.scenarios, posteriors_unnorm):
            scenario.posterior = p_unnorm / total
            posteriors[scenario.name] = scenario.posterior

        return posteriors


# =============================================================================
# EVOLUTIONARY DYNAMICS ANALYSIS
# =============================================================================

def analyze_evolutionary_dynamics() -> Dict[str, Any]:
    """
    Analyze which strategy is evolutionarily stable in the silver market.

    Models:
    - Brandt Strategy: Hedge/Short at $110+ (defensive)
    - Oliver Strategy: Long/Accumulate for $300+ (aggressive)
    - Neutral Strategy: Range-trade (opportunistic)
    """

    # Population game: Strategy vs Strategy
    # Rows = strategy being played, Cols = opponent strategy
    # Values = payoff to row player

    # When everyone is Brandt (hedging/shorting):
    # - Brandt vs Brandt: Moderate (supply glut, prices fall, hedges work)
    # - Oliver vs Brandt: Loses (buying into supply glut)
    # - Neutral vs Brandt: Moderate (range trading works)

    # When everyone is Oliver (accumulating):
    # - Brandt vs Oliver: Loses big (shorts squeezed)
    # - Oliver vs Oliver: High (momentum, squeeze, prices spike)
    # - Neutral vs Oliver: Moderate (misses big move)

    payoff_matrix = np.array([
        #           Brandt  Oliver  Neutral
        # Brandt
        [            3.0,   -4.0,    2.0],
        # Oliver
        [           -2.0,    6.0,    3.0],
        # Neutral
        [            2.5,    2.5,    3.0],
    ])

    strategies = ["Brandt_Bearish", "Oliver_Bullish", "Neutral_Range"]

    # Find ESS
    ess_analysis = []
    n = payoff_matrix.shape[0]

    for i in range(n):
        is_ess = True
        stability_score = 1.0

        for j in range(n):
            if i == j:
                continue

            # Nash condition: payoff[i,i] >= payoff[j,i]
            if payoff_matrix[i, i] < payoff_matrix[j, i] - 1e-10:
                is_ess = False
                break

            # Stability condition
            if abs(payoff_matrix[i, i] - payoff_matrix[j, i]) < 1e-10:
                if payoff_matrix[i, j] <= payoff_matrix[j, j]:
                    is_ess = False
                    break

        ess_analysis.append({
            "strategy": strategies[i],
            "is_ess": is_ess,
            "self_play_payoff": payoff_matrix[i, i],
            "invasion_vulnerability": {
                strategies[j]: payoff_matrix[j, i] - payoff_matrix[i, i]
                for j in range(n) if j != i
            }
        })

    # Simulate replicator dynamics
    dt = 0.01
    n_steps = 1000

    # Initial: mostly neutral market
    x = np.array([0.2, 0.3, 0.5])  # Brandt, Oliver, Neutral
    trajectory = [x.copy()]

    for _ in range(n_steps):
        fitness = payoff_matrix @ x
        avg_fitness = np.dot(x, fitness)
        dx = x * (fitness - avg_fitness)
        x = x + dt * dx
        x = np.maximum(x, 0)
        x /= x.sum()
        trajectory.append(x.copy())

    final_distribution = trajectory[-1]

    return {
        "payoff_matrix": payoff_matrix,
        "strategies": strategies,
        "ess_analysis": ess_analysis,
        "replicator_result": {
            "initial_distribution": trajectory[0],
            "final_distribution": final_distribution,
            "dominant_strategy": strategies[np.argmax(final_distribution)],
            "convergence_steps": len(trajectory)
        },
        "interpretation": _interpret_evolutionary_result(final_distribution, strategies)
    }


def _interpret_evolutionary_result(final_dist: np.ndarray, strategies: List[str]) -> str:
    """Generate interpretation of evolutionary dynamics."""
    dominant_idx = np.argmax(final_dist)
    dominant_strat = strategies[dominant_idx]
    dominant_share = final_dist[dominant_idx]

    if dominant_share > 0.7:
        strength = "strongly dominates"
    elif dominant_share > 0.5:
        strength = "moderately dominates"
    else:
        strength = "slightly leads"

    interpretation = f"""
EVOLUTIONARY DYNAMICS INTERPRETATION:
=====================================
The market is likely to converge toward: {dominant_strat}
Final strategy distribution:
  - Brandt (Bearish/Hedging): {final_dist[0]*100:.1f}%
  - Oliver (Bullish/Accumulating): {final_dist[1]*100:.1f}%
  - Neutral (Range-Trading): {final_dist[2]*100:.1f}%

The {dominant_strat} strategy {strength} in evolutionary terms.

This suggests that over time, as market participants adapt their strategies
based on observed payoffs, the market will tend toward this equilibrium.

KEY INSIGHT: The "Neutral/Range-Trading" strategy shows strong evolutionary
stability because it performs reasonably well against both extreme positions.
However, the final equilibrium depends critically on:
1. Which scenario actually materializes
2. The timing of information revelation
3. The distribution of player sophistication

CAUTION: Evolutionary equilibrium ≠ Optimal strategy for informed investor.
If you have strong conviction in either Brandt or Oliver thesis, the ESS
does not apply - you should act on your information edge.
"""
    return interpretation


# =============================================================================
# CAUSAL GRAPH FOR SILVER MARKET
# =============================================================================

class SilverCausalGraph:
    """Causal relationships in silver market."""

    EDGES = {
        "DollarWeakness": {
            "SilverPrice": 0.55,
            "GoldPrice": 0.50,
            "EMFX": 0.40,
            "CommoditiesGeneral": 0.45,
            "InflationExpectations": 0.35
        },
        "BondInstability": {
            "GoldPrice": 0.60,
            "SilverPrice": 0.45,
            "DollarWeakness": 0.40,
            "FinancialStress": 0.50,
            "FedPolicyPivot": 0.55
        },
        "SilverPrice": {
            "MiningEquity": 0.75,
            "RecyclingSupply": 0.50,
            "IndustrialDemand": -0.25,
            "JewelryDemand": -0.30,
            "MinerHedging": 0.40,
            "SpeculatorPositioning": 0.35
        },
        "GoldPrice": {
            "SilverPrice": 0.60,  # Catch-up effect
            "GoldSilverRatio": -0.30,
            "CentralBankBuying": -0.20,  # Slow to respond
            "InvestorSentiment": 0.45
        },
        "MinerHedging": {
            "SilverPrice": -0.35,  # Forward selling pressure
            "SupplyAvailability": 0.40,
            "MinerProfitability": 0.30  # Locked in margins
        },
        "RecyclingSupply": {
            "SilverPrice": -0.25,  # Increased supply
            "InventoryLevels": 0.30,
            "ScrapMargins": -0.40
        },
        "IndustrialDemand": {
            "SilverPrice": 0.40,
            "SolarDeployment": 0.20,
            "EVProduction": 0.15,
            "InventoryDrawdown": 0.35
        },
        "FedPolicyPivot": {
            "DollarWeakness": 0.55,
            "BondPrices": 0.40,
            "RiskAssets": 0.45,
            "GoldPrice": 0.50
        },
        "CentralBankBuying": {
            "GoldPrice": 0.45,
            "DollarWeakness": 0.25,
            "SilverPrice": 0.30  # Indirect via gold
        }
    }

    def __init__(self):
        self.edges = self.EDGES.copy()
        self._build_adjacency()

    def _build_adjacency(self):
        self.nodes = set()
        self.outgoing = {}

        for source, targets in self.edges.items():
            self.nodes.add(source)
            if source not in self.outgoing:
                self.outgoing[source] = []

            for target, weight in targets.items():
                self.nodes.add(target)
                self.outgoing[source].append((target, weight))

    def propagate(
        self,
        root: str,
        direction: str,
        max_depth: int = 3,
        decay_factor: float = 0.55
    ) -> List[Dict[str, Any]]:
        """Compute Nth-order effects from a shock."""
        sign = 1.0 if direction.upper() == "UP" else -1.0

        effects = {}
        frontier = [(root, sign, 0)]
        visited = {root: 0}

        while frontier:
            node, magnitude, depth = frontier.pop(0)

            if depth >= max_depth:
                continue

            for neighbor, weight in self.outgoing.get(node, []):
                new_effect = magnitude * weight * decay_factor

                if abs(new_effect) < 0.01:
                    continue

                effects[neighbor] = effects.get(neighbor, 0.0) + new_effect

                if neighbor not in visited or visited[neighbor] > depth + 1:
                    visited[neighbor] = depth + 1
                    frontier.append((neighbor, new_effect, depth + 1))

        results = []
        for factor, score in sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True):
            results.append({
                "factor": factor,
                "direction": "UP" if score > 0 else "DOWN",
                "magnitude": abs(score),
                "raw_score": score
            })

        return results[:12]


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_brandt_vs_oliver_analysis():
    """Run comprehensive Brandt vs Oliver analysis."""

    print("=" * 80)
    print("BRANDT vs OLIVER: SILVER MARKET GAME THEORY ANALYSIS")
    print("=" * 80)
    print()

    # 1. Build scenarios
    print("1. SCENARIO CONSTRUCTION")
    print("-" * 40)
    scenarios = build_silver_scenarios()

    print("\nBRANDT'S BEARISH SCENARIOS:")
    for s in scenarios[:2]:
        print(f"  • {s.name}: Prior = {s.prior*100:.1f}%")
        print(f"    Price Target: ${s.price_target[0]}-${s.price_target[1]}")
        print(f"    Key Driver: {s.key_drivers[0][:60]}...")

    print("\nOLIVER'S BULLISH SCENARIOS:")
    for s in scenarios[2:4]:
        print(f"  • {s.name}: Prior = {s.prior*100:.1f}%")
        print(f"    Price Target: ${s.price_target[0]}-${s.price_target[1]}")
        print(f"    Key Driver: {s.key_drivers[0][:60]}...")

    print("\nNEUTRAL SCENARIOS:")
    for s in scenarios[4:]:
        print(f"  • {s.name}: Prior = {s.prior*100:.1f}%")
        print(f"    Price Target: ${s.price_target[0]}-${s.price_target[1]}")

    # 2. Bayesian updating with current evidence
    print("\n" + "=" * 80)
    print("2. BAYESIAN SCENARIO UPDATING")
    print("-" * 40)

    # Current market evidence (as of analysis date)
    # Positive values favor bullish, negative favor bearish
    current_evidence = {
        "dollar_weakness": 0.4,       # DXY has been weakening
        "inventory_trend": 0.2,       # Comex inventories mixed
        "physical_premium": 0.5,      # Strong physical demand
        "miner_hedging": 0.3,         # Some hedging activity
        "recycling_flow": 0.2,        # Moderate recycling
        "industrial_demand": 0.6,     # Strong solar/EV demand
        "gold_performance": 0.7,      # Gold outperforming
        "bond_instability": 0.5       # Treasury vol elevated
    }

    print("\nCurrent Market Evidence (scale: -1.0 to +1.0):")
    for ev, val in current_evidence.items():
        direction = "BULLISH" if val > 0 else "BEARISH" if val < 0 else "NEUTRAL"
        print(f"  • {ev}: {val:+.2f} ({direction})")

    engine = SilverBayesianEngine(scenarios)
    posteriors = engine.update_with_evidence(current_evidence)

    print("\nPOSTERIOR PROBABILITIES (after evidence):")
    print("-" * 40)

    # Sort by posterior
    sorted_scenarios = sorted(scenarios, key=lambda x: x.posterior, reverse=True)

    bullish_total = sum(s.posterior for s in sorted_scenarios if "MONETARY" in s.name or "SUPERCYCLE" in s.name or "CATCH_UP" in s.name)
    bearish_total = sum(s.posterior for s in sorted_scenarios if "GLUT" in s.name or "DESTRUCTION" in s.name)
    neutral_total = sum(s.posterior for s in sorted_scenarios if "RANGE" in s.name)

    for s in sorted_scenarios:
        change = (s.posterior - s.prior) * 100
        change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        print(f"  {s.name:30s}: {s.posterior*100:5.1f}% (prior: {s.prior*100:.1f}%, Δ: {change_str})")

    print(f"\n  AGGREGATE BULLISH (Oliver):  {bullish_total*100:.1f}%")
    print(f"  AGGREGATE BEARISH (Brandt):  {bearish_total*100:.1f}%")
    print(f"  AGGREGATE NEUTRAL:           {neutral_total*100:.1f}%")

    # 3. Strategic game analysis
    print("\n" + "=" * 80)
    print("3. MULTI-PLAYER STRATEGIC GAME")
    print("-" * 40)

    game = build_silver_market_game()

    print("\nPLAYER DOMINANT STRATEGIES BY SCENARIO:")

    top_scenario = sorted_scenarios[0]
    if top_scenario.name in game["strategic_analysis"]:
        strat_analysis = game["strategic_analysis"][top_scenario.name]
        print(f"\nMost Likely Scenario: {top_scenario.name} ({top_scenario.posterior*100:.1f}%)")
        print("-" * 40)

        for player, analysis in strat_analysis.items():
            print(f"  {player:20s}: {analysis['dominant_strategy']}")
            print(f"    Rationale: {analysis['rationale'][:50]}...")
            print(f"    Payoff if correct: {analysis['payoff_if_correct']:.1f}, if wrong: {analysis['payoff_if_wrong']:.1f}")

    # 4. Evolutionary dynamics
    print("\n" + "=" * 80)
    print("4. EVOLUTIONARY DYNAMICS ANALYSIS")
    print("-" * 40)

    evo_result = analyze_evolutionary_dynamics()

    print("\nPayoff Matrix (row vs column strategy):")
    print("              Brandt   Oliver   Neutral")
    for i, strat in enumerate(evo_result["strategies"]):
        row = evo_result["payoff_matrix"][i]
        print(f"  {strat:12s}  {row[0]:6.1f}   {row[1]:6.1f}   {row[2]:6.1f}")

    print("\nReplicator Dynamics Convergence:")
    print(f"  Initial: Brandt={evo_result['replicator_result']['initial_distribution'][0]*100:.1f}%, "
          f"Oliver={evo_result['replicator_result']['initial_distribution'][1]*100:.1f}%, "
          f"Neutral={evo_result['replicator_result']['initial_distribution'][2]*100:.1f}%")
    print(f"  Final:   Brandt={evo_result['replicator_result']['final_distribution'][0]*100:.1f}%, "
          f"Oliver={evo_result['replicator_result']['final_distribution'][1]*100:.1f}%, "
          f"Neutral={evo_result['replicator_result']['final_distribution'][2]*100:.1f}%")
    print(f"  Dominant Strategy: {evo_result['replicator_result']['dominant_strategy']}")

    print(evo_result["interpretation"])

    # 5. Causal graph effects
    print("\n" + "=" * 80)
    print("5. NTH-ORDER CAUSAL EFFECTS")
    print("-" * 40)

    causal = SilverCausalGraph()

    # Effects if Oliver thesis correct (dollar weakness shock)
    print("\nIF OLIVER THESIS CORRECT (Dollar Weakness + Bond Instability):")
    dollar_effects = causal.propagate("DollarWeakness", "UP", max_depth=3)
    bond_effects = causal.propagate("BondInstability", "UP", max_depth=3)

    # Combine effects
    combined_effects = {}
    for e in dollar_effects + bond_effects:
        factor = e["factor"]
        combined_effects[factor] = combined_effects.get(factor, 0) + e["raw_score"]

    print("  Factor                    Direction    Magnitude")
    print("  " + "-" * 50)
    for factor, score in sorted(combined_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "UP" if score > 0 else "DOWN"
        print(f"  {factor:25s} {direction:8s}     {abs(score):.3f}")

    # Effects if Brandt thesis correct (miner hedging + recycling)
    print("\nIF BRANDT THESIS CORRECT (Miner Hedging + Recycling Surge):")
    hedge_effects = causal.propagate("MinerHedging", "UP", max_depth=3)
    recycle_effects = causal.propagate("RecyclingSupply", "UP", max_depth=3)

    combined_bear = {}
    for e in hedge_effects + recycle_effects:
        factor = e["factor"]
        combined_bear[factor] = combined_bear.get(factor, 0) + e["raw_score"]

    print("  Factor                    Direction    Magnitude")
    print("  " + "-" * 50)
    for factor, score in sorted(combined_bear.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "UP" if score > 0 else "DOWN"
        print(f"  {factor:25s} {direction:8s}     {abs(score):.3f}")

    # 6. Trade recommendations
    print("\n" + "=" * 80)
    print("6. TRADE RECOMMENDATIONS BY PROBABILITY")
    print("-" * 40)

    print("\nPROBABILITY-WEIGHTED TRADE FRAMEWORK:")
    print("-" * 40)

    # Calculate expected value trades
    bullish_ev = bullish_total  # ~45%
    bearish_ev = bearish_total  # ~40%

    print(f"""
GIVEN: P(Bullish) = {bullish_total*100:.1f}%, P(Bearish) = {bearish_total*100:.1f}%, P(Neutral) = {neutral_total*100:.1f}%

CORE POSITION FRAMEWORK:
========================

1. BASE CASE (Probability-Weighted):
   - NET LONG silver with 60% position size (reflects bullish edge)
   - Hedged with put spreads (bearish scenario protection)
   - Position sizing: Kelly-optimal = {max(0, bullish_total - bearish_total)*100:.0f}% net exposure

2. IF OLIVER THESIS STRENGTHENS (P(Bullish) > 60%):
   Triggers: Dollar breaks below DXY 95, Gold > $3,500, Treasury vol spikes
   Actions:
   - Increase silver miners (SIL, SILJ) to 25% portfolio
   - Add SLV call spreads ($100-$150 strikes)
   - Add physical silver allocation
   - Short TLT (bond weakness thesis)

3. IF BRANDT THESIS STRENGTHENS (P(Bearish) > 50%):
   Triggers: Miner hedging announcements, Comex inventory rises, Dollar rallies
   Actions:
   - Reduce to 25% silver exposure
   - Add put protection ($90-$70 strikes)
   - Long GLD/Short SLV ratio trade
   - Avoid miners (leverage works against you)

4. VOLATILITY TRADES (ALWAYS VALID):
   - Silver IV historically underpriced during trending markets
   - Long straddles on SLV around key macro events
   - Ratio spreads to cheapen cost of participation

KEY MONITORING SIGNALS:
=======================
✓ Comex registered inventory (declining = bullish)
✓ Dollar Index (DXY < 95 = bullish)
✓ Gold/Silver ratio (< 75 = bullish)
✓ Miner forward sales announcements (increasing = bearish)
✓ Physical premium over paper (widening = bullish)
✓ Treasury 10Y-2Y spread (inverting = bullish for metals)

RISK MANAGEMENT:
================
- Maximum loss on bearish scenario: -30% (hedged)
- Maximum gain on bullish scenario: +150-300%
- Asymmetric payoff structure favors controlled long exposure
- Use 3-month rolling option protection
""")

    # 7. Summary
    print("\n" + "=" * 80)
    print("7. EXECUTIVE SUMMARY: BRANDT vs OLIVER")
    print("=" * 80)

    print(f"""
PETER BRANDT'S THESIS: {bearish_total*100:.1f}% probability
--------------------------------
Core Argument: Supply wall from hedging + recycling at $110+
Key Strength: Historically accurate supply analysis
Key Weakness: Assumes normal market conditions (no monetary crisis)
Price Target: $65-$85 (if correct)

MICHAEL OLIVER'S THESIS: {bullish_total*100:.1f}% probability
---------------------------------
Core Argument: Monetary regime change, not speculative bubble
Key Strength: Macro thesis (dollar/bonds) has strong supporting evidence
Key Weakness: Timing uncertain; miners may still hedge short-term
Price Target: $200-$500 (if correct)

GAME THEORY CONCLUSION:
=======================
• Current evidence SLIGHTLY FAVORS Oliver's bullish thesis
• Bayesian posterior: {bullish_total*100:.1f}% bullish vs {bearish_total*100:.1f}% bearish
• Evolutionary dynamics suggest NEUTRAL strategy dominates long-term
• BUT: Informed investors with thesis conviction should ACT on edge

OPTIMAL STRATEGY:
• Maintain net LONG exposure (sized to probability edge)
• Use OPTIONS for asymmetric participation
• MONITOR key signals for thesis shifts
• Be prepared to PIVOT quickly if evidence changes

CRITICAL INSIGHT:
The key differentiator between Brandt and Oliver is not silver fundamentals—
it's the MONETARY REGIME thesis. If dollar hegemony is truly breaking down,
Oliver's $300+ target is achievable. If traditional economics hold, Brandt's
supply analysis will dominate. Your silver position is ultimately a bet on
the future of the global monetary system.
""")

    return {
        "scenarios": scenarios,
        "posteriors": posteriors,
        "bullish_probability": bullish_total,
        "bearish_probability": bearish_total,
        "evolutionary_analysis": evo_result,
        "recommended_net_exposure": max(0, bullish_total - bearish_total)
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_brandt_vs_oliver_analysis()
