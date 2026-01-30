#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
KEVIN WARSH FED CHAIR: MARKET IMPLICATIONS ANALYSIS
================================================================================

Comprehensive Game-Theoretic Analysis of Kevin Warsh as Federal Reserve Chair

BACKGROUND ON KEVIN WARSH:
- Former Fed Governor (2006-2011), youngest in history at appointment
- Known as a monetary policy "hawk" - inflation fighter
- Goldman Sachs background (M&A)
- Close to Trump administration
- Critical of QE and balance sheet expansion
- Advocates for rules-based monetary policy (Taylor Rule advocate)
- Favors faster normalization of interest rates
- Skeptical of forward guidance as policy tool

KEY POLICY IMPLICATIONS:
1. Higher terminal rates expected
2. Faster balance sheet reduction (QT)
3. Less dovish put / more pain tolerance
4. Stronger dollar bias
5. Less accommodation for risk assets
6. More inflation-fighting credibility

Analysis Framework:
- Multi-player strategic game (Fed, Markets, Treasury, Foreign CBs)
- Bayesian scenario analysis with probability updating
- Nth-order causal effects across asset classes
- Market positioning and equilibrium analysis

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from enum import Enum, auto
import math

# =============================================================================
# WARSH POLICY SCENARIOS
# =============================================================================

@dataclass
class PolicyScenario:
    """Represents a Fed policy scenario under Warsh."""
    name: str
    description: str
    prior: float
    posterior: float = 0.0
    policy_stance: str = ""
    terminal_rate: Tuple[float, float] = (0, 0)
    balance_sheet: str = ""
    timeframe: str = "12-24 months"
    key_drivers: List[str] = field(default_factory=list)
    confirming_signals: List[str] = field(default_factory=list)
    asset_implications: Dict[str, str] = field(default_factory=dict)
    nth_order_effects: Dict[str, float] = field(default_factory=dict)


def build_warsh_scenarios() -> List[PolicyScenario]:
    """Build scenario tree for Warsh Fed Chair."""

    scenarios = [
        PolicyScenario(
            name="HAWKISH_VOLCKER_REDUX",
            description="Warsh pursues aggressive inflation fighting, accepts recession risk",
            prior=0.25,
            policy_stance="Very Hawkish",
            terminal_rate=(5.5, 6.5),
            balance_sheet="Accelerated QT, $100B+/month runoff",
            timeframe="6-18 months",
            key_drivers=[
                "Warsh's stated preference for rules-based policy",
                "Inflation remains sticky above 3%",
                "Trump willing to accept short-term pain for long-term gains",
                "Warsh views credibility restoration as paramount",
                "Historical precedent: Volcker success"
            ],
            confirming_signals=[
                "Warsh speeches emphasize inflation fighting",
                "FOMC dots shift up significantly",
                "QT pace accelerates beyond $95B/month",
                "Fed funds futures reprice higher",
                "No pivot language despite market stress"
            ],
            asset_implications={
                "Equities": "BEARISH - P/E compression, growth stocks crushed",
                "Bonds": "MIXED - Short pain, long-term gain",
                "Dollar": "BULLISH - Rate differentials widen",
                "Gold": "BEARISH short-term, BULLISH long-term (policy error risk)",
                "Commodities": "BEARISH - Demand destruction",
                "Crypto": "BEARISH - Risk-off, liquidity drain"
            },
            nth_order_effects={
                "SPX": -0.65,
                "NDX": -0.80,
                "TLT": -0.30,
                "DXY": 0.70,
                "GLD": -0.25,
                "HYG": -0.55,
                "VIX": 0.75,
                "EMFX": -0.60,
                "Banks": 0.20,
                "REITs": -0.70,
                "Bitcoin": -0.65
            }
        ),

        PolicyScenario(
            name="HAWKISH_BUT_PRAGMATIC",
            description="Warsh leans hawkish but responds to data, gradual normalization",
            prior=0.35,
            policy_stance="Moderately Hawkish",
            terminal_rate=(4.75, 5.5),
            balance_sheet="Steady QT at $60-95B/month",
            timeframe="12-24 months",
            key_drivers=[
                "Warsh tempers ideology with pragmatism",
                "Inflation gradually declining toward 2.5%",
                "Labor market cooling but not crashing",
                "Political pressure for balanced approach",
                "Market functioning concerns limit aggression"
            ],
            confirming_signals=[
                "Warsh emphasizes data-dependency",
                "Gradual rate path, no shock therapy",
                "QT continues but with flexibility",
                "Communication emphasizes dual mandate",
                "Some acknowledgment of financial stability"
            ],
            asset_implications={
                "Equities": "NEUTRAL to SLIGHT BEARISH - Grinding, not crashing",
                "Bonds": "BEARISH - Higher for longer",
                "Dollar": "BULLISH - Relative hawkishness",
                "Gold": "NEUTRAL - Cross currents",
                "Commodities": "NEUTRAL - Supply/demand driven",
                "Crypto": "NEUTRAL - Range-bound"
            },
            nth_order_effects={
                "SPX": -0.20,
                "NDX": -0.30,
                "TLT": -0.40,
                "DXY": 0.45,
                "GLD": -0.10,
                "HYG": -0.25,
                "VIX": 0.30,
                "EMFX": -0.35,
                "Banks": 0.35,
                "REITs": -0.35,
                "Bitcoin": -0.20
            }
        ),

        PolicyScenario(
            name="CONSTRAINED_HAWK",
            description="Warsh wants to be hawkish but political/market realities constrain",
            prior=0.20,
            policy_stance="Hawkish Rhetoric, Moderate Action",
            terminal_rate=(4.25, 5.0),
            balance_sheet="QT paused or slowed under stress",
            timeframe="12-18 months",
            key_drivers=[
                "Trump administration prioritizes growth/markets",
                "Treasury issuance concerns limit QT",
                "Market stress forces accommodation",
                "Fiscal dominance limits Fed independence",
                "Warsh talks tough but acts moderately"
            ],
            confirming_signals=[
                "Gap between rhetoric and action",
                "QT pauses during market volatility",
                "Treasury/Fed coordination signals",
                "Lower terminal rate than initial guidance",
                "Forward guidance softens over time"
            ],
            asset_implications={
                "Equities": "BULLISH - Soft landing achieved",
                "Bonds": "NEUTRAL - Curve steepens",
                "Dollar": "NEUTRAL to BEARISH - Credibility questions",
                "Gold": "BULLISH - Policy inconsistency",
                "Commodities": "BULLISH - Growth maintained",
                "Crypto": "BULLISH - Risk-on returns"
            },
            nth_order_effects={
                "SPX": 0.25,
                "NDX": 0.30,
                "TLT": 0.10,
                "DXY": -0.15,
                "GLD": 0.35,
                "HYG": 0.20,
                "VIX": -0.25,
                "EMFX": 0.20,
                "Banks": 0.15,
                "REITs": 0.15,
                "Bitcoin": 0.40
            }
        ),

        PolicyScenario(
            name="CRISIS_RESPONSE_MODE",
            description="External shock forces Warsh into crisis management, ideology abandoned",
            prior=0.10,
            policy_stance="Emergency Accommodation",
            terminal_rate=(2.0, 3.5),
            balance_sheet="QE resumes, balance sheet expansion",
            timeframe="6-12 months",
            key_drivers=[
                "Major financial crisis (bank failures, credit event)",
                "Geopolitical shock (war escalation, energy crisis)",
                "Recession deeper than expected",
                "Systemic risk forces Fed hand",
                "Lender of last resort function activated"
            ],
            confirming_signals=[
                "Emergency rate cuts (50bp+)",
                "New lending facilities announced",
                "QT immediately halted, QE discussed",
                "Coordination with Treasury/FDIC",
                "2008/2020 playbook deployed"
            ],
            asset_implications={
                "Equities": "CRASH then RALLY - V-shaped if contained",
                "Bonds": "FLIGHT TO QUALITY - Yields collapse",
                "Dollar": "SPIKE then WEAKNESS - Safe haven then debasement",
                "Gold": "STRONGLY BULLISH - Crisis hedge",
                "Commodities": "CRASH then RECOVERY",
                "Crypto": "CRASH - Liquidity crisis"
            },
            nth_order_effects={
                "SPX": -0.40,  # Net effect: crash then partial recovery
                "NDX": -0.50,
                "TLT": 0.60,
                "DXY": 0.20,  # Net: spike then give back
                "GLD": 0.70,
                "HYG": -0.50,
                "VIX": 0.90,
                "EMFX": -0.45,
                "Banks": -0.60,
                "REITs": -0.40,
                "Bitcoin": -0.55
            }
        ),

        PolicyScenario(
            name="INSTITUTIONAL_REFORM_FOCUS",
            description="Warsh prioritizes Fed reform and communication over rate changes",
            prior=0.10,
            policy_stance="Neutral with Institutional Focus",
            terminal_rate=(4.0, 4.75),
            balance_sheet="Steady, rules-based approach",
            timeframe="24-36 months",
            key_drivers=[
                "Warsh emphasizes Fed transparency/accountability",
                "Rules-based framework implementation",
                "Reduced forward guidance (less market management)",
                "Focus on operational efficiency",
                "Let markets price without Fed interference"
            ],
            confirming_signals=[
                "New communication framework announced",
                "Taylor Rule references in speeches",
                "Less frequent/detailed forward guidance",
                "FOMC reform proposals",
                "Reduced market reaction to Fed statements"
            ],
            asset_implications={
                "Equities": "HIGHER VOLATILITY - Less Fed put",
                "Bonds": "STEEPER CURVE - Less manipulation",
                "Dollar": "NEUTRAL - Market-driven",
                "Gold": "NEUTRAL to BULLISH - Uncertainty premium",
                "Commodities": "NEUTRAL - Fundamentals driven",
                "Crypto": "NEUTRAL - Less correlated to Fed"
            },
            nth_order_effects={
                "SPX": -0.10,
                "NDX": -0.15,
                "TLT": -0.20,
                "DXY": 0.10,
                "GLD": 0.15,
                "HYG": -0.15,
                "VIX": 0.40,  # Higher baseline vol
                "EMFX": -0.10,
                "Banks": 0.10,
                "REITs": -0.10,
                "Bitcoin": 0.05
            }
        ),
    ]

    # Normalize priors
    total = sum(s.prior for s in scenarios)
    for s in scenarios:
        s.prior /= total
        s.posterior = s.prior

    return scenarios


# =============================================================================
# MULTI-PLAYER STRATEGIC GAME
# =============================================================================

def build_fed_policy_game() -> Dict[str, Any]:
    """
    Build N-player game for Fed policy dynamics.

    Players:
    1. Fed (Warsh) - Sets rates, manages balance sheet
    2. Treasury - Manages debt issuance, fiscal policy
    3. Markets - Positions for policy expectations
    4. Foreign Central Banks - Manage reserves, FX
    5. Banks - Manage duration, credit risk
    """

    players = {
        "Fed_Warsh": {
            "actions": ["AGGRESSIVE_TIGHTEN", "GRADUAL_TIGHTEN", "HOLD", "EASE"],
            "preferences": {
                "inflation_target": 2.0,
                "employment": "secondary",
                "financial_stability": "tertiary",
                "credibility": "paramount"
            },
            "constraints": ["Political pressure", "Market functioning", "Treasury coordination"],
            "warsh_bias": "Hawkish - prefers AGGRESSIVE_TIGHTEN or GRADUAL_TIGHTEN"
        },
        "Treasury": {
            "actions": ["FRONT_LOAD_ISSUANCE", "BALANCED_ISSUANCE", "EXTEND_DURATION"],
            "preferences": {
                "funding_cost": "minimize",
                "market_access": "maintain",
                "coordination": "with Fed"
            },
            "constraints": ["Deficit size", "Market demand", "Political calendar"]
        },
        "Markets": {
            "actions": ["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS_HEDGE"],
            "preferences": {
                "returns": "maximize risk-adjusted",
                "liquidity": "maintain",
                "fed_put": "price in"
            },
            "constraints": ["Positioning limits", "Redemptions", "Mandate"]
        },
        "Foreign_CBs": {
            "actions": ["ACCUMULATE_UST", "REDUCE_UST", "DIVERSIFY"],
            "preferences": {
                "reserve_safety": "paramount",
                "yield": "secondary",
                "diversification": "increasing"
            },
            "constraints": ["FX intervention needs", "Domestic policy", "Geopolitics"]
        },
        "Banks": {
            "actions": ["EXTEND_DURATION", "SHORTEN_DURATION", "INCREASE_CREDIT", "TIGHTEN_CREDIT"],
            "preferences": {
                "NIM": "maximize",
                "credit_quality": "maintain",
                "regulatory_capital": "optimize"
            },
            "constraints": ["Regulatory requirements", "Deposit competition", "Credit cycle"]
        }
    }

    # Strategic interactions under Warsh scenarios
    strategic_equilibria = {
        "HAWKISH_VOLCKER_REDUX": {
            "Fed_Warsh": "AGGRESSIVE_TIGHTEN",
            "Treasury": "FRONT_LOAD_ISSUANCE",  # Lock in before rates go higher
            "Markets": "RISK_OFF",
            "Foreign_CBs": "REDUCE_UST",  # Higher US rates but credit risk
            "Banks": "SHORTEN_DURATION",
            "equilibrium_stability": "Unstable - market stress may force pivot",
            "expected_duration": "6-12 months before pressure mounts"
        },
        "HAWKISH_BUT_PRAGMATIC": {
            "Fed_Warsh": "GRADUAL_TIGHTEN",
            "Treasury": "BALANCED_ISSUANCE",
            "Markets": "NEUTRAL",
            "Foreign_CBs": "ACCUMULATE_UST",  # Attractive yields
            "Banks": "SHORTEN_DURATION",
            "equilibrium_stability": "Stable - sustainable path",
            "expected_duration": "12-24 months"
        },
        "CONSTRAINED_HAWK": {
            "Fed_Warsh": "HOLD",  # Talks hawk, acts dove
            "Treasury": "EXTEND_DURATION",
            "Markets": "RISK_ON",
            "Foreign_CBs": "DIVERSIFY",  # Questions about Fed credibility
            "Banks": "INCREASE_CREDIT",
            "equilibrium_stability": "Fragile - credibility at risk",
            "expected_duration": "6-18 months before inflation resurges"
        },
        "CRISIS_RESPONSE_MODE": {
            "Fed_Warsh": "EASE",
            "Treasury": "FRONT_LOAD_ISSUANCE",  # Crisis funding
            "Markets": "CRISIS_HEDGE",
            "Foreign_CBs": "ACCUMULATE_UST",  # Flight to safety
            "Banks": "TIGHTEN_CREDIT",
            "equilibrium_stability": "Temporary - crisis resolution dependent",
            "expected_duration": "3-12 months crisis phase"
        }
    }

    return {
        "players": players,
        "strategic_equilibria": strategic_equilibria
    }


# =============================================================================
# ASSET CLASS IMPACT MATRIX
# =============================================================================

def build_asset_impact_matrix() -> Dict[str, Dict[str, Any]]:
    """
    Detailed asset class implications for Warsh Fed Chair.
    """

    assets = {
        "US_EQUITIES": {
            "ticker_proxy": "SPY/SPX",
            "warsh_impact": "NEGATIVE",
            "magnitude": -0.35,
            "channels": [
                "Higher discount rates compress P/E multiples",
                "Reduced liquidity from QT",
                "Less Fed put = higher equity risk premium",
                "Dollar strength hurts multinationals",
                "Credit tightening reduces buybacks"
            ],
            "sector_impacts": {
                "Technology": {"impact": -0.50, "reason": "Duration assets, rate sensitive"},
                "Financials": {"impact": 0.20, "reason": "NIM expansion, higher rates"},
                "Utilities": {"impact": -0.35, "reason": "Bond proxies, rate sensitive"},
                "Energy": {"impact": -0.15, "reason": "Demand concerns offset by supply"},
                "Healthcare": {"impact": -0.10, "reason": "Defensive, less rate sensitive"},
                "Consumer_Discretionary": {"impact": -0.40, "reason": "Credit-sensitive spending"},
                "Industrials": {"impact": -0.25, "reason": "Capex deferred, dollar headwind"},
                "Real_Estate": {"impact": -0.55, "reason": "Cap rate expansion, refinancing risk"}
            },
            "trading_implications": [
                "Reduce growth/tech exposure",
                "Add quality/low-beta",
                "Consider bank overweight",
                "Avoid high-multiple stocks",
                "Hedge with VIX calls"
            ]
        },

        "US_TREASURIES": {
            "ticker_proxy": "TLT/IEF",
            "warsh_impact": "NEGATIVE_SHORT_TERM",
            "magnitude": -0.25,
            "channels": [
                "Higher terminal rate expectations",
                "Accelerated QT = supply pressure",
                "Reduced foreign demand if diversifying",
                "Higher term premium with less manipulation",
                "Long-term: eventual rally on recession/credibility"
            ],
            "duration_impacts": {
                "2Y": {"impact": -0.40, "reason": "Most sensitive to Fed funds path"},
                "5Y": {"impact": -0.35, "reason": "Belly takes QT impact"},
                "10Y": {"impact": -0.25, "reason": "Term premium expansion"},
                "30Y": {"impact": -0.15, "reason": "Less sensitive, pension demand"}
            },
            "curve_implications": {
                "2s10s": "STEEPENING likely - short end more impacted",
                "5s30s": "FLATTENING initially, then STEEPENING"
            },
            "trading_implications": [
                "Underweight duration initially",
                "Steepener positions",
                "TIPS underweight (real rates higher)",
                "Eventually: long duration at higher yields"
            ]
        },

        "US_DOLLAR": {
            "ticker_proxy": "DXY",
            "warsh_impact": "POSITIVE",
            "magnitude": 0.45,
            "channels": [
                "Rate differential widens vs other DMs",
                "Hawkish credibility attracts flows",
                "Risk-off sentiment favors USD",
                "QT reduces global dollar liquidity",
                "Safe haven premium in stress"
            ],
            "cross_impacts": {
                "EUR/USD": {"impact": -0.40, "reason": "ECB less hawkish"},
                "USD/JPY": {"impact": 0.50, "reason": "BOJ still dovish"},
                "GBP/USD": {"impact": -0.30, "reason": "UK growth concerns"},
                "USD/CNY": {"impact": 0.35, "reason": "PBOC easing, capital flight risk"},
                "EM_FX": {"impact": 0.45, "reason": "Dollar strength, capital outflows"}
            },
            "trading_implications": [
                "Long DXY or USD vs majors",
                "Short EM FX (high-yielders vulnerable)",
                "Long USD/JPY (carry + rate differential)",
                "Hedge non-USD assets"
            ]
        },

        "GOLD": {
            "ticker_proxy": "GLD/GC",
            "warsh_impact": "NEGATIVE_SHORT_TERM_POSITIVE_LONG_TERM",
            "magnitude": -0.10,  # Net effect is muted
            "channels": [
                "SHORT-TERM NEGATIVE:",
                "  - Higher real rates = opportunity cost",
                "  - Stronger dollar = headwind",
                "  - Risk-off may favor cash initially",
                "LONG-TERM POSITIVE:",
                "  - Policy error risk (overtightening)",
                "  - Recession hedge",
                "  - Debt sustainability concerns",
                "  - Central bank buying continues"
            ],
            "trading_implications": [
                "Tactical SHORT on Warsh confirmation",
                "Accumulate on weakness for long-term",
                "Watch real rates for pivot signal",
                "Gold miners more volatile (leverage)"
            ]
        },

        "CREDIT": {
            "ticker_proxy": "HYG/LQD",
            "warsh_impact": "NEGATIVE",
            "magnitude": -0.40,
            "channels": [
                "Higher base rates = higher all-in yields",
                "Spread widening on risk-off",
                "Refinancing wall becomes acute",
                "Default risk rises with policy tightening",
                "Less Fed support for credit markets"
            ],
            "segment_impacts": {
                "Investment_Grade": {"impact": -0.25, "reason": "Duration hit, modest spread impact"},
                "High_Yield": {"impact": -0.55, "reason": "Spread widening, default risk"},
                "Leveraged_Loans": {"impact": -0.30, "reason": "Floating helps, credit risk hurts"},
                "EM_Credit": {"impact": -0.50, "reason": "Dollar strength, risk-off"}
            },
            "trading_implications": [
                "Reduce HY exposure",
                "Move up in quality",
                "Short HYG as hedge",
                "Avoid CCC-rated issuers",
                "Watch refinancing calendar"
            ]
        },

        "CRYPTO": {
            "ticker_proxy": "BTC/ETH",
            "warsh_impact": "NEGATIVE",
            "magnitude": -0.45,
            "channels": [
                "Liquidity withdrawal (QT) = less speculation",
                "Higher rates = opportunity cost vs yield",
                "Risk-off correlation has increased",
                "Dollar strength = headwind",
                "Regulatory environment may tighten"
            ],
            "trading_implications": [
                "Reduce crypto exposure",
                "Hawkish Fed = bearish crypto",
                "Watch BTC correlation to NDX",
                "Long-term: accumulate post-washout"
            ]
        },

        "COMMODITIES": {
            "ticker_proxy": "DJP/GSG",
            "warsh_impact": "MIXED",
            "magnitude": -0.15,
            "channels": [
                "Dollar strength = headwind",
                "Demand destruction from tightening",
                "BUT: Supply constraints still binding",
                "Energy: Geopolitics > Fed",
                "Agriculture: Weather > Fed"
            ],
            "segment_impacts": {
                "Oil": {"impact": -0.20, "reason": "Demand concerns, dollar headwind"},
                "Natural_Gas": {"impact": -0.10, "reason": "Supply-driven, less Fed sensitive"},
                "Copper": {"impact": -0.35, "reason": "Growth proxy, China demand"},
                "Gold": {"impact": -0.10, "reason": "See above"},
                "Silver": {"impact": -0.25, "reason": "Industrial + monetary headwinds"},
                "Agriculture": {"impact": -0.05, "reason": "Supply/weather driven"}
            },
            "trading_implications": [
                "Underweight cyclical commodities",
                "Energy selective (supply story)",
                "Avoid industrial metals",
                "Agriculture neutral"
            ]
        },

        "EMERGING_MARKETS": {
            "ticker_proxy": "EEM/VWO",
            "warsh_impact": "NEGATIVE",
            "magnitude": -0.50,
            "channels": [
                "Dollar strength = EM weakness",
                "Capital outflows to US yields",
                "Higher US rates = tighter EM financial conditions",
                "Commodity demand concerns",
                "Debt servicing costs rise (USD debt)"
            ],
            "region_impacts": {
                "Asia_ex_China": {"impact": -0.35, "reason": "Tech exposure, export sensitivity"},
                "China": {"impact": -0.40, "reason": "PBOC easing, capital flight risk"},
                "LatAm": {"impact": -0.55, "reason": "Commodity exposure, USD debt"},
                "EMEA": {"impact": -0.50, "reason": "Energy prices, geopolitics"}
            },
            "trading_implications": [
                "Underweight EM equities",
                "Underweight EM FX",
                "Selective on EM local bonds (high carry)",
                "Avoid high external debt countries"
            ]
        }
    }

    return assets


# =============================================================================
# BAYESIAN SCENARIO ENGINE
# =============================================================================

class WarshBayesianEngine:
    """Bayesian updating for Warsh policy scenarios."""

    def __init__(self, scenarios: List[PolicyScenario]):
        self.scenarios = scenarios

    def update_with_evidence(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """
        Update scenario probabilities with market/policy evidence.

        Evidence types:
        - "warsh_rhetoric": Hawkish speeches (+1) to dovish (-1)
        - "inflation_data": Hot (+1) to cool (-1)
        - "employment_data": Strong (+1) to weak (-1)
        - "market_stress": High stress (+1) to calm (-1)
        - "treasury_coordination": Strong (+1) to weak (-1)
        - "political_pressure": High (+1) to low (-1)
        - "global_conditions": Supportive (+1) to challenging (-1)
        """

        evidence_weights = {
            "HAWKISH_VOLCKER_REDUX": {
                "warsh_rhetoric": 0.8,
                "inflation_data": 0.7,
                "employment_data": 0.3,
                "market_stress": -0.4,  # Stress might force pivot
                "treasury_coordination": -0.3,
                "political_pressure": -0.5,
                "global_conditions": -0.2
            },
            "HAWKISH_BUT_PRAGMATIC": {
                "warsh_rhetoric": 0.4,
                "inflation_data": 0.3,
                "employment_data": 0.2,
                "market_stress": -0.2,
                "treasury_coordination": 0.3,
                "political_pressure": 0.1,
                "global_conditions": 0.2
            },
            "CONSTRAINED_HAWK": {
                "warsh_rhetoric": -0.3,
                "inflation_data": -0.2,
                "employment_data": -0.3,
                "market_stress": 0.4,
                "treasury_coordination": 0.5,
                "political_pressure": 0.7,
                "global_conditions": -0.3
            },
            "CRISIS_RESPONSE_MODE": {
                "warsh_rhetoric": -0.6,
                "inflation_data": -0.5,
                "employment_data": -0.6,
                "market_stress": 0.9,
                "treasury_coordination": 0.4,
                "political_pressure": 0.3,
                "global_conditions": -0.7
            },
            "INSTITUTIONAL_REFORM_FOCUS": {
                "warsh_rhetoric": 0.2,
                "inflation_data": 0.0,
                "employment_data": 0.0,
                "market_stress": -0.2,
                "treasury_coordination": 0.1,
                "political_pressure": -0.2,
                "global_conditions": 0.1
            }
        }

        posteriors_unnorm = []

        for scenario in self.scenarios:
            log_lr = 0.0
            weights = evidence_weights.get(scenario.name, {})

            for ev_type, ev_value in evidence.items():
                weight = weights.get(ev_type, 0.0)
                log_lr += weight * ev_value

            scenario.bayes_log_lr = log_lr
            posteriors_unnorm.append(scenario.prior * math.exp(log_lr))

        total = sum(posteriors_unnorm) + 1e-12
        posteriors = {}

        for scenario, p_unnorm in zip(self.scenarios, posteriors_unnorm):
            scenario.posterior = p_unnorm / total
            posteriors[scenario.name] = scenario.posterior

        return posteriors


# =============================================================================
# CAUSAL GRAPH FOR FED POLICY TRANSMISSION
# =============================================================================

class FedPolicyCausalGraph:
    """Causal relationships for Fed policy transmission."""

    EDGES = {
        "FedFundsRate": {
            "ShortRates": 0.95,
            "LongRates": 0.45,
            "MortgageRates": 0.60,
            "CorporateBorrowing": 0.55,
            "DXY": 0.50,
            "BankNIM": 0.40,
            "EquityMultiples": -0.45,
            "HousingDemand": -0.50
        },
        "QT_BalanceSheet": {
            "LongRates": 0.35,
            "TermPremium": 0.40,
            "BankReserves": -0.60,
            "MarketLiquidity": -0.50,
            "RiskAssets": -0.40,
            "CreditSpreads": 0.35
        },
        "DXY": {
            "EMFX": -0.65,
            "Commodities": -0.40,
            "GoldPrice": -0.35,
            "MultinationalEarnings": -0.30,
            "EMDebt": 0.40,
            "ImportPrices": -0.25
        },
        "ShortRates": {
            "MoneyMarketFlows": 0.60,
            "EquityRiskPremium": 0.35,
            "CreditConditions": 0.45,
            "ConsumerCredit": -0.40
        },
        "LongRates": {
            "MortgageRates": 0.80,
            "CorpBondYields": 0.75,
            "EquityDuration": -0.55,
            "PensionFunding": 0.30,
            "REITValuations": -0.50
        },
        "CreditSpreads": {
            "HighYield": -0.70,
            "LeveragedLoans": -0.55,
            "CapexSpending": -0.40,
            "DefaultRate": 0.60,
            "BankLending": -0.45
        },
        "MarketLiquidity": {
            "VIX": -0.55,
            "BidAskSpreads": -0.50,
            "RiskAssets": 0.45,
            "CryptoAssets": 0.50,
            "PositionUnwinds": -0.40
        },
        "EquityMultiples": {
            "SPX": 0.70,
            "NDX": 0.85,
            "GrowthStocks": 0.80,
            "ValueStocks": 0.40,
            "IPOActivity": 0.60
        },
        "HousingDemand": {
            "HomebuilderStocks": 0.65,
            "REITs": 0.45,
            "ConstructionJobs": 0.50,
            "ConsumerWealth": 0.35
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
        """Compute Nth-order effects from a policy shock."""
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

        return results[:15]


# =============================================================================
# MARKET REACTION ANALYSIS
# =============================================================================

def analyze_market_reaction_to_warsh() -> Dict[str, Any]:
    """
    Analyze expected market reaction to Warsh nomination/confirmation.
    """

    # Initial market reaction (announcement effect)
    initial_reaction = {
        "timeframe": "1-5 days post-announcement",
        "expected_moves": {
            "SPX": {"move": "-2% to -4%", "confidence": "High", "reason": "Hawkish repricing"},
            "NDX": {"move": "-3% to -5%", "confidence": "High", "reason": "Duration/growth hit"},
            "TLT": {"move": "-2% to -4%", "confidence": "High", "reason": "Yield curve reprices higher"},
            "2Y_Yield": {"move": "+15-30bp", "confidence": "Very High", "reason": "Terminal rate repricing"},
            "10Y_Yield": {"move": "+10-20bp", "confidence": "High", "reason": "Term premium expansion"},
            "DXY": {"move": "+1% to +2%", "confidence": "High", "reason": "Rate differential widens"},
            "GLD": {"move": "-2% to -4%", "confidence": "Medium", "reason": "Real rates higher"},
            "HYG": {"move": "-1% to -2%", "confidence": "High", "reason": "Spread widening"},
            "BTC": {"move": "-5% to -10%", "confidence": "Medium", "reason": "Risk-off, liquidity concerns"},
            "VIX": {"move": "+3 to +5 points", "confidence": "High", "reason": "Uncertainty spike"}
        }
    }

    # Medium-term positioning shifts
    positioning_shifts = {
        "timeframe": "1-3 months post-confirmation",
        "institutional_flows": {
            "Equity_Funds": "Reduce duration/growth, add quality/value",
            "Bond_Funds": "Shorten duration, add floating rate",
            "Hedge_Funds": "Increase USD longs, reduce EM, add vol",
            "Pensions": "May extend duration at higher yields",
            "Foreign_CBs": "Mixed - yield attractive but diversification ongoing",
            "Retail": "Likely late to react, may chase strength in USD"
        },
        "sector_rotation": {
            "Overweight": ["Financials", "Healthcare", "Energy", "Quality Factor"],
            "Underweight": ["Technology", "REITs", "Utilities", "Consumer Discretionary"],
            "Neutral": ["Industrials", "Materials", "Staples"]
        }
    }

    # Risk scenarios
    risk_scenarios = {
        "UPSIDE_RISK": {
            "description": "Warsh more dovish than expected, soft landing achieved",
            "probability": 0.20,
            "market_impact": {
                "SPX": "+10-15%",
                "TLT": "+5-10%",
                "GLD": "-5%",
                "DXY": "-3%"
            }
        },
        "BASE_CASE": {
            "description": "Hawkish but pragmatic, moderate tightening",
            "probability": 0.45,
            "market_impact": {
                "SPX": "-5% to flat",
                "TLT": "-5% to -10%",
                "GLD": "-5% to +5%",
                "DXY": "+3% to +7%"
            }
        },
        "DOWNSIDE_RISK": {
            "description": "Aggressive Volcker-style tightening, recession",
            "probability": 0.25,
            "market_impact": {
                "SPX": "-15% to -25%",
                "TLT": "Flat to +5% (flight to quality)",
                "GLD": "+10% to +20%",
                "DXY": "+5% to +10%"
            }
        },
        "TAIL_RISK": {
            "description": "Policy error triggers financial crisis",
            "probability": 0.10,
            "market_impact": {
                "SPX": "-30% to -40%",
                "TLT": "+15% to +25%",
                "GLD": "+25% to +40%",
                "DXY": "Spike then collapse"
            }
        }
    }

    return {
        "initial_reaction": initial_reaction,
        "positioning_shifts": positioning_shifts,
        "risk_scenarios": risk_scenarios
    }


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_warsh_fed_analysis():
    """Run comprehensive Warsh Fed Chair analysis."""

    print("=" * 80)
    print("KEVIN WARSH FED CHAIR: COMPREHENSIVE MARKET ANALYSIS")
    print("=" * 80)
    print()

    # 1. Background
    print("1. KEVIN WARSH PROFILE & POLICY EXPECTATIONS")
    print("-" * 60)
    print("""
BACKGROUND:
• Former Fed Governor (2006-2011) - youngest at appointment
• Goldman Sachs background (M&A investment banking)
• Known monetary policy HAWK
• Critical of QE, balance sheet expansion
• Advocates rules-based policy (Taylor Rule)
• Close to Trump administration
• Favors faster rate normalization

EXPECTED POLICY BIAS:
• Terminal Rate: HIGHER than current market pricing
• Balance Sheet: FASTER QT, less accommodation
• Forward Guidance: LESS hand-holding of markets
• Financial Stability: LESS "Fed Put" protection
• Inflation: PRIMARY focus, willing to accept pain
• Employment: SECONDARY consideration
""")

    # 2. Build and update scenarios
    print("\n" + "=" * 80)
    print("2. POLICY SCENARIO ANALYSIS")
    print("-" * 60)

    scenarios = build_warsh_scenarios()

    # Current evidence assessment
    current_evidence = {
        "warsh_rhetoric": 0.6,       # Hawkish statements expected
        "inflation_data": 0.3,       # Inflation sticky but moderating
        "employment_data": 0.4,      # Labor market still strong
        "market_stress": -0.2,       # Markets relatively calm
        "treasury_coordination": 0.2, # Some coordination expected
        "political_pressure": 0.3,   # Trump wants results
        "global_conditions": -0.1    # Global uncertainty moderate
    }

    print("\nCurrent Evidence Assessment (scale: -1.0 to +1.0):")
    for ev, val in current_evidence.items():
        direction = "HAWKISH" if val > 0 else "DOVISH" if val < 0 else "NEUTRAL"
        print(f"  • {ev:25s}: {val:+.2f} ({direction})")

    engine = WarshBayesianEngine(scenarios)
    posteriors = engine.update_with_evidence(current_evidence)

    print("\nSCENARIO PROBABILITIES (Bayesian Posterior):")
    print("-" * 60)

    sorted_scenarios = sorted(scenarios, key=lambda x: x.posterior, reverse=True)

    for s in sorted_scenarios:
        change = (s.posterior - s.prior) * 100
        change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        print(f"\n  {s.name}")
        print(f"    Probability: {s.posterior*100:5.1f}% (prior: {s.prior*100:.1f}%, Δ: {change_str})")
        print(f"    Policy Stance: {s.policy_stance}")
        print(f"    Terminal Rate: {s.terminal_rate[0]:.2f}% - {s.terminal_rate[1]:.2f}%")
        print(f"    Balance Sheet: {s.balance_sheet}")

    # Aggregate probabilities
    hawkish_total = sum(s.posterior for s in scenarios if "HAWKISH" in s.name)
    dovish_total = sum(s.posterior for s in scenarios if "CONSTRAINED" in s.name or "CRISIS" in s.name)
    neutral_total = sum(s.posterior for s in scenarios if "REFORM" in s.name)

    print(f"\n  AGGREGATE HAWKISH:  {hawkish_total*100:.1f}%")
    print(f"  AGGREGATE DOVISH:   {dovish_total*100:.1f}%")
    print(f"  AGGREGATE NEUTRAL:  {neutral_total*100:.1f}%")

    # 3. Strategic game analysis
    print("\n" + "=" * 80)
    print("3. MULTI-PLAYER STRATEGIC GAME")
    print("-" * 60)

    game = build_fed_policy_game()

    print("\nPLAYER EQUILIBRIUM STRATEGIES BY SCENARIO:")

    top_scenario = sorted_scenarios[0]
    if top_scenario.name in game["strategic_equilibria"]:
        eq = game["strategic_equilibria"][top_scenario.name]
        print(f"\nMost Likely: {top_scenario.name} ({top_scenario.posterior*100:.1f}%)")
        print("-" * 50)
        for player, action in eq.items():
            if player not in ["equilibrium_stability", "expected_duration"]:
                print(f"  {player:20s}: {action}")
        print(f"\n  Stability: {eq['equilibrium_stability']}")
        print(f"  Duration:  {eq['expected_duration']}")

    # 4. Asset class impacts
    print("\n" + "=" * 80)
    print("4. ASSET CLASS IMPLICATIONS")
    print("-" * 60)

    assets = build_asset_impact_matrix()

    print("\nPROBABILITY-WEIGHTED ASSET IMPACTS:")
    print("-" * 50)

    # Compute probability-weighted impacts
    weighted_impacts = {}
    for s in scenarios:
        for asset, impact in s.nth_order_effects.items():
            weighted_impacts[asset] = weighted_impacts.get(asset, 0) + s.posterior * impact

    print(f"{'Asset':<15} {'Impact':<10} {'Direction':<12} {'Confidence'}")
    print("-" * 50)

    for asset, impact in sorted(weighted_impacts.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "BULLISH" if impact > 0.1 else "BEARISH" if impact < -0.1 else "NEUTRAL"
        confidence = "High" if abs(impact) > 0.3 else "Medium" if abs(impact) > 0.15 else "Low"
        print(f"{asset:<15} {impact:+.2f}      {direction:<12} {confidence}")

    # Detailed asset analysis
    print("\n" + "-" * 60)
    print("DETAILED ASSET ANALYSIS:")
    print("-" * 60)

    for asset_name, asset_data in list(assets.items())[:4]:  # Top 4 assets
        print(f"\n{asset_name} ({asset_data['ticker_proxy']}):")
        print(f"  Overall Impact: {asset_data['warsh_impact']}")
        print(f"  Magnitude: {asset_data['magnitude']:+.2f}")
        print("  Key Channels:")
        for channel in asset_data['channels'][:3]:
            print(f"    • {channel[:65]}...")
        print("  Trading Implications:")
        for impl in asset_data['trading_implications'][:2]:
            print(f"    → {impl}")

    # 5. Causal effects
    print("\n" + "=" * 80)
    print("5. NTH-ORDER CAUSAL EFFECTS")
    print("-" * 60)

    causal = FedPolicyCausalGraph()

    print("\nFED FUNDS RATE +100bp SHOCK (Warsh Tightening):")
    rate_effects = causal.propagate("FedFundsRate", "UP", max_depth=3)

    print(f"  {'Factor':<25} {'Direction':<10} {'Magnitude'}")
    print("  " + "-" * 50)
    for e in rate_effects[:12]:
        print(f"  {e['factor']:<25} {e['direction']:<10} {e['magnitude']:.3f}")

    print("\nQT ACCELERATION SHOCK (Balance Sheet Reduction):")
    qt_effects = causal.propagate("QT_BalanceSheet", "UP", max_depth=3)

    print(f"  {'Factor':<25} {'Direction':<10} {'Magnitude'}")
    print("  " + "-" * 50)
    for e in qt_effects[:10]:
        print(f"  {e['factor']:<25} {e['direction']:<10} {e['magnitude']:.3f}")

    # 6. Market reaction analysis
    print("\n" + "=" * 80)
    print("6. EXPECTED MARKET REACTION")
    print("-" * 60)

    reaction = analyze_market_reaction_to_warsh()

    print("\nINITIAL REACTION (1-5 days post-announcement):")
    print("-" * 50)
    for asset, data in reaction["initial_reaction"]["expected_moves"].items():
        print(f"  {asset:<12}: {data['move']:<15} ({data['confidence']} confidence)")
        print(f"               Reason: {data['reason']}")

    print("\nPOSITIONING SHIFTS (1-3 months):")
    print("-" * 50)
    print("\n  Sector Rotation:")
    print(f"    OVERWEIGHT:  {', '.join(reaction['positioning_shifts']['sector_rotation']['Overweight'])}")
    print(f"    UNDERWEIGHT: {', '.join(reaction['positioning_shifts']['sector_rotation']['Underweight'])}")

    print("\n  Institutional Flows:")
    for investor, action in list(reaction["positioning_shifts"]["institutional_flows"].items())[:4]:
        print(f"    {investor}: {action}")

    print("\nRISK SCENARIOS:")
    print("-" * 50)
    for scenario_name, data in reaction["risk_scenarios"].items():
        print(f"\n  {scenario_name} (P = {data['probability']*100:.0f}%)")
        print(f"    {data['description']}")
        for asset, impact in list(data["market_impact"].items())[:3]:
            print(f"      {asset}: {impact}")

    # 7. Trading recommendations
    print("\n" + "=" * 80)
    print("7. TRADING RECOMMENDATIONS")
    print("-" * 60)

    print(f"""
PROBABILITY-WEIGHTED STRATEGY (Hawkish = {hawkish_total*100:.0f}%)
================================================================

CORE POSITIONS:
---------------
1. UNDERWEIGHT EQUITIES
   • Reduce SPY/QQQ exposure by 20-30%
   • Rotate from Growth → Value/Quality
   • Add defensive sectors (Healthcare, Staples)
   • Consider SPY puts for tail protection

2. SHORT DURATION FIXED INCOME
   • Reduce TLT/long bond exposure
   • Add floating rate (FLOT, bank loans)
   • Consider 2s10s steepener
   • Wait for higher yields to extend

3. LONG USD
   • Long DXY or USD vs EUR, JPY
   • Short EM FX (selective)
   • Hedge non-USD assets

4. UNDERWEIGHT CREDIT
   • Reduce HY exposure (HYG, JNK)
   • Move up in quality (IG only)
   • Avoid CCC-rated names

5. TACTICAL GOLD
   • Short-term: Underweight (real rates rising)
   • Long-term: Accumulate on weakness
   • Miners more volatile, use for trading

6. UNDERWEIGHT EM
   • Reduce EM equity (EEM, VWO)
   • Reduce EM FX exposure
   • Selective on local bonds (high carry)

7. VOLATILITY
   • Own VIX calls as tail hedge
   • Sell vol selectively on spikes
   • Expect higher baseline vol

HEDGING STRATEGY:
-----------------
• Portfolio puts (SPY/QQQ 10% OTM)
• VIX call spreads (20-30 strikes)
• USD longs as natural hedge
• Quality factor tilt

KEY MONITORING SIGNALS:
-----------------------
✓ Fed funds futures repricing
✓ 2Y Treasury yield moves
✓ DXY strength/weakness
✓ Credit spread direction (HY OAS)
✓ VIX term structure
✓ Bank stock performance
✓ Housing data (mortgage apps)
✓ Employment data trajectory

SCENARIO TRIGGERS:
------------------
→ If Warsh MORE hawkish than expected:
  Double down on short duration, long USD, underweight equities

→ If Warsh LESS hawkish than expected (constrained):
  Reduce hedges, add risk selectively, watch gold

→ If CRISIS emerges:
  Pivot to flight-to-quality: Long TLT, Long Gold, Reduce all risk
""")

    # 8. Executive Summary
    print("\n" + "=" * 80)
    print("8. EXECUTIVE SUMMARY")
    print("=" * 80)

    print(f"""
KEVIN WARSH AS FED CHAIR: KEY TAKEAWAYS
========================================

1. POLICY DIRECTION: HAWKISH
   • {hawkish_total*100:.0f}% probability of hawkish outcomes
   • Terminal rate likely 5.0-6.0% (higher than current pricing)
   • Faster QT, less market accommodation
   • Reduced "Fed Put" protection

2. MARKET IMPACT: RISK-OFF
   • Equities: BEARISH (weighted impact: {weighted_impacts.get('SPX', 0):+.2f})
   • Bonds: BEARISH short-term, yields higher
   • Dollar: BULLISH (weighted impact: {weighted_impacts.get('DXY', 0):+.2f})
   • Gold: MIXED (real rates vs crisis hedge)
   • Credit: BEARISH (spreads wider)
   • EM: BEARISH (dollar strength, capital flight)

3. INITIAL REACTION EXPECTED:
   • SPX: -2% to -4%
   • 2Y Yield: +15-30bp
   • DXY: +1% to +2%
   • VIX: +3 to +5 points

4. STRATEGIC POSITIONING:
   • Underweight risk assets
   • Short duration
   • Long USD
   • Own volatility hedges
   • Quality over beta

5. CRITICAL UNCERTAINTY:
   The key question is whether Warsh's hawkish ideology will be
   CONSTRAINED by:
   • Political pressure from Trump administration
   • Treasury coordination requirements
   • Market functioning concerns
   • Global economic conditions

   If constrained → Less hawkish than expected → Relief rally
   If unconstrained → Volcker-style tightening → Significant drawdown

BOTTOM LINE:
Position defensively. The risk/reward favors underweighting risk assets
and owning USD until we see how Warsh actually governs vs. his rhetoric.
Hawkish talk is certain; hawkish action depends on constraints.
""")

    return {
        "scenarios": scenarios,
        "posteriors": posteriors,
        "hawkish_probability": hawkish_total,
        "weighted_asset_impacts": weighted_impacts,
        "market_reaction": reaction
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_warsh_fed_analysis()
