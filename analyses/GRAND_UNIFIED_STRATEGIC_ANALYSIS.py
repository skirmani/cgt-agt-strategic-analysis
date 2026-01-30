#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GRAND UNIFIED STRATEGIC ANALYSIS
================================================================================

Integrating:
1. US-China Grand Strategy Game
2. Warsh Fed Chair Appointment & Monetary Policy
3. Brandt vs Oliver Silver/Precious Metals Thesis
4. Geopolitical, Economic, and Market Dynamics

This analysis pieces together the interconnected strategic games to provide
a unified view of the global macro environment and identify further games
to investigate for a complete picture.

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum, auto
import math

# =============================================================================
# THE GRAND STRATEGIC PICTURE
# =============================================================================

def print_grand_strategic_overview():
    """Print the integrated strategic overview."""

    print("=" * 100)
    print("GRAND UNIFIED STRATEGIC ANALYSIS: THE INTERCONNECTED GAME MATRIX")
    print("=" * 100)

    print("""
THE BIG PICTURE: THREE INTERCONNECTED MEGA-GAMES
=================================================

We have analyzed three seemingly separate strategic situations:

1. WARSH FED APPOINTMENT
   - Trump appoints hawk despite wanting "low rates"
   - Game theory: Credible commitment, signaling, time inconsistency
   - Implication: Tighter US monetary policy, stronger dollar

2. US-CHINA GRAND STRATEGY
   - Technology, trade, finance, geopolitics
   - Game theory: Multi-domain repeated game, Nash equilibria
   - Implication: Decoupling acceleration, supply chain restructuring

3. SILVER MARKET (Brandt vs Oliver)
   - Supply/demand fundamentals vs monetary regime change
   - Game theory: Bayesian updating, evolutionary dynamics
   - Implication: Precious metals as monetary crisis hedge

THE KEY INSIGHT: These are NOT independent games.
================================================

They are NESTED and INTERCONNECTED through:

┌─────────────────────────────────────────────────────────────────────────┐
│                         GLOBAL MONETARY REGIME                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    US DOMESTIC POLICY                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              FED POLICY (WARSH)                          │    │   │
│  │  │  • Higher rates → Stronger USD                           │    │   │
│  │  │  • Credibility → Lower inflation expectations            │    │   │
│  │  │  • QT → Reduced global dollar liquidity                  │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                           ↓                                      │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │           US-CHINA STRATEGIC COMPETITION                 │    │   │
│  │  │  • Tech war → Supply chain restructuring                 │    │   │
│  │  │  • Trade war → Inflation pressures                       │    │   │
│  │  │  • Financial decoupling → De-dollarization pressure      │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 PRECIOUS METALS / COMMODITIES                    │   │
│  │  • If Warsh succeeds (dollar strong) → Brandt thesis favored    │   │
│  │  • If US-China escalates (system stress) → Oliver thesis rises  │   │
│  │  • If de-dollarization accelerates → Monetary metals surge      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

THE MASTER GAME: Dollar Hegemony vs Multipolar Monetary Order
""")


# =============================================================================
# THE MASTER CAUSAL GRAPH
# =============================================================================

class MasterCausalGraph:
    """
    Unified causal graph connecting all strategic domains.
    """

    UNIFIED_EDGES = {
        # FED POLICY TRANSMISSION
        "Warsh_Fed_Policy": {
            "US_Interest_Rates": 0.85,
            "Dollar_Strength": 0.60,
            "QT_Liquidity_Drain": 0.70,
            "Fed_Credibility": 0.75,
            "Inflation_Expectations": -0.65,
            "Treasury_Yields": 0.55
        },

        "US_Interest_Rates": {
            "Dollar_Strength": 0.50,
            "EM_Capital_Flows": -0.55,
            "US_Credit_Conditions": 0.60,
            "Housing_Market": -0.50,
            "Corporate_Borrowing": -0.45,
            "Gold_Opportunity_Cost": 0.40
        },

        "Dollar_Strength": {
            "US_Export_Competitiveness": -0.45,
            "EM_Debt_Burden": 0.55,
            "Commodity_Prices_USD": -0.40,
            "China_Yuan_Pressure": 0.50,
            "De_Dollarization_Incentive": 0.35,
            "Gold_Price_USD": -0.35,
            "Silver_Price_USD": -0.40
        },

        # US-CHINA DYNAMICS
        "US_China_Tech_War": {
            "Chip_Supply_Chains": -0.65,
            "China_Self_Sufficiency_Drive": 0.70,
            "Global_Tech_Bifurcation": 0.60,
            "Rare_Earth_Tensions": 0.55,
            "AI_Development_Race": 0.50,
            "Semiconductor_Prices": 0.45
        },

        "US_China_Trade_War": {
            "Global_Trade_Volume": -0.45,
            "Supply_Chain_Restructuring": 0.60,
            "Inflation_Imported_Goods": 0.40,
            "Manufacturing_Reshoring": 0.50,
            "Vietnam_Mexico_Beneficiaries": 0.55
        },

        "US_China_Financial_Decoupling": {
            "De_Dollarization_Incentive": 0.65,
            "BRICS_Currency_Efforts": 0.55,
            "China_Gold_Accumulation": 0.60,
            "Yuan_Internationalization": 0.50,
            "Capital_Controls_China": 0.45,
            "Hong_Kong_Financial_Status": -0.40
        },

        # MONETARY SYSTEM STRESS
        "De_Dollarization_Incentive": {
            "Central_Bank_Gold_Buying": 0.70,
            "BRICS_Currency_Efforts": 0.60,
            "Dollar_Reserve_Share": -0.45,
            "Alternative_Payment_Systems": 0.55,
            "Oliver_Thesis_Probability": 0.65
        },

        "Fed_Credibility": {
            "Inflation_Expectations": -0.60,
            "Treasury_Demand_Foreign": 0.50,
            "Dollar_Reserve_Status": 0.55,
            "Brandt_Thesis_Probability": 0.40
        },

        # PRECIOUS METALS DYNAMICS
        "Central_Bank_Gold_Buying": {
            "Gold_Price_USD": 0.55,
            "Silver_Price_USD": 0.40,
            "Oliver_Thesis_Probability": 0.50,
            "Monetary_Metal_Premium": 0.60
        },

        "Gold_Price_USD": {
            "Silver_Price_USD": 0.65,  # Gold leads silver
            "Mining_Equity_Valuations": 0.70,
            "Gold_Silver_Ratio": -0.30,
            "Retail_PM_Demand": 0.45
        },

        "Silver_Price_USD": {
            "Silver_Mining_Economics": 0.75,
            "Silver_Recycling_Supply": 0.50,
            "Industrial_Silver_Substitution": 0.35,
            "Miner_Hedging_Incentive": 0.55
        },

        # INDUSTRIAL DEMAND
        "Green_Energy_Transition": {
            "Silver_Industrial_Demand": 0.70,
            "Copper_Demand": 0.65,
            "Rare_Earth_Demand": 0.60,
            "Lithium_Demand": 0.75,
            "Uranium_Demand": 0.50
        },

        "AI_Buildout": {
            "Chip_Demand": 0.80,
            "Data_Center_Energy": 0.70,
            "Copper_Demand": 0.55,
            "Silver_Electronics_Demand": 0.40
        },

        # GEOPOLITICAL RISK
        "Taiwan_Tensions": {
            "Chip_Supply_Risk": 0.85,
            "US_China_Relations": -0.70,
            "Risk_Premium_Global": 0.60,
            "Defense_Spending": 0.55,
            "Gold_Safe_Haven": 0.50
        },

        "Middle_East_Tensions": {
            "Oil_Price": 0.65,
            "Gold_Safe_Haven": 0.45,
            "Inflation_Energy": 0.50,
            "Risk_Premium_Global": 0.40
        },

        "Ukraine_War": {
            "Europe_Energy_Crisis": 0.60,
            "Russia_West_Decoupling": 0.75,
            "Commodity_Supply_Disruption": 0.50,
            "Defense_Spending_NATO": 0.65,
            "De_Dollarization_Incentive": 0.45
        },

        # FISCAL DYNAMICS
        "US_Fiscal_Deficit": {
            "Treasury_Issuance": 0.85,
            "Debt_Sustainability_Concerns": 0.60,
            "Fed_Treasury_Coordination": 0.50,
            "Dollar_Long_Term_Credibility": -0.40,
            "Oliver_Thesis_Probability": 0.45
        },

        "Treasury_Issuance": {
            "Treasury_Yields": 0.55,
            "QT_Constraints": 0.50,
            "Foreign_Demand_Pressure": 0.45,
            "Bank_Balance_Sheet_Stress": 0.40
        }
    }

    def __init__(self):
        self.edges = self.UNIFIED_EDGES.copy()
        self._build_adjacency()

    def _build_adjacency(self):
        self.nodes = set()
        self.outgoing = {}
        self.incoming = {}

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

    def propagate_shock(
        self,
        root: str,
        direction: str,
        max_depth: int = 4,
        decay_factor: float = 0.60
    ) -> List[Dict[str, Any]]:
        """Propagate shock through unified causal graph."""
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

                if abs(new_effect) < 0.005:
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

        return results

    def find_feedback_loops(self) -> List[Dict[str, Any]]:
        """Identify critical feedback loops in the system."""

        loops = [
            {
                "name": "Dollar Strength - De-dollarization Loop",
                "path": ["Dollar_Strength", "De_Dollarization_Incentive",
                        "Central_Bank_Gold_Buying", "Dollar_Reserve_Status"],
                "type": "STABILIZING",
                "description": "Strong dollar creates incentive to diversify, which eventually weakens dollar"
            },
            {
                "name": "Fed Credibility - Inflation Expectations Loop",
                "path": ["Warsh_Fed_Policy", "Fed_Credibility",
                        "Inflation_Expectations", "US_Interest_Rates"],
                "type": "REINFORCING",
                "description": "Hawkish policy builds credibility, lowering inflation expectations, allowing rates to normalize"
            },
            {
                "name": "US-China Decoupling - Gold Accumulation Loop",
                "path": ["US_China_Financial_Decoupling", "China_Gold_Accumulation",
                        "Gold_Price_USD", "De_Dollarization_Incentive"],
                "type": "REINFORCING",
                "description": "Decoupling drives gold buying, raising prices, encouraging more de-dollarization"
            },
            {
                "name": "Fiscal Deficit - Dollar Credibility Loop",
                "path": ["US_Fiscal_Deficit", "Treasury_Issuance",
                        "Foreign_Demand_Pressure", "Dollar_Long_Term_Credibility"],
                "type": "DESTABILIZING",
                "description": "Rising deficits require more issuance, straining foreign demand, eroding credibility"
            },
            {
                "name": "Silver Price - Supply Response Loop",
                "path": ["Silver_Price_USD", "Silver_Recycling_Supply",
                        "Miner_Hedging_Incentive", "Silver_Price_USD"],
                "type": "STABILIZING (Brandt)",
                "description": "High prices bring supply online, capping further gains"
            }
        ]

        return loops


# =============================================================================
# INTEGRATED SCENARIO MATRIX
# =============================================================================

@dataclass
class IntegratedScenario:
    """A scenario that spans all strategic domains."""
    name: str
    description: str
    probability: float
    fed_policy: str
    us_china_dynamic: str
    monetary_regime: str
    precious_metals_outlook: str
    key_drivers: List[str]
    asset_implications: Dict[str, str]
    risk_factors: List[str]


def build_integrated_scenarios() -> List[IntegratedScenario]:
    """Build scenarios that integrate all strategic games."""

    scenarios = [
        IntegratedScenario(
            name="PAXAMERICANA_RENEWED",
            description="US successfully reasserts hegemony; Warsh controls inflation, China constrained, dollar dominance maintained",
            probability=0.20,
            fed_policy="Hawkish success - inflation controlled, soft landing",
            us_china_dynamic="US tech dominance, China contained, managed competition",
            monetary_regime="Dollar hegemony maintained, de-dollarization slows",
            precious_metals_outlook="BEARISH - Brandt thesis wins, $70-90 silver",
            key_drivers=[
                "Warsh achieves 2% inflation without recession",
                "China tech self-sufficiency fails/delayed",
                "BRICS currency efforts stall",
                "US fiscal situation stabilizes",
                "No major geopolitical escalation"
            ],
            asset_implications={
                "US_Equities": "BULLISH - Risk-on, P/E expansion",
                "US_Treasuries": "NEUTRAL - Stable yields, 4-4.5% 10Y",
                "Dollar": "BULLISH - DXY 110+",
                "Gold": "BEARISH - $2,000-2,500",
                "Silver": "BEARISH - $25-35",
                "EM_Assets": "MIXED - Selective opportunities",
                "Commodities": "NEUTRAL - Supply/demand driven",
                "Crypto": "BULLISH - Risk-on environment"
            },
            risk_factors=[
                "Requires no policy errors from Warsh",
                "Assumes China doesn't escalate Taiwan",
                "Fiscal trajectory must improve"
            ]
        ),

        IntegratedScenario(
            name="MANAGED_MULTIPOLARITY",
            description="Gradual transition to multipolar system; competition intensifies but no crisis",
            probability=0.30,
            fed_policy="Hawkish but pragmatic - 3-3.5% inflation tolerated",
            us_china_dynamic="Intense competition, partial decoupling, no hot war",
            monetary_regime="Gradual de-dollarization, BRICS alternatives emerge",
            precious_metals_outlook="MODERATELY BULLISH - $150-200 silver over 3-5 years",
            key_drivers=[
                "Warsh constrained by fiscal realities",
                "US-China cold war stabilizes",
                "BRICS payment systems gain traction",
                "Central bank gold buying continues",
                "Green energy transition accelerates"
            ],
            asset_implications={
                "US_Equities": "NEUTRAL - Range-bound, sector rotation",
                "US_Treasuries": "BEARISH - 5-6% 10Y as term premium rises",
                "Dollar": "BEARISH - DXY 90-100",
                "Gold": "BULLISH - $3,500-5,000",
                "Silver": "BULLISH - $75-150",
                "EM_Assets": "BULLISH - Beneficiaries of multipolar shift",
                "Commodities": "BULLISH - Supply constraints, demand growth",
                "Crypto": "MIXED - Regulatory uncertainty"
            },
            risk_factors=[
                "Transition could accelerate unexpectedly",
                "US fiscal situation could deteriorate faster",
                "Geopolitical flashpoints remain"
            ]
        ),

        IntegratedScenario(
            name="MONETARY_REGIME_CRISIS",
            description="Oliver thesis materializes - dollar crisis, bond instability, precious metals surge",
            probability=0.15,
            fed_policy="Crisis response - Warsh forced to pivot, QE resumes",
            us_china_dynamic="Accelerated decoupling, financial warfare",
            monetary_regime="Dollar crisis, BRICS alternatives accelerate, gold remonetization",
            precious_metals_outlook="EXTREMELY BULLISH - Oliver's $300-500 silver",
            key_drivers=[
                "Treasury market dysfunction",
                "Foreign central banks dump Treasuries",
                "Inflation expectations unanchor",
                "Fed credibility collapses",
                "BRICS gold-backed currency announced",
                "US-China financial war escalates"
            ],
            asset_implications={
                "US_Equities": "CRASH - 40-50% drawdown",
                "US_Treasuries": "CRISIS - Yields spike then Fed intervenes",
                "Dollar": "CRASH - DXY 70-80",
                "Gold": "SURGE - $8,000-10,000",
                "Silver": "SURGE - $200-500",
                "EM_Assets": "MIXED - Some benefit, others collapse",
                "Commodities": "SURGE - Dollar collapse drives prices",
                "Crypto": "VOLATILE - Flight to alternatives"
            },
            risk_factors=[
                "Self-fulfilling prophecy dynamics",
                "Policy response could stabilize",
                "Global coordination could prevent worst case"
            ]
        ),

        IntegratedScenario(
            name="STAGFLATIONARY_GRIND",
            description="Warsh fights inflation but US-China tensions keep prices elevated; growth stalls",
            probability=0.20,
            fed_policy="Hawkish but ineffective - supply-side inflation persists",
            us_china_dynamic="Escalating trade/tech war, supply chain disruptions",
            monetary_regime="Dollar maintains but under pressure, volatility elevated",
            precious_metals_outlook="BULLISH - $100-150 silver, gold $4,000+",
            key_drivers=[
                "Tariffs and reshoring drive inflation",
                "Warsh can't control supply-side pressures",
                "Growth slows but inflation stays 4%+",
                "Fiscal stimulus offset by monetary tightening",
                "Political pressure on Fed intensifies"
            ],
            asset_implications={
                "US_Equities": "BEARISH - P/E compression, earnings pressure",
                "US_Treasuries": "BEARISH - Real rates negative despite high nominals",
                "Dollar": "MIXED - Volatile, no clear trend",
                "Gold": "BULLISH - Stagflation hedge",
                "Silver": "BULLISH - Industrial + monetary demand",
                "EM_Assets": "BEARISH - Stagflation worst of both worlds",
                "Commodities": "MIXED - Supply constraints vs demand destruction",
                "Crypto": "BEARISH - Risk-off, regulatory pressure"
            },
            risk_factors=[
                "Could tip into full recession",
                "Political crisis if inflation persists",
                "Fed credibility damaged"
            ]
        ),

        IntegratedScenario(
            name="GEOPOLITICAL_SHOCK",
            description="Taiwan crisis, Middle East war, or other black swan triggers system stress",
            probability=0.15,
            fed_policy="Emergency response - all bets off",
            us_china_dynamic="Hot conflict or near-conflict, full decoupling",
            monetary_regime="Flight to safety, system stress, then reset",
            precious_metals_outlook="SPIKE then recalibration - depends on resolution",
            key_drivers=[
                "Taiwan invasion or blockade",
                "Middle East regional war (Iran)",
                "Major cyber attack on financial infrastructure",
                "Unexpected sovereign default",
                "Energy supply disruption"
            ],
            asset_implications={
                "US_Equities": "CRASH then recovery - V or L shaped",
                "US_Treasuries": "FLIGHT TO QUALITY initially",
                "Dollar": "SPIKE on safe haven, then depends on response",
                "Gold": "SURGE - Ultimate safe haven",
                "Silver": "VOLATILE - Industrial hit, monetary bid",
                "EM_Assets": "CRASH - Risk-off",
                "Commodities": "SPIKE in affected commodities",
                "Crypto": "CRASH - Liquidity crisis"
            },
            risk_factors=[
                "Inherently unpredictable",
                "Response determines duration",
                "Could trigger any other scenario"
            ]
        )
    ]

    # Normalize probabilities
    total = sum(s.probability for s in scenarios)
    for s in scenarios:
        s.probability /= total

    return scenarios


# =============================================================================
# ADDITIONAL GAMES TO INVESTIGATE
# =============================================================================

def identify_further_games() -> List[Dict[str, Any]]:
    """Identify additional strategic games that need investigation."""

    games = [
        # MONETARY/FINANCIAL GAMES
        {
            "game_name": "TREASURY AUCTION GAME",
            "priority": "CRITICAL",
            "players": ["US Treasury", "Primary Dealers", "Foreign CBs", "Domestic Investors", "Fed"],
            "description": """
            With $35T+ debt and large deficits, Treasury must refinance constantly.
            Key questions:
            - Will foreign CBs (China, Japan) continue buying?
            - Can domestic market absorb if foreigners retreat?
            - Will Fed be forced to monetize (stealth QE)?
            - How does Warsh's QT interact with Treasury issuance?

            This game determines long-term interest rates and dollar stability.
            """,
            "key_metrics": [
                "Bid-to-cover ratios at auctions",
                "Foreign official holdings (TIC data)",
                "Primary dealer inventory",
                "Fed balance sheet trajectory"
            ],
            "implications": {
                "If_Foreign_CBs_Retreat": "Yields spike, Fed forced to intervene, Oliver thesis strengthens",
                "If_Auction_Demand_Strong": "Warsh can maintain QT, Brandt thesis holds"
            }
        },

        {
            "game_name": "BRICS CURRENCY GAME",
            "priority": "HIGH",
            "players": ["China", "Russia", "India", "Brazil", "Saudi Arabia", "UAE", "US"],
            "description": """
            BRICS exploring alternative to dollar system.
            Key questions:
            - Will BRICS launch gold-backed or commodity-backed currency?
            - Can they overcome coordination problems (India-China tensions)?
            - What's the timeline for viable alternative?
            - How does US respond (sanctions, incentives)?

            This is the ultimate long-term game for dollar hegemony.
            """,
            "key_metrics": [
                "BRICS summit communiques",
                "Bilateral trade in local currencies",
                "Cross-border payment system adoption",
                "Central bank gold purchases by BRICS"
            ],
            "implications": {
                "If_BRICS_Currency_Succeeds": "Dollar loses 20-30% reserve share, gold/silver surge",
                "If_BRICS_Fails_Coordinate": "Dollar hegemony extended, status quo"
            }
        },

        {
            "game_name": "CENTRAL BANK GOLD ACCUMULATION GAME",
            "priority": "HIGH",
            "players": ["PBOC", "RBI", "CBR", "Bundesbank", "Fed", "BIS"],
            "description": """
            Central banks bought record gold in 2022-2024.
            Key questions:
            - Is this diversification or preparation for regime change?
            - What's China's true gold holdings (officially 2,200t, estimated 5,000t+)?
            - Will Western CBs follow or resist?
            - What triggers a gold remonetization event?

            Gold accumulation is the revealed preference for monetary regime change.
            """,
            "key_metrics": [
                "Monthly CB gold purchases (WGC data)",
                "China/Russia gold production retention",
                "BIS gold lending activity",
                "COMEX/LBMA inventory levels"
            ],
            "implications": {
                "If_Accumulation_Accelerates": "Gold floor rises, silver follows, Oliver thesis",
                "If_Accumulation_Slows": "Status quo, Brandt supply analysis dominates"
            }
        },

        # GEOPOLITICAL GAMES
        {
            "game_name": "TAIWAN SEMICONDUCTOR GAME",
            "priority": "CRITICAL",
            "players": ["US", "China", "Taiwan (TSMC)", "Japan", "South Korea", "EU"],
            "description": """
            TSMC produces 90%+ of advanced chips.
            Key questions:
            - Does China blockade/invade Taiwan?
            - Can US/allies build alternative capacity in time?
            - What's the deterrence equilibrium?
            - How does this affect US-China tech competition?

            This is potentially the most consequential geopolitical game.
            """,
            "key_metrics": [
                "TSMC Arizona/Japan fab progress",
                "China military exercises near Taiwan",
                "US chip inventory levels",
                "Semiconductor equipment sales to China"
            ],
            "implications": {
                "If_Taiwan_Crisis": "Global chip shortage, tech crash, defense stocks surge",
                "If_Deterrence_Holds": "Gradual decoupling, managed competition"
            }
        },

        {
            "game_name": "MIDDLE EAST ENERGY GAME",
            "priority": "HIGH",
            "players": ["Saudi Arabia", "Iran", "Israel", "US", "China", "Russia"],
            "description": """
            Middle East stability affects global energy and dollar (petrodollar).
            Key questions:
            - Does Saudi-Iran conflict escalate?
            - Will Saudis price oil in yuan?
            - What's Israel-Iran trajectory?
            - How does US manage competing alliances?

            The petrodollar system is a key pillar of dollar hegemony.
            """,
            "key_metrics": [
                "Saudi-China relationship developments",
                "Oil priced in non-dollar currencies",
                "Iran nuclear program status",
                "Red Sea/Strait of Hormuz incidents"
            ],
            "implications": {
                "If_Petrodollar_Weakens": "Dollar loses ~10-15% value, gold surges",
                "If_Status_Quo": "Energy volatility but no regime change"
            }
        },

        {
            "game_name": "EUROPEAN ENERGY/DEFENSE GAME",
            "priority": "MEDIUM",
            "players": ["EU", "Germany", "France", "UK", "Russia", "US"],
            "description": """
            Europe caught between US and Russia, energy transition pressure.
            Key questions:
            - Does Europe achieve energy independence?
            - Will EU defense spending rise to US demands?
            - How does Trump tariff Europe?
            - Does EU fragment or integrate further?

            European trajectory affects global trade and dollar alternatives.
            """,
            "key_metrics": [
                "EU defense spending as % GDP",
                "Russian gas imports",
                "Euro/Dollar exchange rate",
                "EU-China trade relationship"
            ],
            "implications": {
                "If_Europe_Strengthens": "Multipolar world, euro as partial alternative",
                "If_Europe_Fragments": "Dollar dominance by default, risk-off"
            }
        },

        # ECONOMIC/MARKET GAMES
        {
            "game_name": "CORPORATE DEBT REFINANCING GAME",
            "priority": "HIGH",
            "players": ["Corporations", "Banks", "Bond Funds", "Fed", "Private Credit"],
            "description": """
            $3T+ corporate debt needs refinancing in 2025-2027.
            Key questions:
            - Can companies refinance at higher rates?
            - Will defaults spike?
            - Does private credit fill the gap?
            - Is there a credit event brewing?

            Credit conditions determine real economy transmission.
            """,
            "key_metrics": [
                "HY spreads",
                "Default rates",
                "Private credit AUM growth",
                "Bank lending standards (SLOOS)"
            ],
            "implications": {
                "If_Refinancing_Smooth": "Soft landing, Warsh successful",
                "If_Credit_Crunch": "Recession, Fed forced to ease, risk-off"
            }
        },

        {
            "game_name": "COMMERCIAL REAL ESTATE GAME",
            "priority": "HIGH",
            "players": ["Property Owners", "Regional Banks", "CMBS Investors", "Fed", "Cities"],
            "description": """
            $1.5T+ CRE debt maturing, office values down 30-50%.
            Key questions:
            - Do regional banks face solvency crisis?
            - Does CRE become systemic risk?
            - How does Fed respond?
            - What's the extend-and-pretend timeline?

            CRE is the potential 2008-style hidden risk.
            """,
            "key_metrics": [
                "CRE cap rates",
                "Regional bank stock prices",
                "CMBS delinquencies",
                "Office vacancy rates"
            ],
            "implications": {
                "If_CRE_Crisis": "Banking stress, Fed forced to ease, risk-off",
                "If_Managed_Workout": "Gradual value discovery, banks survive"
            }
        },

        {
            "game_name": "MINER HEDGING GAME",
            "priority": "MEDIUM",
            "players": ["Silver/Gold Miners", "Bullion Banks", "Speculators", "Industrial Users"],
            "description": """
            Brandt's thesis hinges on miner hedging behavior at $100+ silver.
            Key questions:
            - Will miners hedge forward production?
            - How much hedging can market absorb?
            - Do speculators overwhelm supply?
            - What's the recycling supply elasticity?

            This determines the supply-side constraint for precious metals.
            """,
            "key_metrics": [
                "Miner hedge book disclosures",
                "COMEX commercial positioning",
                "Scrap supply data (Silver Institute)",
                "Producer cost curves"
            ],
            "implications": {
                "If_Miners_Hedge_Heavily": "Brandt thesis - supply caps price at $100-120",
                "If_Miners_Hold": "Oliver thesis - nothing stops price discovery"
            }
        },

        {
            "game_name": "AI COMPUTE ARMS RACE GAME",
            "priority": "MEDIUM",
            "players": ["US Tech Giants", "China Tech", "NVIDIA", "Hyperscalers", "Governments"],
            "description": """
            AI development requires massive compute investment.
            Key questions:
            - Does US maintain AI chip advantage?
            - Can China develop alternatives?
            - What's the energy constraint?
            - How does AI affect productivity/inflation?

            AI is the next industrial revolution - critical for strategic advantage.
            """,
            "key_metrics": [
                "NVIDIA data center revenue",
                "China AI chip development",
                "Data center power consumption",
                "AI productivity gains (measured)"
            ],
            "implications": {
                "If_US_AI_Dominance": "Tech leadership, productivity boost, dollar strength",
                "If_China_Catches_Up": "Tech bifurcation complete, strategic competition intensifies"
            }
        },

        # DOMESTIC US GAMES
        {
            "game_name": "FISCAL TRAJECTORY GAME",
            "priority": "CRITICAL",
            "players": ["Congress", "Trump Admin", "Treasury", "Fed", "Bond Vigilantes"],
            "description": """
            US running 6%+ deficit at full employment.
            Key questions:
            - Do tax cuts extend/expand?
            - Is there any spending restraint?
            - When do markets enforce discipline?
            - Can growth outpace debt accumulation?

            This is the ultimate constraint on all other games.
            """,
            "key_metrics": [
                "CBO deficit projections",
                "Debt/GDP trajectory",
                "Interest cost as % of revenue",
                "Term premium on long bonds"
            ],
            "implications": {
                "If_Fiscal_Consolidation": "Dollar strengthens, Warsh has room, Brandt thesis",
                "If_Fiscal_Explosion": "Bond crisis, Fed monetization, Oliver thesis"
            }
        },

        {
            "game_name": "LABOR MARKET DYNAMICS GAME",
            "priority": "MEDIUM",
            "players": ["Workers", "Employers", "Unions", "Immigration Policy", "Fed"],
            "description": """
            Labor market determines inflation stickiness.
            Key questions:
            - Does immigration restriction tighten labor?
            - Will AI replace workers (deflationary)?
            - How sticky are wage gains?
            - What's the NAIRU?

            Labor market determines whether Warsh can achieve soft landing.
            """,
            "key_metrics": [
                "Wage growth (AHE, ECI)",
                "Job openings / unemployed ratio",
                "Immigration flows",
                "AI displacement indicators"
            ],
            "implications": {
                "If_Labor_Loosens": "Inflation falls, soft landing, risk-on",
                "If_Labor_Stays_Tight": "Inflation sticky, Warsh must tighten more, risk-off"
            }
        }
    ]

    return games


# =============================================================================
# CROSS-GAME DEPENDENCY ANALYSIS
# =============================================================================

def analyze_cross_game_dependencies() -> Dict[str, Any]:
    """Analyze how games interact and create dependencies."""

    print("\n" + "=" * 100)
    print("CROSS-GAME DEPENDENCY MATRIX")
    print("=" * 100)

    dependencies = {
        "Warsh_Fed_Policy": {
            "depends_on": ["Fiscal_Trajectory", "US_China_Dynamics", "Labor_Market"],
            "influences": ["Treasury_Auctions", "Dollar_Strength", "Credit_Conditions", "Precious_Metals"],
            "critical_path": "Fed policy is CENTRAL node - affects everything"
        },
        "US_China_Dynamics": {
            "depends_on": ["Taiwan_Situation", "Tech_Competition", "Warsh_Policy"],
            "influences": ["Supply_Chains", "De_Dollarization", "Commodity_Demand", "Gold_Accumulation"],
            "critical_path": "US-China is the defining geopolitical game"
        },
        "Precious_Metals": {
            "depends_on": ["Dollar_Strength", "De_Dollarization", "CB_Gold_Buying", "Industrial_Demand"],
            "influences": ["Miner_Behavior", "Retail_Investment", "CB_Reserves"],
            "critical_path": "Precious metals are OUTPUT indicator of monetary regime"
        },
        "Treasury_Auctions": {
            "depends_on": ["Foreign_CB_Behavior", "Fiscal_Trajectory", "Fed_Policy"],
            "influences": ["Interest_Rates", "Dollar_Credibility", "Risk_Assets"],
            "critical_path": "Treasury market is CONSTRAINT on all other games"
        },
        "De_Dollarization": {
            "depends_on": ["US_China_Decoupling", "BRICS_Coordination", "US_Sanctions_Policy"],
            "influences": ["Gold_Demand", "Dollar_Value", "Reserve_Allocations"],
            "critical_path": "De-dollarization is LONG-TERM structural shift"
        }
    }

    print("""
CRITICAL PATH ANALYSIS:
=======================

The games form a HIERARCHY of dependencies:

Level 1 (Exogenous/Structural):
├── Geopolitical Shocks (Taiwan, Middle East)
├── Demographics/Labor
└── Technology Trajectory (AI, Chips)

Level 2 (Policy Choices):
├── US Fiscal Trajectory ← Determines everything
├── Fed Policy (Warsh) ← Constrained by fiscal
├── US-China Strategy ← Shapes global order
└── BRICS Coordination ← Alternative system

Level 3 (Market Outcomes):
├── Treasury Auctions ← Tests fiscal sustainability
├── Dollar Value ← Reveals system health
├── Credit Conditions ← Real economy transmission
└── Equity Markets ← Risk sentiment

Level 4 (Asset Prices - Observable Outcomes):
├── Precious Metals ← Monetary regime indicator
├── Commodities ← Real economy + geopolitics
├── Currencies ← Relative strength
└── Crypto ← Speculative/alternative

KEY INSIGHT:
============
To understand precious metals (Level 4), you must understand:
- Fed policy (Level 2)
- Which depends on fiscal trajectory (Level 2)
- Which interacts with US-China dynamics (Level 2)
- Which could be shocked by geopolitics (Level 1)

The Brandt vs Oliver debate is really about which LEVEL 2 games dominate:
- Brandt: Fed credibility + fiscal stability → Dollar strong → Supply dominates PM prices
- Oliver: Fiscal unsustainability + de-dollarization → Dollar weak → Monetary demand dominates

""")

    return dependencies


# =============================================================================
# SCENARIO PROBABILITY TREE
# =============================================================================

def build_probability_tree():
    """Build conditional probability tree across games."""

    print("\n" + "=" * 100)
    print("CONDITIONAL PROBABILITY TREE")
    print("=" * 100)

    print("""
DECISION TREE: PATH TO EACH SCENARIO
=====================================

                            WARSH FED OUTCOME
                           /                  \\
                  Credibility              Credibility
                  MAINTAINED               DAMAGED
                  (P=0.70)                 (P=0.30)
                    /    \\                   /    \\
                   /      \\                 /      \\
        US-China    US-China         US-China    US-China
        STABLE      ESCALATE         STABLE      ESCALATE
        (P=0.60)    (P=0.40)        (P=0.40)    (P=0.60)
           |           |               |           |
           |           |               |           |
    FISCAL      FISCAL          FISCAL      FISCAL
    STABLE      BLOWOUT         CRISIS      CRISIS
    (P=0.50)    (P=0.50)        (P=0.30)    (P=0.70)
       |           |               |           |
       ↓           ↓               ↓           ↓
    PAX         STAGFLATION    MANAGED     MONETARY
    AMERICANA   GRIND          MULTIPOLAR  CRISIS
    (21%)       (14.7%)        (6.3%)      (12.6%)

    [Plus GEOPOLITICAL SHOCK overlay: 15% independent probability]


RESULTING PROBABILITY DISTRIBUTION:
===================================

Scenario                      Probability    Key Condition
-----------------------------------------------------------------
PAX AMERICANA RENEWED          ~20%          Warsh succeeds + Stable geopolitics
MANAGED MULTIPOLARITY          ~30%          Gradual transition, no crisis
STAGFLATIONARY GRIND          ~20%          US-China tension + sticky inflation
MONETARY REGIME CRISIS         ~15%          Fed fails + fiscal crisis
GEOPOLITICAL SHOCK            ~15%          Exogenous (Taiwan, etc.)


PRECIOUS METALS PROBABILITY-WEIGHTED EXPECTATION:
=================================================

E[Silver Price] = Σ P(scenario) × E[Silver | scenario]

= 0.20 × $35 (Pax Americana)
+ 0.30 × $125 (Managed Multipolar)
+ 0.20 × $100 (Stagflation)
+ 0.15 × $350 (Monetary Crisis)
+ 0.15 × $150 (Geopolitical Shock)

= $7 + $37.5 + $20 + $52.5 + $22.5
= $139.50 probability-weighted expected silver price

This is why even with only 15% crisis probability,
the EXPECTED VALUE is well above current prices.
The distribution is RIGHT-SKEWED (fat right tail).

GOLD EXPECTED VALUE:
E[Gold Price] = 0.20 × $2,200 + 0.30 × $4,000 + 0.20 × $3,500 + 0.15 × $9,000 + 0.15 × $5,000
= $440 + $1,200 + $700 + $1,350 + $750
= $4,440 probability-weighted expected gold price

""")


# =============================================================================
# INVESTMENT IMPLICATIONS
# =============================================================================

def print_investment_framework():
    """Print integrated investment framework."""

    print("\n" + "=" * 100)
    print("INTEGRATED INVESTMENT FRAMEWORK")
    print("=" * 100)

    print("""
PORTFOLIO CONSTRUCTION ACROSS SCENARIOS:
========================================

Given the multi-scenario world with fat-tailed outcomes, optimal portfolio is:

CORE POSITIONS (60% of portfolio):
----------------------------------
These work across MOST scenarios:

1. SHORT DURATION FIXED INCOME (15%)
   - Money markets, T-bills, 1-2Y Treasuries
   - Works in: Pax Americana (carry), Stagflation (no duration hit), Crisis (liquidity)
   - Fails in: Monetary Crisis (inflation erodes)

2. QUALITY EQUITIES (20%)
   - High FCF, low debt, pricing power
   - Works in: Pax Americana, Managed Multipolar, Stagflation (relative)
   - Fails in: Crisis, Geopolitical Shock (but recovers)

3. PHYSICAL GOLD (15%)
   - Core monetary hedge
   - Works in: Managed Multipolar, Stagflation, Crisis, Geopolitical
   - Fails in: Pax Americana (opportunity cost only)

4. USD CASH (10%)
   - Liquidity and optionality
   - Works in: Pax Americana, Geopolitical Shock (initially)
   - Fails in: Monetary Crisis (but have other hedges)


SCENARIO-SPECIFIC HEDGES (30% of portfolio):
--------------------------------------------
Asymmetric payoffs for specific scenarios:

5. SILVER / SILVER MINERS (10%)
   - Leveraged precious metals exposure
   - Works in: Crisis (+300-500%), Managed Multipolar (+100-200%)
   - Fails in: Pax Americana (-30-50%), but size limits damage

6. LONG VOLATILITY (5%)
   - VIX calls, put spreads on indices
   - Works in: Crisis, Geopolitical Shock
   - Fails in: Calm scenarios (expected cost)

7. COMMODITY PRODUCERS (5%)
   - Energy, copper, uranium - supply constrained
   - Works in: Stagflation, Managed Multipolar, Geopolitical
   - Fails in: Demand destruction scenarios

8. EM EQUITIES - SELECTIVE (5%)
   - India, Vietnam, Mexico - supply chain beneficiaries
   - Works in: Managed Multipolar
   - Fails in: Crisis (correlations spike)

9. TIPS / INFLATION HEDGES (5%)
   - Real rate protection
   - Works in: Stagflation, Crisis
   - Fails in: Pax Americana (deflation risk?)


DYNAMIC ALLOCATION TRIGGERS:
----------------------------

INCREASE RISK (move toward Pax Americana allocation):
- Warsh achieves credibility (inflation < 3% sustained)
- US-China relations stabilize
- Treasury auctions well-bid
- Dollar strengthens above 110 DXY

DECREASE RISK (move toward Crisis allocation):
- Treasury auction problems (tail bids, failed auctions)
- Foreign CB selling Treasuries
- BRICS currency announcement
- Taiwan tensions escalate
- Inflation expectations unanchor (5Y5Y > 3%)

INCREASE PRECIOUS METALS:
- Gold breaks above $3,000 with volume
- Gold/Silver ratio falls below 70
- Central bank buying accelerates
- Dollar breaks below 95 DXY


EXPECTED RETURNS BY SCENARIO:
=============================

                      Pax     Managed   Stagflation  Crisis  Geopolitical
Portfolio Return     +15%      +8%        -5%         +25%      +5%
Probability           20%      30%        20%         15%       15%

Expected Portfolio Return = 0.20(15) + 0.30(8) + 0.20(-5) + 0.15(25) + 0.15(5)
                         = 3 + 2.4 - 1 + 3.75 + 0.75
                         = +8.9% expected return

RISK MANAGEMENT:
================

Maximum Drawdown by Scenario:
- Pax Americana: -5% (opportunity cost on hedges)
- Managed Multipolar: -10% (volatility during transition)
- Stagflation: -20% (equities down, hedges lag)
- Monetary Crisis: -30% then +50% (V-shape, hedges kick in)
- Geopolitical Shock: -25% then recovery (depends on resolution)

Kelly Criterion Position Sizing:
- Given asymmetric payoffs, Kelly suggests OVERWEIGHT tail hedges
- Expected value of precious metals > current prices by ~50%
- Size PM position at 15-20% despite 15% crisis probability
- Because: (0.15 × 300%) + (0.30 × 100%) + (0.20 × 50%) >> (0.20 × -40%)
""")


# =============================================================================
# MONITORING DASHBOARD
# =============================================================================

def print_monitoring_dashboard():
    """Print key metrics to monitor across all games."""

    print("\n" + "=" * 100)
    print("STRATEGIC MONITORING DASHBOARD")
    print("=" * 100)

    print("""
TIER 1: CRITICAL INDICATORS (Check Daily/Weekly)
=================================================

FED / MONETARY:
□ Fed Funds Futures (terminal rate pricing)
□ 2Y Treasury Yield (policy expectations)
□ 10Y Treasury Yield (growth/inflation expectations)
□ 2s10s Spread (recession signal)
□ MOVE Index (bond volatility)
□ Fed Balance Sheet (QT pace)

DOLLAR / FX:
□ DXY Index (dollar strength)
□ EUR/USD, USD/JPY (major pairs)
□ USD/CNY (China pressure)
□ EM FX Index (risk sentiment)

PRECIOUS METALS:
□ Gold Spot Price
□ Silver Spot Price
□ Gold/Silver Ratio
□ COMEX Inventory (registered/eligible)
□ GLD/SLV ETF Flows

CREDIT / RISK:
□ HY OAS Spread (credit risk)
□ VIX Index (equity volatility)
□ Investment Grade Spreads
□ Bank CDS Spreads


TIER 2: WEEKLY/MONTHLY STRUCTURAL INDICATORS
============================================

TREASURY MARKET:
□ Treasury Auction Results (bid-to-cover, tail)
□ TIC Data (foreign holdings)
□ Primary Dealer Positioning
□ Fed RRP Usage (liquidity)

US-CHINA:
□ Semiconductor Equipment Sales to China
□ Bilateral Trade Data
□ Rare Earth Prices
□ TSMC Revenue by Geography

CENTRAL BANK GOLD:
□ WGC Monthly CB Purchases
□ China Official Gold Holdings (reported)
□ Russia Gold Reserves
□ BIS Gold Statistics

FISCAL:
□ Treasury Monthly Statement (deficit)
□ CBO Updates
□ Debt Ceiling Status
□ Interest Cost Trajectory


TIER 3: QUALITATIVE / EVENT MONITORING
======================================

GEOPOLITICAL:
□ Taiwan Strait Activity (ship/aircraft transits)
□ BRICS Summit Outcomes
□ Saudi-China Relationship
□ Iran Nuclear Status
□ Ukraine War Trajectory

POLICY:
□ Warsh Speeches / FOMC Communications
□ Treasury Issuance Announcements
□ Trade Policy Announcements
□ Sanctions Updates

MARKET STRUCTURE:
□ COMEX Delivery Activity
□ LBMA Vault Holdings
□ Miner Hedging Announcements
□ ETF Creation/Redemption Patterns


TRIGGER ALERT THRESHOLDS:
=========================

BULLISH PRECIOUS METALS:
⚠️ DXY < 95 sustained
⚠️ 10Y yield > 5.5% (fiscal stress)
⚠️ Gold > $3,000 breakout
⚠️ Gold/Silver ratio < 70
⚠️ COMEX registered < 25M oz (silver)
⚠️ CB gold buying > 300t/quarter

BEARISH PRECIOUS METALS:
⚠️ DXY > 110 sustained
⚠️ Real rates > 2.5% (10Y TIPS)
⚠️ Miner hedge announcements surge
⚠️ COMEX inventory builds
⚠️ Gold/Silver ratio > 90

RISK-OFF ALERT:
⚠️ VIX > 30 sustained
⚠️ HY spreads > 500bp
⚠️ Bank CDS > 100bp
⚠️ Treasury auction tail > 3bp
⚠️ MOVE > 150
""")


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def print_executive_summary():
    """Print the grand unified executive summary."""

    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY: THE GRAND UNIFIED STRATEGIC PICTURE")
    print("=" * 100)

    print("""
THE INTEGRATED VIEW:
====================

We are witnessing a PHASE TRANSITION in the global monetary and geopolitical order.
The key games are deeply interconnected:

1. WARSH FED APPOINTMENT = Attempt to maintain dollar credibility
   - Trump learned from Biden: inflation is politically fatal
   - Hawkish Fed is NECESSARY for fiscal sustainability
   - But fiscal trajectory may OVERWHELM monetary policy

2. US-CHINA COMPETITION = Acceleration of multipolar shift
   - Tech decoupling is structural, not cyclical
   - Financial decoupling drives de-dollarization
   - No resolution likely - managed competition at best

3. PRECIOUS METALS = Thermometer of monetary regime health
   - Brandt thesis = Fed credibility holds, supply dominates
   - Oliver thesis = Fed credibility fails, monetary demand dominates
   - Current evidence: 65-70% probability favors Oliver direction

THE MASTER GAME: Dollar Hegemony vs Multipolar Order
====================================================

Everything flows from this meta-game:

IF DOLLAR HEGEMONY MAINTAINED:
- Warsh succeeds, inflation controlled
- US-China competition managed
- Treasury market stable
- Gold $2,000-3,000, Silver $30-50
- Risk assets perform, USD strong

IF MULTIPOLAR TRANSITION:
- Fed constrained by fiscal dominance
- De-dollarization accelerates
- Treasury market stressed
- Gold $5,000-10,000, Silver $150-500
- Risk assets volatile, USD weakens

PROBABILITY ASSESSMENT:
- Dollar hegemony maintained: 30-35%
- Gradual multipolar transition: 40-45%
- Disorderly transition/crisis: 20-25%

INVESTMENT IMPLICATION:
======================

The EXPECTED VALUE calculation strongly favors precious metals exposure
even though crisis probability is "only" 20-25%. Why?

Because:
- Upside in crisis: +300-500%
- Upside in gradual transition: +100-200%
- Downside in hegemony maintained: -30-40%

The distribution is ASYMMETRIC. The expected value of precious metals
is ABOVE current prices in most models.

FURTHER GAMES TO INVESTIGATE:
=============================

CRITICAL (investigate immediately):
1. Treasury Auction Game - tests fiscal sustainability
2. BRICS Currency Game - alternative system viability
3. Taiwan Semiconductor Game - geopolitical flashpoint
4. Fiscal Trajectory Game - master constraint

HIGH PRIORITY:
5. Central Bank Gold Accumulation - revealed preferences
6. Corporate Debt Refinancing - credit cycle
7. Commercial Real Estate - hidden risk
8. Miner Hedging Game - supply response

MONITOR:
9. Middle East Energy Game
10. European Defense/Energy Game
11. AI Compute Arms Race
12. Labor Market Dynamics

THE BOTTOM LINE:
================

We are in a multi-game strategic environment where:

• The FED is trying to maintain credibility (Warsh appointment)
• While FISCAL trajectory threatens to overwhelm
• And US-CHINA competition accelerates structural change
• Leading to PRECIOUS METALS as key regime indicator

The Brandt vs Oliver debate is really a debate about which GAME dominates:
• Brandt: Fed credibility game dominates → supply/demand analysis works
• Oliver: Fiscal/geopolitical game dominates → monetary demand overwhelms supply

Current evidence suggests we are in TRANSITION toward Oliver's world,
but the timing and path remain uncertain.

RECOMMENDED STRATEGY:
1. Core portfolio: Balanced, quality-focused
2. Precious metals: 15-20% allocation (asymmetric payoff)
3. Tail hedges: Volatility exposure for crisis scenario
4. Monitor: Treasury auctions, CB gold buying, DXY trajectory
5. Be prepared: To increase PM allocation if fiscal stress emerges

"The market can remain irrational longer than you can remain solvent"
— but also —
"In the long run, the market is a weighing machine"

Position for the WEIGHING MACHINE while surviving the VOTING MACHINE.
""")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_grand_unified_analysis():
    """Run the complete grand unified analysis."""

    print_grand_strategic_overview()

    # Master causal graph analysis
    print("\n" + "=" * 100)
    print("MASTER CAUSAL GRAPH ANALYSIS")
    print("=" * 100)

    graph = MasterCausalGraph()

    # Key shock propagations
    print("\nSHOCK: Warsh Implements Aggressive Tightening")
    print("-" * 60)
    effects = graph.propagate_shock("Warsh_Fed_Policy", "UP", max_depth=4)
    print(f"{'Factor':<35} {'Direction':<10} {'Magnitude'}")
    print("-" * 60)
    for e in effects[:15]:
        print(f"{e['factor']:<35} {e['direction']:<10} {e['magnitude']:.4f}")

    print("\nSHOCK: US-China Financial Decoupling Accelerates")
    print("-" * 60)
    effects = graph.propagate_shock("US_China_Financial_Decoupling", "UP", max_depth=4)
    print(f"{'Factor':<35} {'Direction':<10} {'Magnitude'}")
    print("-" * 60)
    for e in effects[:15]:
        print(f"{e['factor']:<35} {e['direction']:<10} {e['magnitude']:.4f}")

    print("\nSHOCK: Taiwan Tensions Escalate")
    print("-" * 60)
    effects = graph.propagate_shock("Taiwan_Tensions", "UP", max_depth=4)
    print(f"{'Factor':<35} {'Direction':<10} {'Magnitude'}")
    print("-" * 60)
    for e in effects[:15]:
        print(f"{e['factor']:<35} {e['direction']:<10} {e['magnitude']:.4f}")

    # Feedback loops
    print("\n" + "=" * 100)
    print("CRITICAL FEEDBACK LOOPS")
    print("=" * 100)

    loops = graph.find_feedback_loops()
    for loop in loops:
        print(f"\n{loop['name']} ({loop['type']})")
        print(f"  Path: {' → '.join(loop['path'])}")
        print(f"  Description: {loop['description']}")

    # Integrated scenarios
    print("\n" + "=" * 100)
    print("INTEGRATED SCENARIOS")
    print("=" * 100)

    scenarios = build_integrated_scenarios()
    for s in scenarios:
        print(f"\n{s.name} (P = {s.probability*100:.0f}%)")
        print(f"  {s.description}")
        print(f"  Fed: {s.fed_policy}")
        print(f"  US-China: {s.us_china_dynamic}")
        print(f"  Monetary: {s.monetary_regime}")
        print(f"  Precious Metals: {s.precious_metals_outlook}")

    # Additional games
    print("\n" + "=" * 100)
    print("FURTHER GAMES TO INVESTIGATE")
    print("=" * 100)

    games = identify_further_games()
    for game in games:
        print(f"\n{game['priority']}: {game['game_name']}")
        print(f"  Players: {', '.join(game['players'][:5])}")
        print(f"  Key Metrics: {', '.join(game['key_metrics'][:3])}")

    # Cross-game dependencies
    analyze_cross_game_dependencies()

    # Probability tree
    build_probability_tree()

    # Investment framework
    print_investment_framework()

    # Monitoring dashboard
    print_monitoring_dashboard()

    # Executive summary
    print_executive_summary()

    return {
        "scenarios": scenarios,
        "games_to_investigate": games,
        "causal_graph": graph,
        "key_insight": "Dollar hegemony vs multipolar transition is the master game"
    }


if __name__ == "__main__":
    result = run_grand_unified_analysis()
