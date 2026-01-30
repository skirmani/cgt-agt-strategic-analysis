#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GAME THEORY ANALYSIS: WHY TRUMP APPOINTED WARSH
================================================================================

The Paradox: Trump criticized Powell for being too hawkish, yet appointed
Kevin Warsh - an even MORE hawkish figure. This analysis explores the
multi-dimensional strategic game behind this seemingly contradictory decision.

Key Question: What is Trump's actual utility function, and how does appointing
a hawk maximize it?

Framework:
1. Multi-Principal Game (Trump, Congress, Markets, Fed, Foreign Actors)
2. Signaling Game (Credibility, Commitment)
3. Time-Inconsistency Problem (Short vs Long-term)
4. Mechanism Design (Institutional Constraints)

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import math

# =============================================================================
# THE STRATEGIC PUZZLE
# =============================================================================

def print_the_paradox():
    """Explain the apparent contradiction."""

    print("=" * 80)
    print("THE WARSH PARADOX: WHY WOULD TRUMP APPOINT A HAWK?")
    print("=" * 80)

    print("""
THE SURFACE-LEVEL CONTRADICTION:
================================

TRUMP'S STATED PREFERENCES (2018-2024):
• Criticized Powell repeatedly for raising rates
• Called Fed "crazy" and "loco" for tightening
• Wanted lower rates to boost economy/markets
• Preferred weak dollar for trade competitiveness
• Publicly pressured Fed for easier policy

KEVIN WARSH'S KNOWN POSITIONS:
• Monetary policy HAWK
• Critical of QE and balance sheet expansion
• Advocates rules-based policy (Taylor Rule)
• Favors faster rate normalization
• Less market accommodation than Powell
• Skeptical of forward guidance

THE PUZZLE:
If Trump wants easy money, why appoint someone MORE hawkish than Powell?

This analysis explores the GAME-THEORETIC reasons why this appointment
may actually be RATIONAL given Trump's true utility function.
""")


# =============================================================================
# TRUMP'S TRUE UTILITY FUNCTION
# =============================================================================

@dataclass
class TrumpUtilityFunction:
    """
    Model Trump's actual multi-dimensional utility function.

    Key insight: Trump's PUBLIC statements about wanting low rates
    may not reflect his ACTUAL strategic objectives.
    """

    # Weights on different objectives (sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "inflation_control": 0.25,      # Voters HATE inflation
        "economic_growth": 0.20,        # GDP growth matters
        "market_performance": 0.15,     # Stock market as scorecard
        "employment": 0.10,             # Unemployment politically toxic
        "dollar_credibility": 0.10,     # Reserve currency status
        "fed_loyalty": 0.10,            # Wants Fed that listens
        "legacy_institutional": 0.05,   # Historical standing
        "trade_competitiveness": 0.05   # Weak dollar for exports
    })

    def compute_utility(self, scenario: Dict[str, float]) -> float:
        """Compute weighted utility across objectives."""
        return sum(
            self.weights[obj] * scenario.get(obj, 0)
            for obj in self.weights
        )


def analyze_trump_utility():
    """Analyze Trump's true strategic objectives."""

    print("\n" + "=" * 80)
    print("1. TRUMP'S TRUE UTILITY FUNCTION")
    print("=" * 80)

    print("""
REVEALED PREFERENCE ANALYSIS:
=============================

What Trump SAYS he wants:
• Low interest rates
• Weak dollar
• Accommodative Fed

What Trump ACTUALLY needs (for political success):
• LOW INFLATION - Voters punish incumbents for inflation
• STRONG ECONOMY - GDP growth, jobs
• CREDIBILITY - Institutional respect, global standing
• LOYAL ALLIES - People who will coordinate with his agenda

KEY INSIGHT: Trump's criticism of Powell was about CONTROL, not DOVISHNESS.

Trump didn't want Powell to be more dovish per se - he wanted Powell to
COORDINATE with his economic agenda. Powell's "sin" was INDEPENDENCE,
not hawkishness.

TRUMP'S REAL UTILITY FUNCTION:
==============================
""")

    utility = TrumpUtilityFunction()

    print("Objective                  Weight    Rationale")
    print("-" * 70)

    rationales = {
        "inflation_control": "Voters punish inflation; 2022 midterms lesson",
        "economic_growth": "GDP growth = political success metric",
        "market_performance": "Stock market as visible scorecard",
        "employment": "Unemployment politically toxic",
        "dollar_credibility": "Preserve reserve currency, attract capital",
        "fed_loyalty": "Coordination with fiscal policy agenda",
        "legacy_institutional": "Historical standing, 'great president'",
        "trade_competitiveness": "Manufacturing jobs, trade deals"
    }

    for obj, weight in sorted(utility.weights.items(), key=lambda x: -x[1]):
        print(f"{obj:25s}  {weight:5.0%}     {rationales[obj]}")

    print("""
CRITICAL REALIZATION:
=====================
Inflation control (25%) + Dollar credibility (10%) = 35% of utility
is served by HAWKISH Fed policy!

Trump learned from Biden's political damage from inflation.
A hawk who CONTROLS inflation may serve Trump better than a dove
who lets inflation run.
""")

    return utility


# =============================================================================
# SIGNALING GAME ANALYSIS
# =============================================================================

def analyze_signaling_game():
    """Analyze Warsh appointment as a signaling game."""

    print("\n" + "=" * 80)
    print("2. SIGNALING GAME: CREDIBILITY & COMMITMENT")
    print("=" * 80)

    print("""
SIGNALING THEORY FRAMEWORK:
===========================

In game theory, COSTLY SIGNALS are credible because they're expensive
to fake. Appointing a known hawk sends powerful signals to multiple audiences.

SIGNAL #1: TO BOND MARKETS / FOREIGN CREDITORS
----------------------------------------------
Message: "We are serious about inflation and fiscal responsibility"

Why it matters:
• US has $35T+ debt, needs continued foreign financing
• Treasury issuance massive under any scenario
• If markets doubt US inflation commitment → higher yields → fiscal crisis
• A hawk at Fed = lower risk premium on Treasuries

Payoff Matrix (US Government as borrower):
                        Markets Believe    Markets Doubt
Appoint Hawk            Lower yields       Still credible
Appoint Dove            Higher yields      CRISIS (yields spike)

Appointing Warsh is DOMINANT STRATEGY for debt sustainability.


SIGNAL #2: TO DOMESTIC VOTERS
-----------------------------
Message: "I learned from Biden's inflation mistake"

Why it matters:
• 2022 midterms showed inflation is politically toxic
• Trump needs to differentiate from "Bidenomics"
• Appointing hawk = "I'm the inflation fighter"
• Pre-emptive defense against any future inflation

Political calculus:
• If inflation stays low → "My Fed chair kept it down"
• If inflation rises → "See, I tried, blame Congress/external factors"


SIGNAL #3: TO GLOBAL ACTORS (China, BRICS, etc.)
------------------------------------------------
Message: "Dollar hegemony will be defended"

Why it matters:
• De-dollarization concerns are real
• China, Russia, BRICS exploring alternatives
• A dovish Fed = accelerated de-dollarization
• Hawkish Fed = "We will maintain dollar's value"

This COUNTERS the narrative that Trump wants to debase the dollar.


SIGNAL #4: TO THE FED ITSELF (Institutional)
--------------------------------------------
Message: "I want competence and coordination, not just loyalty"

Why it matters:
• Powell was seen as too independent
• But Judy Shelton types seen as unqualified
• Warsh = respected, credible, but ALIGNED with Trump world
• Signal that Trump administration is "serious"
""")

    # Signaling game payoff structure
    signals = {
        "Appoint_Hawk_Warsh": {
            "bond_market_reaction": 0.7,
            "voter_perception": 0.6,
            "global_credibility": 0.8,
            "institutional_respect": 0.7,
            "actual_policy_control": 0.5,  # Less direct control
            "aggregate_signal_value": 0.65
        },
        "Appoint_Dove_Loyalist": {
            "bond_market_reaction": -0.5,
            "voter_perception": -0.3,
            "global_credibility": -0.6,
            "institutional_respect": -0.7,
            "actual_policy_control": 0.8,  # More direct control
            "aggregate_signal_value": -0.25
        },
        "Reappoint_Powell": {
            "bond_market_reaction": 0.3,
            "voter_perception": 0.0,
            "global_credibility": 0.4,
            "institutional_respect": 0.5,
            "actual_policy_control": 0.2,
            "aggregate_signal_value": 0.30
        }
    }

    print("\nSIGNALING PAYOFF MATRIX:")
    print("-" * 70)
    print(f"{'Choice':<25} {'Markets':<10} {'Voters':<10} {'Global':<10} {'Instit.':<10} {'Control':<10}")
    print("-" * 70)

    for choice, payoffs in signals.items():
        print(f"{choice:<25} {payoffs['bond_market_reaction']:+.1f}       "
              f"{payoffs['voter_perception']:+.1f}       "
              f"{payoffs['global_credibility']:+.1f}       "
              f"{payoffs['institutional_respect']:+.1f}       "
              f"{payoffs['actual_policy_control']:+.1f}")

    print("""
CONCLUSION: Warsh appointment DOMINATES in signaling value.
The only dimension where a dove wins is "direct control" - but that
control is useless if markets and voters punish you for using it.
""")

    return signals


# =============================================================================
# TIME INCONSISTENCY PROBLEM
# =============================================================================

def analyze_time_inconsistency():
    """Analyze the time inconsistency problem in monetary policy."""

    print("\n" + "=" * 80)
    print("3. TIME INCONSISTENCY: THE CREDIBLE COMMITMENT PROBLEM")
    print("=" * 80)

    print("""
THE CLASSIC TIME INCONSISTENCY PROBLEM:
=======================================

Kydland-Prescott (1977) / Barro-Gordon (1983):

Politicians have an INCENTIVE to promise low inflation, but then
INFLATE once expectations are set. Rational agents anticipate this,
leading to WORSE outcomes for everyone.

The Solution: DELEGATE to a conservative central banker who has
stronger anti-inflation preferences than the politician.

                        Short-Run         Long-Run
Dove Fed Chair          Stimulus boost    Inflation, credibility loss
Hawk Fed Chair          Less stimulus     Low inflation, credibility

TRUMP'S STRATEGIC INSIGHT:
==========================

By appointing Warsh (a hawk), Trump:

1. COMMITS CREDIBLY to low inflation
   - Can't easily reverse (4-year term)
   - Warsh's reputation precedes him
   - Markets believe the commitment

2. SOLVES HIS OWN TIME INCONSISTENCY
   - Trump might be tempted to pressure for easy money
   - But Warsh won't comply → Trump protected from himself
   - "Tying himself to the mast" (Odysseus strategy)

3. CREATES POLITICAL COVER
   - If economy struggles: "Fed is independent, not my fault"
   - If inflation rises: "I appointed a hawk, blame external factors"
   - If things go well: "My excellent Fed chair"

4. LEARNS FROM ERDOGAN DISASTER
   - Turkey: President controls central bank, fires governors
   - Result: Currency collapse, hyperinflation
   - Trump sees this as cautionary tale


THE ROGOFF "CONSERVATIVE CENTRAL BANKER" MODEL:
===============================================

Optimal delegation: Appoint someone MORE hawkish than socially optimal,
because this overcomes the inflation bias from time inconsistency.

Society's loss function: L = (π - π*)² + λ(y - y*)²
Central banker's loss: L = (π - π*)² + λ_CB(y - y*)²

Where λ_CB < λ (central banker cares less about output than society)

Trump appointing Warsh = TEXTBOOK Rogoff solution!

Warsh's hawkishness is a FEATURE, not a bug. It provides the credibility
that a dove could never achieve.
""")

    # Time inconsistency game
    print("\nTIME INCONSISTENCY PAYOFF ANALYSIS:")
    print("-" * 60)

    scenarios = {
        "Dove_Fed_Expectations_Low": {
            "description": "Appoint dove, markets expect low inflation (IMPOSSIBLE)",
            "probability": 0.0,
            "inflation": "N/A - markets won't believe it"
        },
        "Dove_Fed_Expectations_High": {
            "description": "Appoint dove, markets price in inflation",
            "probability": 0.85,
            "inflation": "High (self-fulfilling)",
            "yields": "Higher (inflation premium)",
            "growth": "Short boost, then stagflation",
            "political_outcome": "Blamed for inflation"
        },
        "Hawk_Fed_Credible": {
            "description": "Appoint hawk, markets believe commitment",
            "probability": 0.90,
            "inflation": "Low (expectations anchored)",
            "yields": "Lower (credibility premium)",
            "growth": "Moderate but sustainable",
            "political_outcome": "Credit for stability"
        }
    }

    for name, data in scenarios.items():
        print(f"\n{name}:")
        print(f"  Probability market believes: {data.get('probability', 'N/A')}")
        print(f"  Inflation outcome: {data.get('inflation', 'N/A')}")
        if 'yields' in data:
            print(f"  Treasury yields: {data['yields']}")
        if 'political_outcome' in data:
            print(f"  Political outcome: {data['political_outcome']}")

    print("""
GAME THEORY CONCLUSION:
=======================
Appointing a hawk is the SUBGAME PERFECT EQUILIBRIUM.

The "desire" for easy money is a dominated strategy when you account
for rational expectations. Trump (or his advisors) understand this.
""")


# =============================================================================
# COORDINATION GAME WITH TREASURY
# =============================================================================

def analyze_treasury_coordination():
    """Analyze Fed-Treasury coordination game."""

    print("\n" + "=" * 80)
    print("4. FED-TREASURY COORDINATION GAME")
    print("=" * 80)

    print("""
THE FISCAL DOMINANCE PROBLEM:
=============================

With $35T+ debt and large deficits, there's a risk of "fiscal dominance"
where Fed policy is dictated by Treasury's financing needs.

Historical precedent:
• 1942-1951: Fed pegged rates to help Treasury (inflationary)
• 1951 Treasury-Fed Accord: Fed regained independence
• Post-2008: QE arguably served fiscal purposes

TRUMP'S DILEMMA:
================

Trump wants to:
1. Cut taxes (increase deficit)
2. Maintain spending (increase deficit)
3. Keep Treasury yields manageable (requires Fed cooperation)
4. Avoid inflation (requires Fed independence)

These objectives are in TENSION.

THE WARSH SOLUTION:
===================

Warsh represents a SOPHISTICATED coordination mechanism:

1. PUBLICLY INDEPENDENT
   - Warsh has hawkish credentials
   - Markets see him as inflation-fighter
   - This LOWERS yields (credibility premium)

2. PRIVATELY COORDINATED
   - Warsh is Trump-aligned politically
   - Goldman background = understands market plumbing
   - Will coordinate on Treasury issuance strategy
   - Can adjust QT pace to avoid market dysfunction

3. THE "GOOD COP / BAD COP" DYNAMIC
   - Warsh plays "bad cop" (hawkish rhetoric)
   - Treasury plays "good cop" (fiscal stimulus)
   - Net effect: Stimulus without losing credibility

This is NOT the same as appointing a puppet. A puppet would be
discredited and yields would spike. Warsh provides the COVER
for coordination without the STIGMA.
""")

    # Coordination game matrix
    print("\nFED-TREASURY COORDINATION MATRIX:")
    print("-" * 70)
    print("""
                            Treasury
                    Fiscal Restraint    Fiscal Expansion

Fed     Hawkish     (Credibility++,     (Credibility+,
                     Growth--)           Growth+)
                     "Austerity"         "OPTIMAL ZONE"

        Dovish      (Credibility-,      (Credibility--,
                     Growth+)            Growth++)
                     "Odd combo"         "Fiscal Dominance"
                                         → Eventually crisis

WARSH APPOINTMENT = Upper-right quadrant is achievable.

With a credible hawk, Treasury can be MORE expansionary without
triggering a bond market crisis. The hawk provides AIR COVER
for fiscal stimulus.
""")

    # Quantitative analysis
    coordination_outcomes = {
        "Hawk_Fed_Fiscal_Expansion": {
            "10Y_yield_impact": "+50bp",
            "credibility_impact": "Maintained",
            "growth_impact": "+1.5% GDP",
            "sustainability": "High",
            "political_viability": "High"
        },
        "Dove_Fed_Fiscal_Expansion": {
            "10Y_yield_impact": "+150-200bp",
            "credibility_impact": "Damaged",
            "growth_impact": "+2% then crash",
            "sustainability": "Low - crisis likely",
            "political_viability": "Low - blamed for inflation"
        },
        "Hawk_Fed_Fiscal_Restraint": {
            "10Y_yield_impact": "-50bp",
            "credibility_impact": "Strong",
            "growth_impact": "-0.5% GDP",
            "sustainability": "High",
            "political_viability": "Low - recession risk"
        }
    }

    print("\nOUTCOME ANALYSIS:")
    print("-" * 70)

    for outcome, data in coordination_outcomes.items():
        print(f"\n{outcome}:")
        for key, value in data.items():
            print(f"  {key}: {value}")


# =============================================================================
# FED INDEPENDENCE STRATEGIC ANALYSIS
# =============================================================================

def analyze_fed_independence():
    """Analyze the Fed independence dimension."""

    print("\n" + "=" * 80)
    print("5. FED INDEPENDENCE: THE STRATEGIC CALCULUS")
    print("=" * 80)

    print("""
THE FED INDEPENDENCE PARADOX:
=============================

Conventional view: Trump wants to REDUCE Fed independence
Actual strategy: Trump wants ALIGNED independence

There's a crucial distinction:

1. PUPPET FED (No independence)
   - Does whatever White House says
   - Markets discount this → yields spike
   - Inflation expectations unanchored
   - EVERYONE loses

2. HOSTILE INDEPENDENT FED (Powell model)
   - Acts contrary to administration
   - Creates policy conflict
   - Uncertainty hurts economy
   - Political friction

3. ALIGNED INDEPENDENT FED (Warsh model)
   - Has credibility/independence reputation
   - But SHARES administration's macro framework
   - Coordination without subordination
   - OPTIMAL OUTCOME


TRUMP'S POWELL CRITIQUE - REINTERPRETED:
========================================

Trump's attacks on Powell weren't really about dovishness vs hawkishness.
They were about ALIGNMENT and COORDINATION.

What Trump actually wanted:
• Fed that COMMUNICATED with Treasury
• Fed that considered FISCAL POLICY interaction
• Fed that understood POLITICAL constraints
• Fed that was PREDICTABLE for planning

Powell's "sin" was uncertainty and non-coordination, not hawkishness.

Warsh provides:
• PREDICTABILITY (rules-based, Taylor Rule)
• COORDINATION (aligned worldview)
• CREDIBILITY (hawk reputation)
• COMMUNICATION (former insider, understands system)


THE "CONSERVATIVE BUT LOYAL" EQUILIBRIUM:
=========================================

This is actually a well-known solution in political economy:

Appoint someone who is:
1. Conservative enough to be CREDIBLE
2. Loyal enough to be COORDINATED
3. Competent enough to be EFFECTIVE
4. Independent enough to be RESPECTED

Warsh checks all four boxes. A pure dove or pure loyalist fails #1 and #4.
A hostile independent fails #2. An incompetent fails #3.
""")

    # Independence spectrum analysis
    print("\nFED CHAIR INDEPENDENCE SPECTRUM:")
    print("-" * 70)

    spectrum = {
        "Judy_Shelton": {
            "independence": 0.2,
            "credibility": 0.3,
            "coordination": 0.9,
            "competence": 0.4,
            "market_reaction": "Negative - seen as political",
            "aggregate_score": 0.35
        },
        "Kevin_Warsh": {
            "independence": 0.7,
            "credibility": 0.8,
            "coordination": 0.7,
            "competence": 0.8,
            "market_reaction": "Positive - credible hawk",
            "aggregate_score": 0.75
        },
        "Jerome_Powell": {
            "independence": 0.8,
            "credibility": 0.7,
            "coordination": 0.3,
            "competence": 0.7,
            "market_reaction": "Neutral - known quantity",
            "aggregate_score": 0.60
        },
        "Random_Dove": {
            "independence": 0.4,
            "credibility": 0.3,
            "coordination": 0.8,
            "competence": 0.5,
            "market_reaction": "Negative - inflation fears",
            "aggregate_score": 0.40
        }
    }

    print(f"{'Candidate':<20} {'Indep.':<8} {'Credib.':<8} {'Coord.':<8} {'Compet.':<8} {'Score':<8}")
    print("-" * 70)

    for name, scores in sorted(spectrum.items(), key=lambda x: -x[1]['aggregate_score']):
        print(f"{name:<20} {scores['independence']:<8.1f} {scores['credibility']:<8.1f} "
              f"{scores['coordination']:<8.1f} {scores['competence']:<8.1f} {scores['aggregate_score']:<8.2f}")

    print("""
WARSH DOMINATES on the aggregate score that matters for Trump's objectives.
""")


# =============================================================================
# THE INFLATION POLITICAL ECONOMY
# =============================================================================

def analyze_inflation_politics():
    """Analyze the political economy of inflation."""

    print("\n" + "=" * 80)
    print("6. INFLATION POLITICAL ECONOMY: THE BIDEN LESSON")
    print("=" * 80)

    print("""
TRUMP LEARNED FROM BIDEN'S POLITICAL DISASTER:
==============================================

Biden's approval rating trajectory:
• Jan 2021: 55% (honeymoon)
• June 2021: 52% (still ok)
• Dec 2021: 43% (inflation hitting)
• June 2022: 39% (inflation peak)
• Never recovered above 45%

Inflation was THE defining issue that destroyed Biden's presidency.

KEY DATA POINTS:
• CPI peaked at 9.1% (June 2022)
• Gas prices peaked at $5+/gallon
• Grocery prices up 20%+ cumulative
• Voters blamed Biden DIRECTLY

TRUMP'S CALCULUS:
=================

Expected utility calculation:

Scenario A: Dovish Fed, risk inflation
• P(inflation > 4%) = 40%
• Political cost if inflation: CATASTROPHIC (-30 approval points)
• Expected political cost: 0.4 × (-30) = -12 points

Scenario B: Hawkish Fed, control inflation
• P(inflation > 4%) = 10%
• Political cost if inflation: Same (-30 points)
• Political cost of slower growth: Modest (-5 points)
• Expected political cost: 0.1 × (-30) + 0.9 × (-5) = -7.5 points

HAWKISH FED HAS HIGHER EXPECTED UTILITY!

The asymmetry is crucial:
• Voters PUNISH inflation severely
• Voters tolerate moderate growth
• Inflation is VISIBLE (gas, groceries)
• Growth is ABSTRACT (GDP numbers)


THE "BLAME ALLOCATION" GAME:
============================

With independent hawkish Fed:

If economy does well:
• Trump: "My policies created growth"
• Warsh: Gets some credit
• Result: WIN-WIN

If inflation rises:
• Trump: "I appointed inflation fighter, blame external factors"
• Warsh: Takes some heat but was doing his job
• Result: SHARED BLAME

If recession:
• Trump: "Fed was too tight, I wanted to help but..."
• Warsh: Takes primary blame
• Result: BLAME DEFLECTION

With puppet dovish Fed:

ANY bad outcome:
• Trump: Gets 100% of blame
• No deflection possible
• Result: FULL ACCOUNTABILITY


This is classic AGENCY THEORY:
Delegate to an agent with different preferences to achieve
better outcomes AND better political positioning.
""")

    # Political payoff matrix
    print("\nPOLITICAL PAYOFF MATRIX:")
    print("-" * 70)

    political_payoffs = {
        "Outcome": ["Strong growth + low inflation", "Moderate growth + low inflation",
                   "Strong growth + high inflation", "Recession + low inflation",
                   "Stagflation"],
        "Probability_Hawk": [0.20, 0.45, 0.05, 0.25, 0.05],
        "Probability_Dove": [0.15, 0.20, 0.30, 0.10, 0.25],
        "Political_Impact": ["+15 approval", "+5 approval", "-20 approval",
                           "-10 approval", "-30 approval"]
    }

    print(f"{'Outcome':<35} {'P(Hawk)':<10} {'P(Dove)':<10} {'Political Impact'}")
    print("-" * 70)

    for i, outcome in enumerate(political_payoffs["Outcome"]):
        print(f"{outcome:<35} {political_payoffs['Probability_Hawk'][i]:<10.0%} "
              f"{political_payoffs['Probability_Dove'][i]:<10.0%} "
              f"{political_payoffs['Political_Impact'][i]}")

    # Expected value calculation
    impacts = [15, 5, -20, -10, -30]
    ev_hawk = sum(p * i for p, i in zip(political_payoffs['Probability_Hawk'], impacts))
    ev_dove = sum(p * i for p, i in zip(political_payoffs['Probability_Dove'], impacts))

    print(f"\nExpected Political Impact:")
    print(f"  Hawkish Fed: {ev_hawk:+.1f} approval points")
    print(f"  Dovish Fed:  {ev_dove:+.1f} approval points")
    print(f"\n  HAWK ADVANTAGE: {ev_hawk - ev_dove:+.1f} points")


# =============================================================================
# MULTI-PLAYER EXTENSIVE FORM GAME
# =============================================================================

def analyze_extensive_form_game():
    """Analyze as extensive form game with multiple players."""

    print("\n" + "=" * 80)
    print("7. EXTENSIVE FORM GAME: MULTI-PLAYER DYNAMICS")
    print("=" * 80)

    print("""
GAME TREE STRUCTURE:
====================

                        TRUMP
                       /      \\
              Appoint Hawk    Appoint Dove
                  |              |
                WARSH          (DOVE)
               /     \\        /     \\
          Tight    Moderate  Tight   Loose
             |        |        |        |
          MARKETS  MARKETS  MARKETS  MARKETS
          /    \\   /    \\   /    \\   /    \\
       Risk   Risk Risk  Risk Risk Risk Risk Risk
       Off    On   Off   On   Off  On   Off  On
         |     |    |     |    |    |    |    |
      [Outcomes at terminal nodes...]


BACKWARD INDUCTION ANALYSIS:
============================

Step 1: Markets' Best Response
------------------------------
• If Hawk + Tight → Markets go Risk-Off (defensive)
• If Hawk + Moderate → Markets cautiously Risk-On
• If Dove + Tight → Markets confused, volatile
• If Dove + Loose → Markets initially Risk-On, then inflation fear

Step 2: Fed Chair's Best Response
---------------------------------
WARSH (if appointed):
• Anticipates market reaction
• Chooses "Moderate" to balance objectives
• Has CREDIBILITY to be moderate without inflation expectations spiking

DOVE (if appointed):
• Loose policy → inflation expectations rise → forced to tighten
• OR stays loose → inflation spiral
• NO GOOD OPTIONS because lacks credibility

Step 3: Trump's Optimal Choice
------------------------------
• Warsh → Moderate policy → Cautious Risk-On → Good outcome
• Dove → Bad equilibrium regardless of dove's choice

SUBGAME PERFECT EQUILIBRIUM:
Trump appoints Warsh → Warsh chooses Moderate → Markets cautiously optimistic


THE "COMMITMENT VALUE" OF HAWKISHNESS:
======================================

Warsh's hawkish REPUTATION allows him to be ACTUALLY moderate.

This is counterintuitive but follows from game theory:

• A known hawk saying "I'll be moderate" is BELIEVED
• A known dove saying "I'll be moderate" is NOT believed
• Markets price the EXPECTED policy, not announced policy
• Hawk's expected policy is anchored by reputation

Example:
• Warsh says "rates at 5%" → Markets believe 4.75-5.25% range
• Dove says "rates at 5%" → Markets believe 4.0-4.5% range (discount)

So Warsh can ACHIEVE the same actual policy with better expectations!
""")

    # Equilibrium outcomes
    print("\nEQUILIBRIUM ANALYSIS BY APPOINTMENT:")
    print("-" * 70)

    equilibria = {
        "Warsh_Appointed": {
            "fed_policy": "Moderate-Hawkish (5.0-5.5%)",
            "market_belief": "Credible, anchored",
            "actual_inflation": "2.5-3.0%",
            "growth": "2.0-2.5%",
            "market_outcome": "Cautious risk-on",
            "political_outcome": "Favorable",
            "equilibrium_type": "Subgame Perfect, Stable"
        },
        "Dove_Appointed": {
            "fed_policy": "Uncertain (markets don't believe)",
            "market_belief": "Inflation expectations elevated",
            "actual_inflation": "3.5-5.0%",
            "growth": "Boom-bust cycle",
            "market_outcome": "Volatile, eventual crisis",
            "political_outcome": "Unfavorable",
            "equilibrium_type": "Unstable, multiple equilibria"
        }
    }

    for appointment, outcome in equilibria.items():
        print(f"\n{appointment}:")
        for key, value in outcome.items():
            print(f"  {key}: {value}")


# =============================================================================
# WARSH-SPECIFIC STRATEGIC VALUE
# =============================================================================

def analyze_warsh_specific_value():
    """Analyze why Warsh specifically, not just any hawk."""

    print("\n" + "=" * 80)
    print("8. WHY WARSH SPECIFICALLY?")
    print("=" * 80)

    print("""
WARSH'S UNIQUE STRATEGIC VALUE:
===============================

Not just any hawk would do. Warsh has specific attributes:

1. INSIDER KNOWLEDGE
   • Former Fed Governor (2006-2011)
   • Knows where the bodies are buried
   • Understands Fed politics and bureaucracy
   • Can navigate institution effectively

2. CRISIS EXPERIENCE
   • Was at Fed during 2008 financial crisis
   • Understands emergency lending, market dysfunction
   • Won't panic in crisis
   • BUT skeptical of permanent emergency measures

3. GOLDMAN SACHS NETWORK
   • Understands Wall Street perspective
   • Has relationships with market participants
   • Can communicate with markets effectively
   • Treasury/Fed coordination easier

4. REPUBLICAN CREDIBILITY
   • Clearly aligned with conservative economics
   • Hoover Institution fellow
   • Won't be seen as compromising to Democrats
   • Gives Trump political cover on the right

5. YOUNG ENOUGH FOR LEGACY
   • Born 1970, still in prime
   • Could serve multiple terms
   • Thinking about historical legacy
   • Won't just "mail it in"

6. MEDIA SAVVY
   • Comfortable with public communication
   • WSJ contributor, knows financial media
   • Can explain policy clearly
   • Important for expectations management

7. RULES-BASED FRAMEWORK
   • Taylor Rule advocate = PREDICTABILITY
   • Markets can model Fed behavior
   • Reduces policy uncertainty
   • Helps Treasury with issuance planning


COMPARISON TO ALTERNATIVES:
===========================
""")

    alternatives = {
        "Kevin_Warsh": {
            "fed_experience": "★★★★★",
            "crisis_experience": "★★★★★",
            "market_credibility": "★★★★☆",
            "trump_alignment": "★★★★☆",
            "academic_respect": "★★★☆☆",
            "predictability": "★★★★☆",
            "confirmation_odds": "★★★★☆",
            "overall": "OPTIMAL CHOICE"
        },
        "John_Taylor": {
            "fed_experience": "★★★☆☆",
            "crisis_experience": "★★☆☆☆",
            "market_credibility": "★★★★★",
            "trump_alignment": "★★★☆☆",
            "academic_respect": "★★★★★",
            "predictability": "★★★★★",
            "confirmation_odds": "★★★☆☆",
            "overall": "Too academic, age concern"
        },
        "Judy_Shelton": {
            "fed_experience": "★☆☆☆☆",
            "crisis_experience": "★☆☆☆☆",
            "market_credibility": "★★☆☆☆",
            "trump_alignment": "★★★★★",
            "academic_respect": "★★☆☆☆",
            "predictability": "★★☆☆☆",
            "confirmation_odds": "★★☆☆☆",
            "overall": "Too controversial, lacks credibility"
        },
        "Larry_Kudlow": {
            "fed_experience": "★☆☆☆☆",
            "crisis_experience": "★★☆☆☆",
            "market_credibility": "★★★☆☆",
            "trump_alignment": "★★★★★",
            "academic_respect": "★☆☆☆☆",
            "predictability": "★★☆☆☆",
            "confirmation_odds": "★★★☆☆",
            "overall": "Too political, not credible hawk"
        },
        "Neel_Kashkari": {
            "fed_experience": "★★★★☆",
            "crisis_experience": "★★★★☆",
            "market_credibility": "★★★★☆",
            "trump_alignment": "★☆☆☆☆",
            "academic_respect": "★★★☆☆",
            "predictability": "★★★☆☆",
            "confirmation_odds": "★★☆☆☆",
            "overall": "Wrong political alignment"
        }
    }

    print(f"{'Candidate':<18} {'Fed Exp':<8} {'Crisis':<8} {'Markets':<8} {'Trump':<8} {'Predict':<8} {'Overall'}")
    print("-" * 80)

    for name, scores in alternatives.items():
        print(f"{name:<18} {scores['fed_experience']:<8} {scores['crisis_experience']:<8} "
              f"{scores['market_credibility']:<8} {scores['trump_alignment']:<8} "
              f"{scores['predictability']:<8} {scores['overall']}")


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def print_executive_summary():
    """Print the executive summary."""

    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: WHY TRUMP APPOINTED WARSH")
    print("=" * 80)

    print("""
THE GAME THEORY ANSWER:
=======================

Trump's appointment of Warsh is NOT a contradiction - it's sophisticated
game theory in action. Here's why:

1. TRUMP'S TRUE UTILITY ≠ STATED PREFERENCES
   --------------------------------------------
   • Public: "I want low rates"
   • Actual: "I want low INFLATION, growth, and coordination"
   • Inflation control (learned from Biden) dominates rate preferences
   • 35% of Trump's utility served by hawkish Fed

2. SIGNALING & CREDIBILITY
   ------------------------
   • Appointing hawk signals seriousness to bond markets
   • Lowers Treasury yields through credibility premium
   • Essential for financing $35T+ debt
   • Signal to global actors: dollar will be defended

3. TIME INCONSISTENCY SOLUTION
   ---------------------------
   • Classic Rogoff "conservative central banker" model
   • Warsh's hawkishness solves Trump's commitment problem
   • "Tying himself to the mast" - Odysseus strategy
   • Better long-run equilibrium through delegation

4. COORDINATION WITHOUT SUBORDINATION
   -----------------------------------
   • Warsh is independent enough to be credible
   • Aligned enough to coordinate with Treasury
   • Provides political cover for fiscal expansion
   • "Good cop / bad cop" dynamic works

5. POLITICAL RISK MANAGEMENT
   --------------------------
   • Biden's inflation disaster = cautionary tale
   • Expected political value of hawk > dove
   • Blame allocation: independent Fed absorbs downside
   • Credit allocation: Trump takes upside

6. WARSH-SPECIFIC VALUE
   --------------------
   • Insider knowledge of Fed
   • Crisis experience (2008)
   • Goldman network for coordination
   • Rules-based = predictable
   • Right age, right credentials


BOTTOM LINE:
============

The appointment reveals Trump (or his advisors) understand:
• Monetary economics (time inconsistency)
• Political economy (inflation punishment)
• Game theory (signaling, commitment)
• Institutional dynamics (Fed independence value)

Appointing a hawk is the DOMINANT STRATEGY when you account for:
• Rational expectations
• Long-run equilibrium
• Political accountability
• Debt sustainability

Trump's criticism of Powell was about COORDINATION, not DOVISHNESS.
Warsh provides hawkish CREDIBILITY with Trump-aligned COORDINATION.

This is not a contradiction - it's optimal mechanism design.

"The best way to get low inflation is to appoint someone who REALLY
hates inflation. Then you don't have to do anything else."
    - Simplified Rogoff theorem
""")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_full_analysis():
    """Run the complete analysis."""

    print_the_paradox()
    analyze_trump_utility()
    analyze_signaling_game()
    analyze_time_inconsistency()
    analyze_treasury_coordination()
    analyze_fed_independence()
    analyze_inflation_politics()
    analyze_extensive_form_game()
    analyze_warsh_specific_value()
    print_executive_summary()

    return {
        "conclusion": "Warsh appointment is game-theoretically optimal",
        "key_insight": "Hawkishness is a feature, not a bug",
        "mechanism": "Credible commitment solves time inconsistency",
        "political_logic": "Inflation prevention > Rate preferences"
    }


if __name__ == "__main__":
    result = run_full_analysis()
