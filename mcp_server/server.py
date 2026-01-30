#!/usr/bin/env python3
"""
================================================================================
MCP SERVER: CGT/AGT Strategic Analysis System
================================================================================

Model Context Protocol server that exposes the CGT/AGT system to Claude.

This allows Claude to:
1. Run game-theoretic analyses
2. Update Bayesian scenarios with evidence
3. Propagate causal shocks
4. Query scenario probabilities and expected values

Setup:
1. Install dependencies: pip install mcp httpx
2. Run server: python server.py
3. Configure in Claude Desktop settings

================================================================================
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MCP imports (install with: pip install mcp)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Install with: pip install mcp")

import numpy as np
import math


# =============================================================================
# CORE ANALYSIS FUNCTIONS (Simplified for MCP)
# =============================================================================

class CGTAnalysisEngine:
    """Simplified CGT/AGT engine for MCP server."""

    def __init__(self):
        self.scenarios = self._init_scenarios()
        self.causal_edges = self._init_causal_graph()

    def _init_scenarios(self) -> Dict[str, Dict]:
        """Initialize scenario definitions."""
        return {
            "PAX_AMERICANA": {
                "description": "US hegemony maintained, Warsh succeeds",
                "prior": 0.20,
                "posterior": 0.20,
                "silver_target": (25, 50),
                "gold_target": (2000, 2500)
            },
            "MANAGED_MULTIPOLARITY": {
                "description": "Gradual transition, competition without crisis",
                "prior": 0.30,
                "posterior": 0.30,
                "silver_target": (100, 175),
                "gold_target": (3500, 5000)
            },
            "STAGFLATION": {
                "description": "Warsh fights inflation, US-China tensions elevate prices",
                "prior": 0.20,
                "posterior": 0.20,
                "silver_target": (75, 150),
                "gold_target": (3000, 4500)
            },
            "MONETARY_CRISIS": {
                "description": "Dollar crisis, Oliver thesis materializes",
                "prior": 0.15,
                "posterior": 0.15,
                "silver_target": (250, 500),
                "gold_target": (7000, 10000)
            },
            "GEOPOLITICAL_SHOCK": {
                "description": "Taiwan, Middle East, or other black swan",
                "prior": 0.15,
                "posterior": 0.15,
                "silver_target": (100, 200),
                "gold_target": (4000, 6000)
            }
        }

    def _init_causal_graph(self) -> Dict[str, Dict[str, float]]:
        """Initialize causal relationships."""
        return {
            "Warsh_Hawkish": {
                "US_Rates": 0.85,
                "Dollar_Strength": 0.60,
                "Inflation_Expectations": -0.65,
                "Fed_Credibility": 0.75
            },
            "Dollar_Strength": {
                "Gold_Price": -0.35,
                "Silver_Price": -0.40,
                "EM_Pressure": 0.55,
                "De_Dollarization": 0.35
            },
            "US_China_Decoupling": {
                "De_Dollarization": 0.65,
                "Gold_CB_Buying": 0.60,
                "Supply_Chain_Costs": 0.50
            },
            "De_Dollarization": {
                "Gold_CB_Buying": 0.70,
                "Oliver_Thesis": 0.65,
                "Dollar_Reserve_Share": -0.45
            },
            "Gold_Price": {
                "Silver_Price": 0.65,
                "Mining_Equities": 0.70
            }
        }

    def update_scenarios_with_evidence(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """
        Update scenario probabilities with evidence.

        Args:
            evidence: Dict mapping evidence type to value (-1 to +1)
                     Positive = bullish for precious metals
                     Negative = bearish

        Evidence types:
        - dollar_weakness: DXY moves (positive = weak dollar = bullish)
        - gold_performance: Gold price action
        - bond_instability: Treasury volatility
        - inflation_data: CPI/PCE readings
        - fed_hawkishness: Fed rhetoric/actions
        - china_gold_buying: CB accumulation
        """
        evidence_weights = {
            "PAX_AMERICANA": {
                "dollar_weakness": -0.5, "gold_performance": -0.3,
                "bond_instability": -0.4, "inflation_data": -0.3,
                "fed_hawkishness": 0.6, "china_gold_buying": -0.3
            },
            "MANAGED_MULTIPOLARITY": {
                "dollar_weakness": 0.4, "gold_performance": 0.5,
                "bond_instability": 0.3, "inflation_data": 0.2,
                "fed_hawkishness": -0.2, "china_gold_buying": 0.5
            },
            "STAGFLATION": {
                "dollar_weakness": 0.2, "gold_performance": 0.4,
                "bond_instability": 0.3, "inflation_data": 0.7,
                "fed_hawkishness": 0.3, "china_gold_buying": 0.3
            },
            "MONETARY_CRISIS": {
                "dollar_weakness": 0.8, "gold_performance": 0.7,
                "bond_instability": 0.9, "inflation_data": 0.5,
                "fed_hawkishness": -0.5, "china_gold_buying": 0.7
            },
            "GEOPOLITICAL_SHOCK": {
                "dollar_weakness": 0.3, "gold_performance": 0.5,
                "bond_instability": 0.5, "inflation_data": 0.2,
                "fed_hawkishness": -0.2, "china_gold_buying": 0.4
            }
        }

        posteriors = {}
        posteriors_unnorm = []

        for scenario_name, scenario in self.scenarios.items():
            log_lr = 0.0
            weights = evidence_weights.get(scenario_name, {})

            for ev_type, ev_value in evidence.items():
                weight = weights.get(ev_type, 0.0)
                log_lr += weight * ev_value

            posterior_unnorm = scenario["prior"] * math.exp(log_lr)
            posteriors_unnorm.append((scenario_name, posterior_unnorm))

        # Normalize
        total = sum(p for _, p in posteriors_unnorm)
        for name, p in posteriors_unnorm:
            self.scenarios[name]["posterior"] = p / total
            posteriors[name] = p / total

        return posteriors

    def calculate_expected_values(self) -> Dict[str, Dict[str, float]]:
        """Calculate probability-weighted expected values."""
        ev_silver_low = 0
        ev_silver_high = 0
        ev_gold_low = 0
        ev_gold_high = 0

        for name, scenario in self.scenarios.items():
            p = scenario["posterior"]
            ev_silver_low += p * scenario["silver_target"][0]
            ev_silver_high += p * scenario["silver_target"][1]
            ev_gold_low += p * scenario["gold_target"][0]
            ev_gold_high += p * scenario["gold_target"][1]

        return {
            "silver": {
                "low": round(ev_silver_low, 2),
                "high": round(ev_silver_high, 2),
                "mid": round((ev_silver_low + ev_silver_high) / 2, 2)
            },
            "gold": {
                "low": round(ev_gold_low, 2),
                "high": round(ev_gold_high, 2),
                "mid": round((ev_gold_low + ev_gold_high) / 2, 2)
            }
        }

    def propagate_causal_shock(
        self,
        root: str,
        direction: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Propagate a shock through the causal graph."""
        sign = 1.0 if direction.upper() == "UP" else -1.0
        decay = 0.55

        effects = {}
        frontier = [(root, sign, 0)]
        visited = {root: 0}

        while frontier:
            node, magnitude, depth = frontier.pop(0)
            if depth >= max_depth:
                continue

            for neighbor, weight in self.causal_edges.get(node, {}).items():
                new_effect = magnitude * weight * decay
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
                "magnitude": round(abs(score), 4)
            })

        return results[:10]

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of all scenarios."""
        scenarios = []
        for name, data in self.scenarios.items():
            scenarios.append({
                "name": name,
                "description": data["description"],
                "prior": round(data["prior"] * 100, 1),
                "posterior": round(data["posterior"] * 100, 1),
                "silver_range": f"${data['silver_target'][0]}-${data['silver_target'][1]}",
                "gold_range": f"${data['gold_target'][0]}-${data['gold_target'][1]}"
            })

        return {
            "scenarios": sorted(scenarios, key=lambda x: x["posterior"], reverse=True),
            "expected_values": self.calculate_expected_values()
        }


# =============================================================================
# MCP SERVER IMPLEMENTATION
# =============================================================================

# Global engine instance
engine = CGTAnalysisEngine()


def create_mcp_server():
    """Create and configure the MCP server."""

    if not MCP_AVAILABLE:
        print("MCP not available. Install with: pip install mcp")
        return None

    server = Server("cgt-agt-analysis")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="get_scenario_probabilities",
                description="Get current scenario probabilities and expected values for precious metals",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="update_with_evidence",
                description="Update scenario probabilities with market evidence. Evidence values range from -1 (bearish) to +1 (bullish).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dollar_weakness": {
                            "type": "number",
                            "description": "Dollar weakness indicator (-1 to +1). Positive = weak dollar = bullish PM"
                        },
                        "gold_performance": {
                            "type": "number",
                            "description": "Gold price performance (-1 to +1)"
                        },
                        "bond_instability": {
                            "type": "number",
                            "description": "Treasury market volatility (-1 to +1)"
                        },
                        "inflation_data": {
                            "type": "number",
                            "description": "Inflation readings vs expectations (-1 to +1)"
                        },
                        "fed_hawkishness": {
                            "type": "number",
                            "description": "Fed hawkish rhetoric/actions (-1 to +1)"
                        },
                        "china_gold_buying": {
                            "type": "number",
                            "description": "Central bank gold accumulation (-1 to +1)"
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="propagate_shock",
                description="Propagate a shock through the causal graph to see Nth-order effects",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "root": {
                            "type": "string",
                            "description": "Starting node for shock (e.g., 'Warsh_Hawkish', 'Dollar_Strength', 'US_China_Decoupling')"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["UP", "DOWN"],
                            "description": "Direction of the shock"
                        }
                    },
                    "required": ["root", "direction"]
                }
            ),
            Tool(
                name="analyze_fed_policy",
                description="Get analysis of Fed policy scenarios under Warsh",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="analyze_us_china",
                description="Get analysis of US-China strategic dynamics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "enum": ["tech", "trade", "finance"],
                            "description": "Strategic domain to analyze"
                        }
                    },
                    "required": []
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""

        if name == "get_scenario_probabilities":
            result = engine.get_scenario_summary()
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "update_with_evidence":
            evidence = {k: v for k, v in arguments.items() if v is not None}
            posteriors = engine.update_scenarios_with_evidence(evidence)
            result = {
                "updated_probabilities": {k: f"{v*100:.1f}%" for k, v in posteriors.items()},
                "expected_values": engine.calculate_expected_values(),
                "evidence_used": evidence
            }
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "propagate_shock":
            root = arguments.get("root", "Warsh_Hawkish")
            direction = arguments.get("direction", "UP")
            effects = engine.propagate_causal_shock(root, direction)
            result = {
                "shock": f"{root} {direction}",
                "nth_order_effects": effects
            }
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "analyze_fed_policy":
            result = {
                "warsh_assessment": {
                    "hawkish_probability": "76%",
                    "scenarios": [
                        {"name": "Hawkish but Pragmatic", "probability": "41%", "terminal_rate": "4.75-5.50%"},
                        {"name": "Volcker Redux", "probability": "35%", "terminal_rate": "5.50-6.50%"},
                        {"name": "Constrained Hawk", "probability": "13%", "terminal_rate": "4.25-5.00%"},
                        {"name": "Crisis Response", "probability": "4%", "terminal_rate": "2.00-3.50%"}
                    ],
                    "asset_impacts": {
                        "equities": "BEARISH (-0.30)",
                        "dollar": "BULLISH (+0.43)",
                        "gold": "NEUTRAL short-term (-0.05)",
                        "treasuries": "BEARISH (-0.25)"
                    },
                    "key_insight": "Warsh appointment is game-theoretically optimal: solves time inconsistency, provides credible commitment"
                }
            }
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "analyze_us_china":
            domain = arguments.get("domain", "tech")
            result = {
                "domain": domain,
                "equilibrium": {
                    "us_strategy": "Compete" if domain == "tech" else "Compete",
                    "china_strategy": "Compete" if domain == "tech" else "Compete"
                },
                "dynamics": {
                    "tech": "US confront/China compete equilibrium - decoupling accelerating",
                    "trade": "Mutual competition with selective cooperation",
                    "finance": "US competes/China decouples - de-dollarization pressure"
                }.get(domain, "Competition equilibrium"),
                "implications": {
                    "tech": "Supply chain restructuring, chip shortage risk",
                    "trade": "Tariffs persist, inflation pressure",
                    "finance": "De-dollarization accelerates, gold demand rises"
                }.get(domain, "Elevated competition")
            }
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    return server


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("="*60)
        print("MCP SERVER - CGT/AGT Strategic Analysis")
        print("="*60)
        print("\nMCP library not installed.")
        print("Install with: pip install mcp")
        print("\nFor testing without MCP, use the API server instead:")
        print("  uvicorn api.main:app --reload")
        return

    server = create_mcp_server()
    if server:
        print("Starting CGT/AGT MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
