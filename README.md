# Elite CGT/AGT Strategic Analysis System

## Computational & Algorithmic Game Theory for Macro-Financial Analysis

A production-grade framework for analyzing complex strategic games in geopolitics, monetary policy, and financial markets using advanced game theory, Bayesian inference, and causal modeling.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 🎯 Overview

This system applies elite quantitative methods to analyze:

- **US-China Grand Strategy** - Technology, trade, finance, geopolitics
- **Federal Reserve Policy** - Monetary policy implications (Warsh analysis)
- **Precious Metals Markets** - Brandt vs Oliver thesis analysis
- **Global Monetary Regime** - Dollar hegemony vs multipolar transition

### Core Capabilities

| Module | Description |
|--------|-------------|
| **Nash Equilibrium Solver** | N-player games via support enumeration, replicator dynamics, fictitious play |
| **Evolutionary Dynamics** | ESS finding, invasion analysis, population dynamics |
| **Mean Field Games** | HJB-FPK coupled PDEs for large population limits |
| **Bayesian Scenario Engine** | Soft Bayes updating with log-likelihood ratios |
| **Causal Graph Propagation** | Nth-order effects with decay factors |
| **Strategic Game Templates** | Pre-built US-China, precious metals, Fed policy games |

---

## 📁 Project Structure

```
CGT_AGT_PROJECT/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── elite_cgt_agt_system.py      # Core system
│   ├── nash_solver.py                # Nash equilibrium computation
│   ├── bayesian_engine.py            # Scenario analysis
│   ├── causal_graph.py               # Causal propagation
│   └── game_templates.py             # Pre-built strategic games
│
├── analyses/
│   ├── brandt_vs_oliver_silver.py    # Silver market analysis
│   ├── warsh_fed_chair.py            # Fed policy analysis
│   ├── trump_warsh_game_theory.py    # Appointment rationale
│   └── grand_unified_analysis.py     # Integrated framework
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py                     # MCP server implementation
│   └── config.json                   # Server configuration
│
├── tests/
│   ├── test_nash_solver.py
│   ├── test_bayesian_engine.py
│   └── test_causal_graph.py
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── QA_ANALYSIS.md
│   └── API_REFERENCE.md
│
└── claude_project/
    ├── custom_instructions.md        # Claude Project instructions
    └── knowledge_base/               # Files for Claude Project
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cgt-agt-strategic-analysis.git
cd cgt-agt-strategic-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.elite_cgt_agt_system import EliteCGTSystem

# Initialize system
system = EliteCGTSystem()

# Run US-China analysis
equilibrium = system.analyze_us_china_game(domain="tech")
print(f"Nash Equilibrium: US={equilibrium['us_strategy']}, China={equilibrium['china_strategy']}")

# Run Bayesian scenario analysis
scenarios = system.build_scenarios("precious_metals")
posteriors = system.update_with_evidence({
    "dollar_weakness": 0.4,
    "gold_performance": 0.7,
    "bond_instability": 0.5
})
print(f"Scenario Probabilities: {posteriors}")
```

### Run Full Analyses

```bash
# Silver Market Analysis (Brandt vs Oliver)
python analyses/brandt_vs_oliver_silver.py

# Fed Chair Analysis (Warsh)
python analyses/warsh_fed_chair.py

# Grand Unified Analysis
python analyses/grand_unified_analysis.py

# QA Deep Dive
python analyses/qa_deep_dive.py
```

---

## 🔧 Claude Integration

### Option 1: Claude Project (Recommended)

1. Go to [Claude.ai](https://claude.ai) → Projects → Create Project
2. Name: "CGT/AGT Strategic Analysis"
3. Add custom instructions from `claude_project/custom_instructions.md`
4. Upload key files to Project Knowledge

### Option 2: MCP Server (Advanced)

```bash
# Start the MCP server
python mcp_server/server.py

# Configure in Claude Desktop settings
# Add to ~/.config/claude/config.json
```

### Option 3: Web API Deployment

```bash
# Run FastAPI server
uvicorn api.main:app --reload

# Access at http://localhost:8000/docs
```

---

## 📊 Key Analyses Included

### 1. Brandt vs Oliver Silver Analysis
- Bayesian scenario probabilities
- Multi-player strategic game (Miners, Speculators, Industry)
- Evolutionary dynamics of market strategies
- Expected value: **$90-$208 (90% CI)**

### 2. Warsh Fed Chair Analysis
- 5 policy scenarios with probability weighting
- Asset class impact matrix
- Nth-order causal effects
- Hawkish probability: **76%**

### 3. Trump-Warsh Game Theory
- Why appoint a hawk? (Time inconsistency, signaling, Rogoff model)
- Fed independence strategic calculus
- Political economy of inflation

### 4. Grand Unified Analysis
- Master causal graph (50+ nodes)
- Integrated scenario matrix
- Cross-game dependencies
- 12 additional games to investigate

---

## ⚠️ QA Assessment & Confidence Levels

| Dimension | Confidence |
|-----------|------------|
| Mathematical Correctness | ✅ HIGH |
| Logical Consistency | ✅ HIGH |
| Directional Relationships | ✅ HIGH |
| Quantitative Precision | ⚠️ LOW |
| Timing Predictions | ❌ VERY LOW |

**Use as**: Structured thinking framework
**Don't use as**: Precise prediction engine

See `docs/QA_ANALYSIS.md` for full validation report.

---

## 🛠️ API Reference

### Nash Equilibrium Solver

```python
solver = NashEquilibriumSolver(tolerance=1e-6, max_iterations=10000)

# 2-player bimatrix game
sigma_a, sigma_b = solver.solve_two_player_bimatrix(
    payoff_A, payoff_B,
    method="support_enumeration"  # or "replicator", "fictitious_play"
)

# N-player game
strategies = solver.solve_n_player(payoffs, players)
```

### Bayesian Scenario Engine

```python
engine = BayesianScenarioEngine(scenarios)

# Update with evidence
posteriors = engine.soft_bayes_update({
    "MONETARY_CRISIS": 0.5,      # log-likelihood ratio
    "PAX_AMERICANA": -0.3
})
```

### Causal Graph

```python
graph = CausalGraph(edges=custom_edges)  # or use DEFAULT_EDGES

# Propagate shock
effects = graph.propagate(
    root="DollarWeakness",
    direction="UP",
    max_depth=3,
    decay_factor=0.55
)
```

---

## 📚 Documentation

- [Methodology](docs/METHODOLOGY.md) - Theoretical foundations
- [QA Analysis](docs/QA_ANALYSIS.md) - Validation and confidence assessment
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation

---

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

### Priority Improvements

1. **Data Integration** - Connect to real-time market data APIs
2. **Parameter Estimation** - Empirical calibration of causal weights
3. **Uncertainty Quantification** - Bootstrap confidence intervals
4. **Validation Framework** - Track predictions vs outcomes

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Game theory foundations: Nash, Rogoff, Kydland-Prescott
- Bayesian methods: Soft Bayes with log-likelihood ratios
- Causal inference: Pearl's do-calculus framework

---

## 📧 Contact

For questions or collaboration: [your-email@example.com]

---

*"This is a THINKING FRAMEWORK, not a PREDICTION ENGINE. Use it as a MAP, not GPS with turn-by-turn directions."*
