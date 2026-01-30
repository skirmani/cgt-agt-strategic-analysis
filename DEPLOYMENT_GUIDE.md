# CGT/AGT Strategic Analysis System - Deployment Guide

## 🎉 Your Project is Now Live!

**GitHub Repository**: https://github.com/skirmani/cgt-agt-strategic-analysis

---

## 3 Ways to Use This System with Claude

### Option 1: Claude Project (RECOMMENDED - Easiest)

This embeds the system knowledge directly into Claude on claude.ai.

**Steps:**

1. Go to [claude.ai](https://claude.ai)
2. Click **"Projects"** in the sidebar
3. Click **"Create Project"**
4. Name it: `CGT/AGT Strategic Analysis`
5. Click **"Edit Project Instructions"**
6. Copy the contents of `claude_project/custom_instructions.md` into the instructions
7. Click **"Add Content"** → **"Upload Files"**
8. Upload these key files:
   - `src/elite_cgt_agt_system.py`
   - `analyses/BRANDT_VS_OLIVER_SILVER_ANALYSIS.py`
   - `analyses/WARSH_FED_CHAIR_ANALYSIS.py`
   - `analyses/GRAND_UNIFIED_STRATEGIC_ANALYSIS.py`
   - `docs/QA_EXECUTIVE_SUMMARY.md`
9. Start chatting in the project!

**Example prompts:**
- "Analyze the current state of the Brandt vs Oliver thesis"
- "What are the Nash equilibria in the US-China tech game?"
- "Update the scenario probabilities given recent Fed minutes"

---

### Option 2: MCP Server (Advanced - For Claude Desktop)

This gives Claude direct access to run the analysis tools.

**Steps:**

1. Install dependencies:
```bash
cd /path/to/CGT_AGT_PROJECT
pip install -r requirements.txt
```

2. Edit Claude Desktop config at `~/.config/claude/config.json` (macOS/Linux) or `%APPDATA%\Claude\config.json` (Windows):

```json
{
  "mcpServers": {
    "cgt-agt": {
      "command": "python",
      "args": ["/path/to/CGT_AGT_PROJECT/mcp_server/server.py"]
    }
  }
}
```

3. Restart Claude Desktop

4. Claude can now use tools like:
   - `get_scenario_probabilities` - Get current Bayesian posteriors
   - `update_with_evidence` - Update probabilities with new evidence
   - `propagate_shock` - Run causal propagation analysis

---

### Option 3: Web API (For Apps/Integrations)

This runs a REST API you can call from any application.

**Local Development:**
```bash
cd /path/to/CGT_AGT_PROJECT
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Access at: http://localhost:8000/docs (Swagger UI)

**Cloud Deployment (Render.com - Free):**

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click **"New"** → **"Web Service"**
3. Connect your repository: `skirmani/cgt-agt-strategic-analysis`
4. Configure:
   - **Name**: `cgt-agt-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Click **"Create Web Service"**

Your API will be live at: `https://cgt-agt-api.onrender.com`

**API Endpoints:**
- `GET /scenarios` - Get all scenario probabilities
- `POST /scenarios/update` - Update with evidence
- `POST /causal/propagate` - Run causal propagation
- `GET /analysis/fed` - Fed Chair analysis
- `GET /analysis/us-china` - US-China game analysis
- `GET /analysis/precious-metals` - Silver/Gold analysis

---

## Quick Start Example

Once you've set up a Claude Project, try these:

### Analyze Silver Market
```
What's the current Bayesian probability distribution across the
Brandt vs Oliver scenarios? Walk me through the evidence factors.
```

### Update with New Evidence
```
The Fed just signaled a more dovish stance than expected.
How does this update our scenario probabilities for precious metals?
```

### Game Theory Deep Dive
```
Model the current US-China technology competition as a game.
Who are the players, what are their strategies, and what's the equilibrium?
```

### Causal Analysis
```
If the dollar weakens significantly (DXY < 95), propagate this
shock through the causal graph and show me the Nth-order effects.
```

---

## Files in This Repository

```
CGT_AGT_PROJECT/
├── src/
│   ├── elite_cgt_agt_system.py    # Core framework (10K+ lines)
│   └── __init__.py
├── analyses/
│   ├── BRANDT_VS_OLIVER_SILVER_ANALYSIS.py
│   ├── WARSH_FED_CHAIR_ANALYSIS.py
│   ├── TRUMP_WARSH_GAME_THEORY.py
│   ├── GRAND_UNIFIED_STRATEGIC_ANALYSIS.py
│   └── QA_DEEP_DIVE_ANALYSIS.py
├── mcp_server/
│   ├── server.py                   # MCP server for Claude Desktop
│   └── config.json
├── api/
│   └── main.py                     # FastAPI web service
├── claude_project/
│   └── custom_instructions.md      # Copy this to Claude Project
├── docs/
│   └── QA_EXECUTIVE_SUMMARY.md     # Confidence assessment
├── README.md
├── requirements.txt
└── setup.py
```

---

## Key System Capabilities

| Module | What It Does |
|--------|--------------|
| **Nash Solver** | Finds equilibria in strategic games |
| **Bayesian Engine** | Updates scenario probabilities with evidence |
| **Causal Graph** | Propagates shocks through 50+ economic factors |
| **Game Templates** | Pre-built US-China, Fed, precious metals games |

---

## Confidence Reminder

| Trust This | Use With Caution | Don't Trust |
|------------|------------------|-------------|
| Logic & structure | Point estimates | Precise timing |
| Directional insights | Exact probabilities | Specific prices |
| Scenario identification | Payoff magnitudes | When events occur |

**This is a THINKING FRAMEWORK, not a PREDICTION ENGINE.**

---

## Support

- **GitHub Issues**: https://github.com/skirmani/cgt-agt-strategic-analysis/issues
- **Documentation**: See `docs/` folder

---

*Built with Claude Code*
