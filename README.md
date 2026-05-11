# Job Search Agent — LLM-Powered Intelligent Job Matching & Resume Tailoring

A single-LLM agentic pipeline where **LLaMA 3.3-70B** (via Groq) acts as the reasoning engine, dynamically deciding which tool to invoke at each step to filter, rank, and match jobs to a candidate profile — then tailoring the resume for the top result.

---

## What It Does

Most job search tools apply rigid, hard-coded logic. This agent uses an LLM as the **central reasoning engine** — given a candidate profile and a job dataset, it decides dynamically when to call each tool and what to do with the output, rather than following a fixed script.

```
Job Dataset (CSV)
      │
      ▼
┌─────────────┐     Removes excluded companies, experience mismatches,
│ Filter Tool │ ──► location/remote mismatches
└─────────────┘
      │
      ▼
┌─────────────┐     Scores each job across skill overlap (60pts),
│  Rank Tool  │ ──► experience alignment (30pts), location match (10pts)
└─────────────┘
      │
      ▼
┌──────────────────────┐     LLaMA 3.3-70B rewrites resume bullets
│ Resume Tailoring Tool│ ──► to match the top-ranked JD
└──────────────────────┘
      │
      ▼
  Tailored Resume Output
```

---

## Architecture

| Component | Details |
|-----------|---------|
| **LLM** | LLaMA 3.3-70B via Groq API |
| **Tool Orchestration** | LLM decides tool call sequence dynamically |
| **State Management** | Shared state dict persists filtered/ranked results across steps |
| **Dataset** | 24 ML Engineer job listings (CSV) with title, company, location, skills, JD |

---

## Pipeline Stages

### Stage 1 — Filter Tool
Removes jobs that fail any of these constraints:
- Company is in the candidate's excluded list
- Required experience exceeds candidate's max threshold
- Location doesn't match preferred cities AND job is not remote (when candidate is open to remote)

Outputs a filtered DataFrame + removal log with reasons per rejected job.

### Stage 2 — Rank Tool
Scores each remaining job out of 100 using configurable weights:

| Dimension | Max Points | Logic |
|-----------|-----------|-------|
| Skill Match | 60 | % overlap between candidate skills and required skills |
| Experience Alignment | 30 | Scales by distance from required years (exact match = full score) |
| Location / Remote | 10 | Remote match = full points; preferred city = partial credit |

Jobs are sorted by total score descending. Score breakdown (per-dimension) is preserved for inspection.

### Stage 3 — Resume Tailoring Tool
LLaMA 3.3-70B receives the top-ranked job's full description and the candidate's resume, then rewrites the experience bullet points to align with the JD — highlighting relevant skills, matching terminology, and reordering emphasis without fabricating experience.

---

## Candidate Profile

The agent takes a structured candidate profile as input:

```python
CANDIDATE = {
    "name": "...",
    "years_of_experience": 3,
    "location": "Houston, TX",
    "open_to_remote": True,
    "preferred_locations": ["Remote", "Houston, TX", "Austin, TX", ...],
    "excluded_companies": ["Meta", "TikTok"],
    "max_experience_required": 5,
    "skills": ["Python", "PyTorch", "BERT", "RAG", "LangChain", ...]
}
```

Update this dictionary in **Cell 5** to match your own profile before running.

---

## Setup & Usage

### Prerequisites
- Python 3.9+
- Free Groq API key — [console.groq.com](https://console.groq.com)
- Job listings CSV (sample format below)

### Install dependencies
```bash
pip install groq pandas numpy
```

### Run
Open `job_search_agent.ipynb` in Jupyter or Google Colab and run all cells in order. You'll be prompted to enter your Groq API key securely (hidden input, not stored).

### CSV Format
The agent expects a CSV with these columns:

| Column | Example |
|--------|---------|
| Job Title | Machine Learning Engineer |
| Company | Acme Corp |
| Location | San Francisco |
| Remote | Yes / No |
| Required Skills | Python, PyTorch, SQL |
| Years of Experience | 2-4 years |
| Job Description | Build and deploy... |
| URL | https://... |

---

## Scoring Weights

Weights are configurable at the top of the ranking cell:

```python
SCORE_WEIGHTS = {
    "skill":    60,   # skill match importance
    "exp":      30,   # experience alignment importance
    "loc":      10,   # location/remote match bonus
    "loc_pref":  8,   # partial credit for preferred city
}
```

Adjust to prioritize skill fit over location, or vice versa.

---

## Key Design Decisions

**Why a single LLM agent instead of a multi-agent system?**
The filtering and ranking tools are deterministic (rule-based + scoring) — they don't need LLM reasoning. The LLM is used where it adds the most value: orchestrating tool calls based on state and generating the tailored resume output. This keeps latency low and costs minimal while maintaining agent flexibility.

**Why LLaMA 3.3-70B via Groq?**
Groq's inference hardware delivers sub-second token generation on 70B parameter models — making the pipeline fast enough for interactive use without a GPU.

**Why structured state management?**
A shared `state` dict persists filtered/ranked DataFrames and the tailored resume across pipeline steps, preventing context loss between tool calls and enabling inspection of intermediate outputs.

---

## Tech Stack

`Python` · `LLaMA 3.3-70B` · `Groq API` · `Pandas` · `NumPy` · `Jupyter`

---

---

## Author

[Sahithi Akunoor](https://linkedin.com/in/sahithi-akunoor) · [github.com/sahithiakunoor](https://github.com/sahithiakunoor)
