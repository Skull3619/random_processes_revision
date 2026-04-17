# ISE 5414 — Random Processes Study Hub

An interactive Streamlit study app for ISE 5414 (Random Processes) based on Ross Chapters 1–6.

## Features

| Feature | Description |
|---|---|
| 📖 Chapter Tabs | 6 chapters, each with subtopics organized in tabs |
| 🎯 Interactive Visuals | Plotly-based charts: distribution explorer, CLT demo, Gambler's Ruin simulator, Poisson process paths, Markov chain analyzer, CTMC steady-state solver, and more |
| 🔍 Topic Finder | Search 140+ indexed concepts to find exact chapter, section, and page |
| 📄 PDF Download | Generate a full study-notes PDF (definitions, theorems, formulas Ch 1–6) for AI-assisted learning |

## Content Covered

- **Ch 1:** Probability axioms, conditional probability, Bayes' theorem, continuity of probability  
- **Ch 2:** Distributions (Bernoulli, Binomial, Poisson, Geometric, Exponential, Gamma, Normal), expectation, variance, MGFs, CLT, SLLN  
- **Ch 3:** Conditional expectation, tower property, law of total variance, random sums  
- **Ch 4:** Markov chains, stationary distributions, Gambler's ruin, branching processes, time reversibility, MDPs  
- **Ch 5:** Exponential distribution (memoryless), Poisson process, splitting, superposition, compound/nonhomogeneous Poisson  
- **Ch 6:** CTMCs, birth-death processes, Kolmogorov equations, M/M/1 queue, time reversibility  

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ise5414-study-hub.git
cd ise5414-study-hub

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (Free)

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ISE 5414 Study Hub"
   git remote add origin https://github.com/YOUR_USERNAME/ise5414-study-hub.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Repository: `YOUR_USERNAME/ise5414-study-hub`
   - Branch: `main`
   - Main file path: `app.py`
   - Click **"Deploy!"**

3. **Done** — your app will be live at `https://YOUR_USERNAME-ise5414-study-hub-app-XXXXX.streamlit.app`

> **Note:** Streamlit Community Cloud is free for public repos. The app auto-rebuilds whenever you push to `main`.

---

## File Structure

```
.
├── app.py              # Main Streamlit application (all chapters)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Standard Python gitignore
```

---

## Tech Stack

- **Streamlit** — UI framework
- **Plotly** — Interactive charts
- **NumPy / SciPy** — Numerical computations  
- **Pandas** — Data tables  
- **fpdf2** — PDF generation
