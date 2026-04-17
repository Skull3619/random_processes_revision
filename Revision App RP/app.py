"""
ISE 5414 — Random Processes Study Hub
Streamlit Application — All content sourced from course chapters 1–6.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as sps
import pandas as pd
from io import BytesIO
import math
from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISE 5414 — Random Processes Study Hub",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.def-box   {background:#e3f2fd;padding:12px 16px;border-left:4px solid #1565c0;border-radius:4px;margin:8px 0;}
.thm-box   {background:#fff8e1;padding:12px 16px;border-left:4px solid #f57f17;border-radius:4px;margin:8px 0;}
.key-box   {background:#e8f5e9;padding:12px 16px;border-left:4px solid #2e7d32;border-radius:4px;margin:8px 0;}
.rmk-box   {background:#f3e5f5;padding:12px 16px;border-left:4px solid #6a1b9a;border-radius:4px;margin:8px 0;}
.ex-box    {background:#fce4ec;padding:12px 16px;border-left:4px solid #880e4f;border-radius:4px;margin:8px 0;}
.prop-box  {background:#e0f7fa;padding:12px 16px;border-left:4px solid #006064;border-radius:4px;margin:8px 0;}
h2 {color:#1a237e;}
h3 {color:#283593;}
h4 {color:#303f9f;}
.stTabs [data-baseweb="tab-list"] {gap:4px;}
.stTabs [data-baseweb="tab"] {padding:6px 14px;font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

def box(tag, content):
    st.markdown(f'<div class="{tag}">{content}</div>', unsafe_allow_html=True)

def defbox(title, content):  box("def-box",  f"<b>Definition — {title}:</b> {content}")
def thmbox(title, content):  box("thm-box",  f"<b>Theorem — {title}:</b> {content}")
def keybox(content):         box("key-box",  f"<b>🔑 Key Idea:</b> {content}")
def rmkbox(content):         box("rmk-box",  f"<b>Remark:</b> {content}")
def exbox(title, content):   box("ex-box",   f"<b>Example — {title}:</b> {content}")
def propbox(title, content): box("prop-box", f"<b>Proposition — {title}:</b> {content}")

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC INDEX  (chapter, section, page, tab label)
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_INDEX = {
    # ── Ch 1 ──────────────────────────────────────────────────────────────────
    "Sample Space": ("Ch 1", "§1", "p.1", "Events & Axioms"),
    "Events / Subsets": ("Ch 1", "§1", "p.1", "Events & Axioms"),
    "De Morgan's Laws": ("Ch 1", "§1", "p.1", "Events & Axioms"),
    "Probability Axioms": ("Ch 1", "§2", "p.2", "Events & Axioms"),
    "Union Bound (Boole)": ("Ch 1", "§2", "p.2", "Events & Axioms"),
    "Inclusion–Exclusion": ("Ch 1", "§2–§3", "p.2", "Events & Axioms"),
    "Conditional Probability": ("Ch 1", "§4", "p.3", "Conditioning & Bayes"),
    "Multiplication Rule": ("Ch 1", "§4", "p.3", "Conditioning & Bayes"),
    "Chain Rule (probability)": ("Ch 1", "§4", "p.3", "Conditioning & Bayes"),
    "Law of Total Probability": ("Ch 1", "§5", "p.4", "Conditioning & Bayes"),
    "Partition (of sample space)": ("Ch 1", "§5", "p.4", "Conditioning & Bayes"),
    "Independence of Events": ("Ch 1", "§6", "p.5", "Conditioning & Bayes"),
    "Mutual Independence": ("Ch 1", "§6", "p.5", "Conditioning & Bayes"),
    "Bayes' Formula": ("Ch 1", "§7", "p.6", "Conditioning & Bayes"),
    "Continuity from Below": ("Ch 1", "§8", "p.6–7", "Continuity of Probability"),
    "Continuity from Above": ("Ch 1", "§8", "p.6", "Continuity of Probability"),
    # ── Ch 2 ──────────────────────────────────────────────────────────────────
    "Random Variable (RV)": ("Ch 2", "§1", "p.1", "RVs & CDFs"),
    "CDF": ("Ch 2", "§1", "p.2", "RVs & CDFs"),
    "PMF": ("Ch 2", "§2", "p.2", "Discrete Distributions"),
    "Bernoulli": ("Ch 2", "§2.1", "p.3", "Discrete Distributions"),
    "Binomial": ("Ch 2", "§2.1", "p.3", "Discrete Distributions"),
    "Geometric": ("Ch 2", "§2.2", "p.4", "Discrete Distributions"),
    "Poisson (distribution)": ("Ch 2", "§2.2", "p.4", "Discrete Distributions"),
    "Poisson Approx. to Binomial": ("Ch 2", "§2.2", "p.5", "Discrete Distributions"),
    "PDF": ("Ch 2", "§3", "p.6", "Continuous Distributions"),
    "Uniform": ("Ch 2", "§3.1", "p.6", "Continuous Distributions"),
    "Exponential (distribution)": ("Ch 2", "§3.1", "p.6", "Continuous Distributions"),
    "Gamma": ("Ch 2", "§3.1", "p.6", "Continuous Distributions"),
    "Normal": ("Ch 2", "§3.1", "p.6", "Continuous Distributions"),
    "Expectation / LOTUS": ("Ch 2", "§4", "p.7", "Expectation & Variance"),
    "Variance": ("Ch 2", "§4", "p.8", "Expectation & Variance"),
    "Covariance": ("Ch 2", "§4", "p.8", "Expectation & Variance"),
    "Joint PMF / PDF": ("Ch 2", "§5", "p.9–10", "Joint Distributions"),
    "Independence of RVs": ("Ch 2", "§5", "p.10", "Joint Distributions"),
    "CDF Method (transformations)": ("Ch 2", "§6.1", "p.12", "MGFs & Transforms"),
    "Convolution": ("Ch 2", "§6.2", "p.13", "MGFs & Transforms"),
    "MGF": ("Ch 2", "§7", "p.14", "MGFs & Transforms"),
    "Markov's Inequality": ("Ch 2", "§8", "p.15", "Inequalities & Limit Thms"),
    "Chebyshev's Inequality": ("Ch 2", "§8", "p.15", "Inequalities & Limit Thms"),
    "SLLN (Strong Law)": ("Ch 2", "§9", "p.16–17", "Inequalities & Limit Thms"),
    "CLT (Central Limit Theorem)": ("Ch 2", "§9", "p.17", "Inequalities & Limit Thms"),
    "Stochastic Process (intro)": ("Ch 2", "§10", "p.17–18", "Inequalities & Limit Thms"),
    # ── Ch 3 ──────────────────────────────────────────────────────────────────
    "Conditional PMF": ("Ch 3", "§2", "p.1", "Conditional Distributions"),
    "Conditional Expectation (discrete)": ("Ch 3", "§2", "p.1", "Conditional Distributions"),
    "Conditional Density": ("Ch 3", "§3", "p.4", "Conditional Distributions"),
    "Conditional Expectation (continuous)": ("Ch 3", "§3", "p.4", "Conditional Distributions"),
    "Tower Property / Law of Total Expectation": ("Ch 3", "§4.1", "p.5", "Tower Property & Random Sums"),
    "Random Sum E[S] = E[N]μ": ("Ch 3", "§4.2", "p.6", "Tower Property & Random Sums"),
    "Restart Trick / First-Step Recursion": ("Ch 3", "§4.3", "p.7", "Tower Property & Random Sums"),
    "Conditional Variance": ("Ch 3", "§5.1", "p.10", "Conditional Variance"),
    "Law of Total Variance": ("Ch 3", "§5.1", "p.10", "Conditional Variance"),
    "E[X] via tail sums": ("Ch 3", "§6", "p.16", "Conditioning on Events"),
    # ── Ch 4 ──────────────────────────────────────────────────────────────────
    "Markov Chain": ("Ch 4", "§2", "p.1", "Definition & Transition Matrix"),
    "Transition Matrix P": ("Ch 4", "§2", "p.1–2", "Definition & Transition Matrix"),
    "Time-Homogeneous MC": ("Ch 4", "§2", "p.1", "Definition & Transition Matrix"),
    "Chapman–Kolmogorov": ("Ch 4", "§4", "p.4", "Multi-Step Transitions"),
    "n-step Transition P^n": ("Ch 4", "§4", "p.4", "Multi-Step Transitions"),
    "Accessibility / Communication": ("Ch 4", "§5", "p.7–8", "Classes & Recurrence"),
    "Communicating Classes": ("Ch 4", "§5", "p.7–8", "Classes & Recurrence"),
    "Irreducible Chain": ("Ch 4", "§5", "p.8", "Classes & Recurrence"),
    "Return Probability f_ii": ("Ch 4", "§6", "p.9", "Classes & Recurrence"),
    "Recurrent State": ("Ch 4", "§6", "p.9", "Classes & Recurrence"),
    "Transient State": ("Ch 4", "§6", "p.9", "Classes & Recurrence"),
    "Mean Return Time m_j": ("Ch 4", "§7.2", "p.14", "Stationary Distributions"),
    "Positive Recurrence": ("Ch 4", "§7.2", "p.14", "Stationary Distributions"),
    "Null Recurrence": ("Ch 4", "§7.2", "p.14–17", "Stationary Distributions"),
    "Long-Run Proportions M_j": ("Ch 4", "§7.3", "p.14", "Stationary Distributions"),
    "Stationary Distribution π": ("Ch 4", "§7.5", "p.17", "Stationary Distributions"),
    "Balance Equations": ("Ch 4", "§7.5", "p.17", "Stationary Distributions"),
    "Ergodic Chain": ("Ch 4", "§7.6", "p.22", "Stationary Distributions"),
    "Periodicity d(i)": ("Ch 4", "§7.6", "p.21", "Stationary Distributions"),
    "Limiting Probabilities lim P^n_ij": ("Ch 4", "§7.6", "p.22", "Stationary Distributions"),
    "Gambler's Ruin Probability": ("Ch 4", "§8.2", "p.23", "Gambler's Ruin"),
    "Mean Absorption Time": ("Ch 4", "§8 / §3", "p.26", "Gambler's Ruin"),
    "Fundamental Matrix (I-P_T)^-1": ("Ch 4", "§9.2", "p.28", "Transient States"),
    "Expected Time in Transient States": ("Ch 4", "§9", "p.27–29", "Transient States"),
    "Branching Process (Galton–Watson)": ("Ch 4", "§10", "p.31", "Branching Processes"),
    "Extinction Probability π": ("Ch 4", "§10.3", "p.33–35", "Branching Processes"),
    "PGF G(s)": ("Ch 4", "§10.3", "p.33", "Branching Processes"),
    "Offspring Mean μ & Variance σ²": ("Ch 4", "§10.2", "p.32", "Branching Processes"),
    "Detailed Balance (DTMC)": ("Ch 4", "§11.1", "p.36–37", "Time Reversibility & MDPs"),
    "Time Reversibility (DTMC)": ("Ch 4", "§11", "p.36", "Time Reversibility & MDPs"),
    "Cycle Criterion": ("Ch 4", "§11.2", "p.40", "Time Reversibility & MDPs"),
    "MDP (Markov Decision Process)": ("Ch 4", "§12", "p.41", "Time Reversibility & MDPs"),
    "Bellman Optimality Equation": ("Ch 4", "§12.3–12.4", "p.43–44", "Time Reversibility & MDPs"),
    "Value Iteration": ("Ch 4", "§12.5", "p.45", "Time Reversibility & MDPs"),
    "Policy Iteration": ("Ch 4", "§12.5", "p.45", "Time Reversibility & MDPs"),
    "Q-function": ("Ch 4", "§12.4", "p.43", "Time Reversibility & MDPs"),
    # ── Ch 5 ──────────────────────────────────────────────────────────────────
    "Exponential Distribution (Exp(λ))": ("Ch 5", "§2.1", "p.1", "Exponential Distribution"),
    "Memoryless Property (exp.)": ("Ch 5", "§2.2", "p.3", "Exponential Distribution"),
    "Racing Exponentials": ("Ch 5", "§2.3", "p.6–8", "Exponential Distribution"),
    "Minimum of Exponentials": ("Ch 5", "§2.3", "p.7", "Exponential Distribution"),
    "Sum of Exponentials = Gamma": ("Ch 5", "§2.3", "p.6", "Exponential Distribution"),
    "Counting Process": ("Ch 5", "§3.1", "p.11", "Poisson Process"),
    "Poisson Process (definition)": ("Ch 5", "§3.2", "p.11–12", "Poisson Process"),
    "Independent Increments": ("Ch 5", "§3.2", "p.11", "Poisson Process"),
    "Stationary Increments": ("Ch 5", "§3.3", "p.15", "Poisson Process"),
    "Interarrival Times ~ Exp(λ)": ("Ch 5", "§3.3", "p.13", "Poisson Process"),
    "N(t) ~ Poisson(λt)": ("Ch 5", "§3.3", "p.14", "Poisson Process"),
    "Thinning / Splitting (Poisson)": ("Ch 5", "§3.4", "p.17", "Splitting & Superposition"),
    "Superposition of Poisson Processes": ("Ch 5", "§3.5", "p.20", "Splitting & Superposition"),
    "Bernoulli Labeling": ("Ch 5", "§3.5", "p.20", "Splitting & Superposition"),
    "Order Statistics Property": ("Ch 5", "§3.5", "p.22", "Splitting & Superposition"),
    "Time-Varying Classification": ("Ch 5", "§3.6", "p.25", "Splitting & Superposition"),
    "Nonhomogeneous Poisson Process": ("Ch 5", "§4.1", "p.30", "Generalizations"),
    "Mean Value Function m(t)": ("Ch 5", "§4.1", "p.30", "Generalizations"),
    "Compound Poisson Process": ("Ch 5", "§4.2", "p.31", "Generalizations"),
    "E[S(t)] and Var[S(t)] (compound)": ("Ch 5", "§4.2", "p.32", "Generalizations"),
    "Conditional / Mixed Poisson": ("Ch 5", "§4.4", "p.35", "Generalizations"),
    "Overdispersion": ("Ch 5", "§4.4", "p.35", "Generalizations"),
    # ── Ch 6 ──────────────────────────────────────────────────────────────────
    "CTMC (definition)": ("Ch 6", "§2", "p.1", "CTMC Basics"),
    "Holding Times ~ Exp(v_i)": ("Ch 6", "§2", "p.2", "CTMC Basics"),
    "Embedded Chain": ("Ch 6", "§6", "p.15", "CTMC Basics"),
    "Birth–Death Process (CTMC)": ("Ch 6", "§3", "p.2", "Birth–Death Processes"),
    "M/M/1 Queue": ("Ch 6", "§3", "p.4", "Birth–Death Processes"),
    "M/M/s Queue": ("Ch 6", "§3", "p.4", "Birth–Death Processes"),
    "Yule Process": ("Ch 6", "§4.1 / §4 Example", "p.7", "Kolmogorov Equations"),
    "Instantaneous Transition Rates q_ij": ("Ch 6", "§4.2", "p.8", "Kolmogorov Equations"),
    "Kolmogorov Backward Equations": ("Ch 6", "§4.2.1", "p.9", "Kolmogorov Equations"),
    "Kolmogorov Forward Equations": ("Ch 6", "§4.2.2", "p.11", "Kolmogorov Equations"),
    "Generator Matrix R": ("Ch 6", "§7", "p.22", "Kolmogorov Equations"),
    "Matrix Exponential e^{Rt}": ("Ch 6", "§7", "p.22", "Kolmogorov Equations"),
    "CTMC Limiting Probabilities P_j": ("Ch 6", "§5", "p.12", "Steady State & Reversibility"),
    "Global Balance Equations (CTMC)": ("Ch 6", "§5.1", "p.12", "Steady State & Reversibility"),
    "Local Balance / Detailed Balance (CTMC)": ("Ch 6", "§5.1", "p.13", "Steady State & Reversibility"),
    "M/M/1 Steady State ρ = λ/μ": ("Ch 6", "§5 Ex 5.2", "p.14", "Steady State & Reversibility"),
    "Time Reversibility (CTMC)": ("Ch 6", "§6", "p.15", "Steady State & Reversibility"),
    "Birth–Death is Time Reversible": ("Ch 6", "§6 Prop 6.1", "p.17", "Steady State & Reversibility"),
    "Truncation of CTMC": ("Ch 6", "§6", "p.19–20", "Steady State & Reversibility"),
    "M/M/1 Departure Process ~ Poisson": ("Ch 6", "§6 Cor 6.1", "p.17", "Steady State & Reversibility"),
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_study_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text, size=14):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, text, fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def subheading(text):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 6, text, ln=True)
        pdf.set_text_color(0, 0, 0)

    def body(text):
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, text)
        pdf.ln(1)

    def formula(text):
        pdf.set_font("Courier", "", 9)
        pdf.set_fill_color(240, 240, 255)
        pdf.multi_cell(0, 5, text, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.ln(1)

    # Cover
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, "ISE 5414 - Random Processes", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Complete Study Notes for AI-Assisted Learning", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Chapters 1-6 (Based on Ross Textbook)", ln=True, align="C")
    pdf.ln(8)
    pdf.set_text_color(0,0,0)

    body("INSTRUCTIONS FOR AI TUTOR: This document contains all key definitions, theorems, "
         "propositions, formulas, and conceptual notes from the ISE 5414 Random Processes course "
         "(Chapters 1-6). Use this to help the student understand concepts, work through examples, "
         "and check their understanding. Always explain the intuition, not just the math. "
         "Reference specific sections (e.g., Ch 4, SS 7.5) when relevant.")
    pdf.ln(4)

    # ── Ch 1
    heading("CHAPTER 1: Probability Theory Refresher (Events)")
    subheading("1. Sample Space and Events")
    body("Experiment: outcome cannot be predicted with certainty. Sample space S = set of all outcomes. "
         "Event E is a subset E ⊆ S. Operations: E^c, E∪F, E∩F, E\\F = E∩F^c.")
    body("De Morgan's Laws: (E∪F)^c = E^c ∩ F^c  |  (E∩F)^c = E^c ∪ F^c")
    body("Mutually exclusive: E∩F = ∅  (null event)")

    subheading("2. Probability Axioms")
    formula("  (a) 0 ≤ P(E) ≤ 1\n  (b) P(S) = 1\n  (c) Countable additivity: P(∪ Ei) = Σ P(Ei) for disjoint Ei")
    body("Basic identities: P(E^c) = 1-P(E), E⊆F => P(E)≤P(F), P(E∪F)=P(E)+P(F)-P(E∩F)")
    body("Union Bound (Boole): P(∪ Ei) ≤ Σ P(Ei)")
    body("Inclusion-Exclusion: P(E1∪...∪En) = Σ P(Ei) - Σ P(Ei∩Ej) + Σ P(Ei∩Ej∩Ek) - ...")

    subheading("4. Conditional Probability")
    formula("  P(E|F) = P(E∩F)/P(F),  P(F)>0\n  Multiplication rule: P(E∩F) = P(E|F)P(F)\n  Chain rule: P(E1∩...∩En) = Π P(Ek|E1∩...∩Ek-1)")

    subheading("5. Law of Total Probability")
    body("Partition B1,...,Bm: mutually exclusive, union = S, P(Bi)>0")
    formula("  P(E) = Σ P(E|Bi)P(Bi)")

    subheading("6. Independence of Events")
    formula("  Independent: P(E∩F) = P(E)P(F)  ↔  P(E|F)=P(E)")
    body("Mutual independence: P(∩_{i∈I} Ei) = Π_{i∈I} P(Ei) for EVERY subset I. "
         "WARNING: pairwise ≠ mutual independence.")

    subheading("7. Bayes' Formula")
    formula("  P(Bj|E) = P(E|Bj)P(Bj) / [Σ P(E|Bi)P(Bi)]")

    subheading("8. Continuity of Probability")
    body("Increasing sequence E1⊆E2⊆...: P(∪En) = lim P(En) (continuity from below)")
    body("Decreasing sequence E1⊇E2⊇...: P(∩En) = lim P(En) (continuity from above)")
    body("WHY: Used for events like {τ≤t} = ∪{τ≤t-1/n} in stopping-time arguments.")
    pdf.ln(3)

    # ── Ch 2
    heading("CHAPTER 2: Random Variables (Review)")
    subheading("1. Random Variables and CDFs")
    body("RV: function X: S → R. CDF: F(b) = P(X≤b), nondecreasing, F(−∞)=0, F(+∞)=1. "
         "P(a<X≤b) = F(b)−F(a).")

    subheading("2. Key Discrete Distributions")
    formula("  Bernoulli(p):   P(X=1)=p, P(X=0)=1-p.  E=p, Var=p(1-p)\n"
            "  Binomial(n,p):  P(X=i)=C(n,i)p^i(1-p)^(n-i).  E=np, Var=np(1-p)\n"
            "  Geometric(p):   P(X=n)=(1-p)^(n-1)p,  n=1,2,...  E=1/p, Var=(1-p)/p^2\n"
            "  Poisson(λ):     P(X=i)=e^(-λ)λ^i/i!.   E=λ, Var=λ\n"
            "  Poisson approx: Binom(n,p) ≈ Poisson(λ=np)  when n large, p small")

    subheading("3. Key Continuous Distributions")
    formula("  Uniform(a,b):   f(x)=1/(b-a).  E=(a+b)/2, Var=(b-a)^2/12\n"
            "  Exponential(λ): f(x)=λe^(-λx), x≥0.  E=1/λ, Var=1/λ^2\n"
            "  Gamma(α,λ):     f(x)=λe^(-λx)(λx)^(α-1)/Γ(α). E=α/λ, Var=α/λ^2\n"
            "  Normal(μ,σ²):   f(x)=exp(-(x-μ)²/2σ²)/(√(2π)σ).  E=μ, Var=σ^2\n"
            "  If X~N(μ,σ²), aX+b ~ N(aμ+b, a²σ²)")

    subheading("4. Expectation, Variance, Covariance")
    formula("  E[X] = Σ x p(x)  or  ∫ x f(x)dx\n"
            "  LOTUS: E[g(X)] = Σ g(x)p(x)  or  ∫ g(x)f(x)dx\n"
            "  Var(X) = E[X²]-(E[X])², Var(aX+b) = a²Var(X)\n"
            "  Cov(X,Y) = E[XY]-E[X]E[Y]\n"
            "  Var(Σ Xi) = Σ Var(Xi) + 2Σ_{i<j} Cov(Xi,Xj)  [independent: only first sum]")

    subheading("5. Joint Distributions & Independence")
    body("X,Y independent iff f(x,y)=fX(x)fY(y) [or p(x,y)=pX(x)pY(y)]. "
         "Then E[g(X)h(Y)] = E[g(X)]E[h(Y)].")

    subheading("6-7. Transforms")
    formula("  CDF method: FY(y) = P(g(X)≤y), differentiate for density.\n"
            "  Convolution: fZ(z) = ∫ fX(x) fY(z-x) dx  for Z=X+Y independent.\n"
            "  MGF: MX(t) = E[e^(tX)].  M^(n)_X(0) = E[X^n].  X,Y indep => M_{X+Y}=M_X·M_Y")

    subheading("8-9. Inequalities & Limit Theorems")
    formula("  Markov:    P(X≥a) ≤ E[X]/a  (X≥0)\n"
            "  Chebyshev: P(|X-μ|≥a) ≤ σ²/a²\n"
            "  SLLN:      X̄_n → μ  a.s.  (i.i.d., finite mean)\n"
            "  CLT:       (Σ Xi - nμ)/(σ√n) → N(0,1)  in distribution")

    subheading("10. Stochastic Processes")
    body("Stochastic process {X(t): t∈T}: collection of RVs indexed by T. "
         "T countable → discrete time; T=[0,∞) → continuous time. "
         "State space: set of values X(t) can take.")
    pdf.ln(3)

    # ── Ch 3
    heading("CHAPTER 3: Conditional Probability and Conditional Expectation")
    subheading("Conditional PMF / Density")
    formula("  Discrete: p_{X|Y}(x|y) = P(X=x,Y=y)/P(Y=y)\n"
            "  Continuous: f_{X|Y}(x|y) = f_{X,Y}(x,y)/fY(y)\n"
            "  E[X|Y=y] = Σ x P(X=x|Y=y)  or  ∫ x f_{X|Y}(x|y)dx")
    body("KEY: E[X|Y=y] is a NUMBER (y fixed). E[X|Y] is a RANDOM VARIABLE (function of Y).")

    subheading("Tower Property (Law of Total Expectation)")
    formula("  E[X] = E[E[X|Y]] = Σ_y E[X|Y=y]P(Y=y)  (discrete Y)")
    body("Strategy: pick Y so E[X|Y] is easy, then average over Y. "
         "Good Y choices: first step, first trial, or a count that simplifies the problem.")

    subheading("Random Sums")
    formula("  S = X1+...+XN (N independent of i.i.d. Xi with mean μ)\n"
            "  E[S] = E[N]·μ")

    subheading("Law of Total Variance")
    formula("  Var(X) = E[Var(X|Y)] + Var(E[X|Y])\n"
            "    = [avg residual uncertainty after Y] + [uncertainty from randomness of Y]")

    subheading("E[X] via Tail Sums")
    formula("  X nonneg integer: E[X] = Σ_{n=0}^∞ P(X>n)\n"
            "  X nonneg continuous: E[X] = ∫_0^∞ P(X>t) dt")
    pdf.ln(3)

    # ── Ch 4
    heading("CHAPTER 4: Markov Chains")
    subheading("Definition")
    formula("  Markov property: P(Xn+1=j|Xn=i, Xn-1,...,X0) = Pij  (time-homogeneous)\n"
            "  Transition matrix P: Pij≥0, Σj Pij=1 for each i")
    body("Fully specified by: (1) state space S, (2) transition matrix P, (3) initial distribution.")

    subheading("Multi-step Transitions & C-K")
    formula("  P^(n) = P^n  (matrix power)\n"
            "  Chapman-Kolmogorov: P^(n+m) = P^(n)·P^(m)")

    subheading("Classes & Recurrence/Transience")
    body("i→j (accessible): ∃n, P^(n)_ij > 0. i↔j (communicate). "
         "Irreducible: all states communicate.")
    formula("  f_ii = P(return to i | X0=i)\n"
            "  Recurrent: f_ii=1  [equivalently Σ P^(n)_ii = ∞]\n"
            "  Transient: f_ii<1  [equivalently Σ P^(n)_ii < ∞]\n"
            "  Recurrence/transience are CLASS properties.")
    body("Finite state irreducible chain → ALL states recurrent.")

    subheading("Stationary Distributions & Long-Run Behavior")
    formula("  π = πP, π≥0, Σπ_j=1  (balance equations)\n"
            "  Long-run proportion in j: Mj = 1/mj  (mj = mean return time to j)\n"
            "  Positive recurrent: mj < ∞  ↔  Mj > 0\n"
            "  Irreducible + pos.recurrent + aperiodic (ergodic): lim P^(n)_ij = πj")
    body("Periodicity d(i) = gcd{n: P^(n)_ii > 0}. Aperiodic: d(i)=1. "
         "Periodic chain: long-run fractions exist but P^(n)_ij may NOT converge.")

    subheading("Gambler's Ruin (fortune {0,...,N}, p=prob of +1)")
    formula("  Pi = P(hit N before 0 | X0=i)\n"
            "  P0=0, PN=1\n"
            "  pi≠1/2: Pi = [1-(q/p)^i] / [1-(q/p)^N]\n"
            "  p=1/2:   Pi = i/N\n"
            "  Mean absorption time (p=1/2): E[T|X0=i] = i(N-i)")

    subheading("Transient States: Fundamental Matrix")
    formula("  PT = transition sub-matrix restricted to transient states T\n"
            "  S = (I - PT)^(-1)   [fundamental matrix]\n"
            "  sij = E[time in j before absorption | X0=i]\n"
            "  fij = (sij - δij) / sjj  [prob of ever visiting j | start i]")

    subheading("Branching Processes (Galton-Watson)")
    formula("  Xn+1 = Σ_{r=1}^{Xn} Zn,r  (i.i.d. offspring)\n"
            "  μ = E[Z], σ² = Var(Z)\n"
            "  E[Xn] = μ^n  (X0=1)\n"
            "  Var(Xn) = σ²μ^(n-1)(1-μ^n)/(1-μ)  if μ≠1,  else nσ²\n"
            "  G(s) = E[s^Z] = Σ pj s^j  (PGF)\n"
            "  Extinction probability π = G(π) [smallest fixed point in [0,1]]\n"
            "  μ≤1 ⟹ π=1;  μ>1 ⟹ π∈(0,1)")

    subheading("Time Reversibility")
    formula("  Reversed chain: Qij = πj Pji / πi\n"
            "  Time reversible iff detailed balance: πi Pij = πj Pji  for all i,j\n"
            "  Cycle criterion: reversible iff every cycle = its reverse in probability")

    subheading("MDPs")
    body("MDP = (S, {A(s)}, P(·|s,a), r(s,a)). Policy π(a|s). "
         "Fixing π yields a Markov chain with P^π(s,s')=Σ_a π(a|s)P(s'|s,a).")
    formula("  Finite horizon Bellman: V*_t(s) = max_a[r(s,a) + Σ_{s'} P(s'|s,a)V*_{t+1}(s')]\n"
            "  Discounted (γ∈(0,1)): V*(s) = max_a[r(s,a) + γ Σ_{s'} P(s'|s,a)V*(s')]\n"
            "  Value iteration: Vk+1 = T·Vk  →  V*\n"
            "  Policy iteration: eval π then improve greedily (finite convergence)")
    pdf.ln(3)

    # ── Ch 5
    heading("CHAPTER 5: Exponential Distribution and Poisson Process")
    subheading("Exponential Distribution Exp(λ)")
    formula("  f(x)=λe^(-λx), F(x)=1-e^(-λx), x≥0\n"
            "  E[X]=1/λ, Var(X)=1/λ², MGF=λ/(λ-t), t<λ\n"
            "  Memoryless: P(X>s+t|X>s)=P(X>t)  ← UNIQUE continuous memoryless dist!")
    body("Intuition: λ is the rate (events/time). Larger λ = shorter expected wait.")

    subheading("Racing Exponentials")
    formula("  X1~Exp(λ1),...,Xn~Exp(λn) independent:\n"
            "  M = min Xi ~ Exp(λ1+...+λn)\n"
            "  P(Xi = M) = λi / (Σ λj)\n"
            "  M is INDEPENDENT of the rank ordering Π=(i1,...,in)")

    subheading("Poisson Process (rate λ)")
    formula("  Counting process {N(t)} with:\n"
            "  1. Independent increments\n"
            "  2. P(N(t+h)-N(t)=1) = λh + o(h);  P(≥2 events) = o(h)\n"
            "  => N(t) ~ Poisson(λt)  [E=λt, Var=λt]\n"
            "  => Interarrival times T1,T2,... i.i.d. ~ Exp(λ)")
    body("Two views: (counting) N(t)~Poisson(λt) with indep. increments; "
         "(waiting-time) interarrivals i.i.d. Exp(λ). Use whichever is easier.")

    subheading("Splitting and Superposition")
    formula("  Thinning: label each event type I w.p. p, type II w.p. 1-p\n"
            "    => N1~PP(λp), N2~PP(λ(1-p)), independent\n"
            "  Superposition: N(1)+N(2) ~ PP(λ1+λ2)\n"
            "  Bernoulli labeling: given N(t)=n, N1(t)|N(t)=n ~ Binom(n,p), p=λ1/(λ1+λ2)\n"
            "  Order stats: given N(t)=n, arrivals (S1,...,Sn) ~ order stats of n i.i.d. Unif(0,t)")

    subheading("Nonhomogeneous Poisson Process")
    formula("  Rate λ(t) (time-varying). m(t) = ∫_0^t λ(s)ds  (mean value function)\n"
            "  N(t) ~ Poisson(m(t)).  Increments INDEPENDENT but NOT stationary.")

    subheading("Compound Poisson Process")
    formula("  S(t) = Σ_{i=1}^{N(t)} Xi,  Xi i.i.d., indep of N(t)~PP(λ)\n"
            "  E[S(t)] = λt E[X1]\n"
            "  Var(S(t)) = λt E[X1²]  (= λt·E[X²], not λt·Var(X)!)\n"
            "  CLT: S(t) ≈ N(λt E[Y], λt E[Y²])  for large t")
    pdf.ln(3)

    # ── Ch 6
    heading("CHAPTER 6: Continuous-Time Markov Chains (CTMCs)")
    subheading("Definition")
    body("CTMC {X(t),t≥0}: Markov property in continuous time. "
         "Key consequence of time-homogeneity + Markov property: "
         "holding time in state i must be memoryless → Ti ~ Exp(vi). "
         "So the chain is defined by: (i) rates vi (sojourn rates), (ii) jump matrix Pij.")

    subheading("Instantaneous Transition Rates")
    formula("  qij = vi · Pij  (i≠j),  vi = Σ_{j≠i} qij\n"
            "  Small-time: P(leave i in (t,t+h]) ≈ vi·h\n"
            "              P(go i→j in (t,t+h]) ≈ qij·h")

    subheading("Birth–Death Processes")
    formula("  Rates: λn (birth n→n+1), μn (death n→n-1), μ0=0\n"
            "  vn = λn+μn,  Pn,n+1 = λn/(λn+μn)\n"
            "  M/M/1: λn=λ (all n), μn=μ (n≥1)\n"
            "  M/M/s: λn=λ (all n), μn=nμ (1≤n≤s), sμ (n>s)")

    subheading("Kolmogorov Equations")
    formula("  Backward: P'ij(t) = -vi Pij(t) + Σ_{k≠i} qik Pkj(t)\n"
            "  Forward:  P'ij(t) = -vj Pij(t) + Σ_{k≠j} qkj Pik(t)\n"
            "  Matrix form: P'(t) = R·P(t) = P(t)·R  where R_ij = qij (i≠j), R_ii = -vi\n"
            "  Solution: P(t) = e^(Rt) = Σ_{n=0}^∞ (Rt)^n/n!")

    subheading("Limiting Probabilities & Balance Equations")
    formula("  Global balance: vj Pj = Σ_{k≠j} Pk qkj  (rate out = rate in)\n"
            "  Birth-death local balance: λn Pn = μ_{n+1} P_{n+1}\n"
            "  => Pn = (λ0...λ_{n-1})/(μ1...μn) · P0\n"
            "  P0 = [1 + Σ_{n=1}^∞ (λ0...λ_{n-1})/(μ1...μn)]^(-1)\n"
            "  M/M/1: Pn = (1-ρ)ρ^n,  ρ=λ/μ<1")

    subheading("Time Reversibility (CTMC)")
    formula("  Reversed chain rates: qij reversed = Pj qji / Pi\n"
            "  Time reversible iff DETAILED BALANCE: Pi qij = Pj qji  for all i≠j\n"
            "  Every ergodic birth-death process is time reversible.\n"
            "  Truncation: if time-reversible chain truncated to A (irreducible on A),\n"
            "    then P^A_j = Pj / (Σ_{i∈A} Pi)  for j∈A")

    pdf.ln(4)
    body("─" * 80)
    body("END OF STUDY NOTES — ISE 5414 Random Processes (Chapters 1–6)")

    buf = BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# ====================== CHAPTER PAGES ========================================
# ─────────────────────────────────────────────────────────────────────────────

def chapter1():
    st.header("Chapter 1 — Probability Theory Refresher (Events)")
    tabs = st.tabs(["Events & Axioms", "Conditioning & Bayes", "Continuity of Probability", "🎯 Interactive"])

    # ── Tab 0 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("§1  Sample Space and Events")
        defbox("Experiment / Sample Space / Event",
               "An <b>experiment</b> produces an outcome that cannot be predicted with certainty. "
               "The <b>sample space S</b> is the set of all possible outcomes. "
               "An <b>event</b> is any subset E ⊆ S.")
        st.write("**Event operations:**")
        st.latex(r"E^c = S \setminus E,\quad E\cup F,\quad E\cap F,\quad E\setminus F = E\cap F^c")
        propbox("De Morgan's Laws",
                "(E∪F)ᶜ = Eᶜ ∩ Fᶜ &emsp;|&emsp; (E∩F)ᶜ = Eᶜ ∪ Fᶜ")
        keybox("Events like {X<sub>t</sub>∈A}, {τ≤t}, and {sup<sub>s≤t</sub> X<sub>s</sub>≥a} "
               "are built from unions, intersections, and complements — so these laws are used constantly in random processes.")

        st.subheader("§2  Probability Axioms")
        defbox("Probability Measure P(·)",
               "(a) 0 ≤ P(E) ≤ 1 for all events E &nbsp;|&nbsp; "
               "(b) P(S) = 1 &nbsp;|&nbsp; "
               "(c) Countable additivity: pairwise disjoint E₁,E₂,… ⟹ P(∪Eᵢ) = ΣP(Eᵢ)")
        propbox("Basic Identities",
                "P(Eᶜ) = 1−P(E)  |  E⊆F ⟹ P(E)≤P(F)  |  P(E∪F) = P(E)+P(F)−P(E∩F)")
        propbox("Union Bound / Boole", "P(∪ Eᵢ) ≤ Σ P(Eᵢ)")
        propbox("Inclusion–Exclusion",
                "P(E₁∪…∪Eₙ) = ΣP(Eᵢ) − ΣP(Eᵢ∩Eⱼ) + ΣP(Eᵢ∩Eⱼ∩Eₖ) − ⋯")
        exbox("Equally Likely Outcomes",
              "If S is finite and symmetric: P(E) = |E|/|S|. "
              "<b>Do not assume equal-likelihood unless symmetry is clear.</b>")

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("§4  Conditional Probability")
        defbox("Conditional Probability",
               "P(E|F) = P(E∩F)/P(F), when P(F)>0. "
               "Multiplication rule: P(E∩F) = P(E|F)P(F). "
               "Chain rule: P(E₁∩⋯∩Eₙ) = ∏ P(Eₖ | E₁∩⋯∩Eₖ₋₁)")

        st.subheader("§5  Law of Total Probability")
        defbox("Partition", "B₁,…,Bₘ partition S if: mutually exclusive, ∪Bᵢ=S, P(Bᵢ)>0.")
        thmbox("Total Probability", "P(E) = Σᵢ P(E|Bᵢ)P(Bᵢ)")
        keybox("Strategy: pick partition so P(E|Bᵢ) is easy, then average over i.")

        st.subheader("§6  Independence")
        defbox("Independence", "E and F are independent if P(E∩F) = P(E)P(F), "
               "equivalently P(E|F)=P(E) when P(F)>0.")
        rmkbox("<b>WARNING — Mutual ≠ Pairwise independence.</b> "
               "Collection {E₁,…,Eₙ} is mutually independent if "
               "P(∩ᵢ∈ᵢ Eᵢ) = ∏ᵢ∈ᵢ P(Eᵢ) for EVERY nonempty I⊆{1,…,n}.")

        st.subheader("§7  Bayes' Formula")
        thmbox("Bayes",
               "P(Bⱼ|E) = P(E|Bⱼ)P(Bⱼ) / [Σᵢ P(E|Bᵢ)P(Bᵢ)]")
        exbox("Screening Test",
              "Disease prevalence 1%. Sensitivity (P(+|D))=99%, Specificity (P(−|Dᶜ))=95%. "
              "P(D|+) = (0.99×0.01)/[(0.99×0.01)+(0.05×0.99)] ≈ 16.7%. "
              "Most positive tests are FALSE POSITIVES when prevalence is low.")

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("§8  Continuity of Probability")
        thmbox("Continuity from Below",
               "E₁⊆E₂⊆⋯ (increasing): P(∪Eₙ) = lim P(Eₙ)")
        thmbox("Continuity from Above",
               "E₁⊇E₂⊇⋯ (decreasing): P(∩Eₙ) = lim P(Eₙ)")
        st.markdown("**Proof sketch (from below):** Define increments F₁=E₁, Fₙ=Eₙ\\Eₙ₋₁. "
                    "These are pairwise disjoint and ∪Fₖ=∪Eₙ. Apply countable additivity.")
        rmkbox("These facts justify limits in events like {τ≤t} = ∪ₙ{τ≤t−1/n}, "
               "appearing in stopping-time arguments throughout random processes.")

        st.subheader("§9  Derangements (Continuity Example)")
        st.write("Let E = {no one gets their own hat} when n people permute hats uniformly.")
        col1, col2 = st.columns(2)
        with col1:
            ns = list(range(1, 20))
            probs = []
            for n in ns:
                # inclusion-exclusion: P(E) = Σ(-1)^k / k!
                p = sum((-1)**k / math.factorial(k) for k in range(n+1))
                probs.append(max(0, p))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ns, y=probs, mode='lines+markers', name='P(no match)',
                                     line=dict(color='steelblue', width=2)))
            fig.add_hline(y=1/math.e, line_dash='dash', line_color='red',
                          annotation_text=f"1/e ≈ {1/math.e:.4f}")
            fig.update_layout(title="P(derangement) vs n → 1/e", xaxis_title="n", yaxis_title="Probability",
                               height=320, margin=dict(t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Formula (Inclusion–Exclusion):**")
            st.latex(r"P(E_n) = \sum_{k=0}^{n} \frac{(-1)^k}{k!} \xrightarrow{n\to\infty} \frac{1}{e}")
            st.write(f"Limit as n→∞: P(E) → 1/e ≈ {1/math.e:.5f}")
            st.write("The probability quickly converges to 1/e regardless of how many people!")

    # ── Tab 3 Interactive ──────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🎯 Bayes' Theorem Calculator")
        col1, col2 = st.columns(2)
        with col1:
            prevalence = st.slider("Prior P(D) — prevalence", 0.001, 0.5, 0.01, 0.001, format="%.3f")
            sensitivity = st.slider("Sensitivity P(+|D)", 0.5, 1.0, 0.99, 0.01)
            specificity  = st.slider("Specificity P(−|Dᶜ)", 0.5, 1.0, 0.95, 0.01)
        with col2:
            fp_rate = 1 - specificity
            tp = sensitivity * prevalence
            fp = fp_rate * (1 - prevalence)
            ppv = tp / (tp + fp) if (tp+fp) > 0 else 0
            npv_n = specificity * (1-prevalence)
            npv_d = npv_n + (1-sensitivity)*prevalence
            npv = npv_n/npv_d if npv_d>0 else 0
            st.metric("PPV: P(D | test+)", f"{ppv:.1%}")
            st.metric("NPV: P(Dᶜ | test−)", f"{npv:.1%}")
            st.metric("P(+) overall", f"{tp+fp:.3%}")
        # Probability tree visualization
        fig2 = go.Figure()
        # nodes: root (0,1), D (−0.5,0.5), Dc (0.5,0.5), D+ (−0.8,0), D− (−0.2,0), Dc+ (0.2,0), Dc− (0.8,0)
        xs = [0, -0.5, 0.5, -0.75, -0.25, 0.25, 0.75]
        ys = [1.0, 0.55, 0.55, 0.1, 0.1, 0.1, 0.1]
        labels = ["Start",
                  f"D\nP={prevalence:.3f}", f"Dᶜ\nP={1-prevalence:.3f}",
                  f"D∩+\n{tp:.4f}", f"D∩−\n{(1-sensitivity)*prevalence:.4f}",
                  f"Dᶜ∩+\n{fp:.4f}", f"Dᶜ∩−\n{npv_n:.4f}"]
        fig2.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', text=labels,
                                   textposition='bottom center',
                                   marker=dict(size=20, color=['gray','crimson','steelblue',
                                                                'darkred','pink','navy','lightblue']),
                                   textfont=dict(size=10)))
        edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
        for a,b in edges:
            fig2.add_shape(type='line', x0=xs[a],y0=ys[a],x1=xs[b],y1=ys[b],
                            line=dict(color='gray',width=1.5))
        fig2.update_layout(height=360, xaxis=dict(visible=False), yaxis=dict(visible=False),
                            title="Probability Tree", margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)
        keybox(f"Even with high sensitivity & specificity, PPV ≈ {ppv:.1%} when prevalence is only {prevalence:.1%}. "
               "This is the <b>base rate fallacy</b> — always account for prior probability!")


def chapter2():
    st.header("Chapter 2 — Random Variables (Review)")
    tabs = st.tabs(["RVs & CDFs", "Discrete Distributions", "Continuous Distributions",
                    "Expectation & Variance", "Joint Distributions", "MGFs & Transforms",
                    "Inequalities & Limit Thms"])

    # ── Tab 0 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("§1 Random Variables and CDFs")
        defbox("Random Variable", "A function X: S → ℝ from the sample space to the real numbers.")
        defbox("CDF", "F(b) = P(X ≤ b), −∞ < b < ∞. Properties: nondecreasing, F(−∞)=0, F(+∞)=1. "
               "P(a < X ≤ b) = F(b) − F(a).")
        keybox("A stochastic process {X(t)}_{t≥0} is a family of RVs indexed by time. "
               "Mastering single RVs is the foundation for processes.")
        rmkbox("For continuous X: P(X=a)=0 for every fixed a, so P(a&lt;X&lt;b) = P(a≤X≤b).")

    # ── Tab 1 (Discrete) ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("§2 Discrete Random Variables — Key Families")
        dist_choice = st.selectbox("Select distribution", ["Binomial(n,p)", "Geometric(p)", "Poisson(λ)"])
        col1, col2 = st.columns([1, 2])
        with col1:
            if dist_choice == "Binomial(n,p)":
                n_ = st.slider("n", 1, 50, 10)
                p_ = st.slider("p", 0.01, 0.99, 0.3, 0.01)
                xs = np.arange(0, n_+1)
                pmf = sps.binom.pmf(xs, n_, p_)
                cdf = sps.binom.cdf(xs, n_, p_)
                st.markdown(f"**PMF:** P(X=k) = C(n,k)pᵏ(1−p)ⁿ⁻ᵏ")
                st.metric("Mean", f"{n_*p_:.3f}"); st.metric("Variance", f"{n_*p_*(1-p_):.3f}")
                defbox("Binomial(n,p)", f"X = #successes in {n_} i.i.d. Bernoulli({p_}) trials.")
            elif dist_choice == "Geometric(p)":
                p_ = st.slider("p", 0.05, 0.95, 0.3, 0.05)
                n_ = 30
                xs = np.arange(1, n_+1)
                pmf = sps.geom.pmf(xs, p_)
                cdf = sps.geom.cdf(xs, p_)
                st.metric("Mean E[X]", f"{1/p_:.3f}"); st.metric("Variance", f"{(1-p_)/p_**2:.3f}")
                defbox("Geometric(p)", f"X = #trials until first success. P(X=n) = (1−p)ⁿ⁻¹p.")
                rmkbox("Geometric is the DISCRETE memoryless distribution: P(X>m+n|X>m) = P(X>n).")
            else:  # Poisson
                lam = st.slider("λ", 0.1, 20.0, 3.0, 0.1)
                n_ = max(30, int(lam*3)+5)
                xs = np.arange(0, n_)
                pmf = sps.poisson.pmf(xs, lam)
                cdf = sps.poisson.cdf(xs, lam)
                st.metric("Mean = Var", f"{lam:.2f}")
                defbox("Poisson(λ)", f"P(X=i)=e⁻λλⁱ/i!, i=0,1,2,…")
                rmkbox("Poisson approximation to Binomial: Binom(n,p) ≈ Poisson(np) when n large, p small.")
        with col2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["PMF", "CDF"])
            fig.add_trace(go.Bar(x=xs, y=pmf, name="PMF", marker_color='steelblue'), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=cdf, mode='lines+markers', name="CDF",
                                      line=dict(color='crimson', shape='hv')), row=1, col=2)
            fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Distribution table
        st.subheader("Summary Table — Discrete Distributions")
        df = pd.DataFrame({
            "Distribution": ["Bernoulli(p)", "Binomial(n,p)", "Geometric(p)", "Poisson(λ)"],
            "Support": ["0,1", "0,1,…,n", "1,2,…", "0,1,2,…"],
            "PMF p(x)": ["p if x=1; 1−p if x=0", "C(n,x)pˣ(1−p)ⁿ⁻ˣ", "p(1−p)ˣ⁻¹", "e⁻λλˣ/x!"],
            "Mean": ["p", "np", "1/p", "λ"],
            "Variance": ["p(1−p)", "np(1−p)", "(1−p)/p²", "λ"],
            "MGF MX(t)": ["1−p+peᵗ", "(1−p+peᵗ)ⁿ", "peᵗ/(1−(1−p)eᵗ)", "exp{λ(eᵗ−1)}"]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 2 (Continuous) ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("§3 Continuous Random Variables — Key Families")
        dist_choice = st.selectbox("Select distribution ",
                                    ["Normal(μ,σ²)", "Exponential(λ)", "Gamma(α,λ)", "Uniform(a,b)"])
        col1, col2 = st.columns([1, 2])
        with col1:
            x_range = None
            if dist_choice == "Normal(μ,σ²)":
                mu = st.slider("μ (mean)", -5.0, 5.0, 0.0, 0.1)
                sig = st.slider("σ (std dev)", 0.1, 5.0, 1.0, 0.1)
                xs = np.linspace(mu-4*sig, mu+4*sig, 400)
                pdf = sps.norm.pdf(xs, mu, sig)
                cdf_v = sps.norm.cdf(xs, mu, sig)
                st.metric("Mean", f"{mu:.2f}"); st.metric("Variance", f"{sig**2:.2f}")
                defbox("Normal(μ,σ²)", "f(x) = exp(−(x−μ)²/2σ²)/(√(2π)σ)")
                rmkbox("If X~N(μ,σ²), then aX+b ~ N(aμ+b, a²σ²).")
            elif dist_choice == "Exponential(λ)":
                lam = st.slider("λ (rate)", 0.1, 5.0, 1.0, 0.1)
                xs = np.linspace(0, 6/lam, 400)
                pdf = sps.expon.pdf(xs, scale=1/lam)
                cdf_v = sps.expon.cdf(xs, scale=1/lam)
                st.metric("Mean 1/λ", f"{1/lam:.3f}"); st.metric("Variance 1/λ²", f"{1/lam**2:.3f}")
                defbox("Exponential(λ)", "f(x) = λe⁻λˣ, x≥0. CDF: F(x) = 1−e⁻λˣ")
                rmkbox("Memoryless: P(X>s+t|X>s) = P(X>t) — unique continuous memoryless distribution!")
            elif dist_choice == "Gamma(α,λ)":
                alpha = st.slider("α (shape)", 0.5, 10.0, 2.0, 0.5)
                lam = st.slider("λ (rate) ", 0.1, 5.0, 1.0, 0.1)
                xs = np.linspace(0, (alpha+3*np.sqrt(alpha))/lam, 400)
                pdf = sps.gamma.pdf(xs, alpha, scale=1/lam)
                cdf_v = sps.gamma.cdf(xs, alpha, scale=1/lam)
                st.metric("Mean α/λ", f"{alpha/lam:.3f}"); st.metric("Variance α/λ²", f"{alpha/lam**2:.3f}")
                defbox("Gamma(α,λ)", "f(x)=λe⁻λˣ(λx)^(α−1)/Γ(α), x≥0. Sum of α i.i.d. Exp(λ) if α∈ℤ.")
            else:  # Uniform
                a = st.slider("a (lower)", -5.0, 4.9, 0.0, 0.1)
                b = st.slider("b (upper)", a+0.1, 10.0, 1.0, 0.1)
                xs = np.linspace(a-0.5, b+0.5, 400)
                pdf = sps.uniform.pdf(xs, a, b-a)
                cdf_v = sps.uniform.cdf(xs, a, b-a)
                st.metric("Mean", f"{(a+b)/2:.3f}"); st.metric("Variance", f"{(b-a)**2/12:.3f}")
                defbox("Uniform(a,b)", "f(x) = 1/(b−a) for a<x<b, else 0.")
        with col2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["PDF f(x)", "CDF F(x)"])
            fig.add_trace(go.Scatter(x=xs, y=pdf, fill='tozeroy', name="PDF",
                                      line=dict(color='steelblue', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=cdf_v, name="CDF",
                                      line=dict(color='crimson', width=2)), row=1, col=2)
            fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Table — Continuous Distributions")
        df2 = pd.DataFrame({
            "Distribution": ["Uniform(a,b)", "Exponential(λ)", "Gamma(α,λ)", "Normal(μ,σ²)"],
            "Support": ["(a,b)", "[0,∞)", "[0,∞)", "ℝ"],
            "PDF f(x)": ["1/(b−a)", "λe⁻λˣ", "λe⁻λˣ(λx)^(α−1)/Γ(α)", "exp(−(x−μ)²/2σ²)/(σ√2π)"],
            "Mean": ["(a+b)/2", "1/λ", "α/λ", "μ"],
            "Variance": ["(b−a)²/12", "1/λ²", "α/λ²", "σ²"],
            "MGF": ["(eᵗᵇ−eᵗᵃ)/(t(b−a))", "λ/(λ−t), t<λ", "(λ/(λ−t))ⁿ", "exp(μt+σ²t²/2)"]
        })
        st.dataframe(df2, use_container_width=True, hide_index=True)

    # ── Tab 3 (E & Var) ───────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("§4 Expectation, Variance, Covariance")
        defbox("Expectation",
               "Discrete: E[X]=Σx·p(x). Continuous: E[X]=∫x·f(x)dx.")
        propbox("LOTUS (Law of the Unconscious Statistician)",
                "E[g(X)] = Σg(x)p(x) [discrete] or ∫g(x)f(x)dx [continuous]. "
                "No need to find the distribution of g(X) first!")
        propbox("Linearity", "E[aX+b] = aE[X]+b. More generally: E[Σcᵢ Xᵢ] = ΣcᵢE[Xᵢ].")
        defbox("Variance", "Var(X) = E[(X−E[X])²] = E[X²]−(E[X])². Var(aX+b) = a²Var(X).")
        defbox("Covariance", "Cov(X,Y) = E[XY]−E[X]E[Y]. Properties: Cov(X,X)=Var(X), "
               "Cov(X,Y+Z)=Cov(X,Y)+Cov(X,Z), Cov(aX,Y)=aCov(X,Y).")
        propbox("Variance of a Sum",
                "Var(ΣXᵢ) = ΣVar(Xᵢ) + 2Σ_{i<j} Cov(Xᵢ,Xⱼ). If independent: Var(ΣXᵢ)=ΣVar(Xᵢ).")

    # ── Tab 4 (Joint) ─────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("§5 Joint Distributions and Independence")
        defbox("Joint CDF", "F_{X,Y}(x,y) = P(X≤x, Y≤y).")
        defbox("Marginals (continuous)",
               "f_X(x) = ∫f_{X,Y}(x,y)dy. f_Y(y) = ∫f_{X,Y}(x,y)dx.")
        defbox("Independence of RVs",
               "X,Y independent iff P(X∈A,Y∈B)=P(X∈A)P(Y∈B) for all sets A,B. "
               "Sufficient check: f(x,y) = f_X(x)f_Y(y).")
        propbox("Independent ⟹ Factorization",
                "If X,Y independent: E[g(X)h(Y)] = E[g(X)]·E[h(Y)].")
        st.subheader("Visual: Bivariate Normal (Correlated vs Independent)")
        rho = st.slider("Correlation ρ", -0.95, 0.95, 0.0, 0.05)
        cov_mat = [[1, rho], [rho, 1]]
        np.random.seed(42)
        samples = np.random.multivariate_normal([0,0], cov_mat, 600)
        fig_j = go.Figure()
        fig_j.add_trace(go.Scatter(x=samples[:,0], y=samples[:,1], mode='markers',
                                    marker=dict(size=3, color='steelblue', opacity=0.5)))
        fig_j.update_layout(title=f"Joint Normal, ρ={rho}", xaxis_title="X", yaxis_title="Y",
                             height=320, margin=dict(t=40))
        st.plotly_chart(fig_j, use_container_width=True)
        if abs(rho) < 0.05:
            st.success("ρ≈0: X and Y appear independent (for normal RVs, uncorrelated = independent!).")
        else:
            st.info(f"ρ={rho}: X and Y are correlated (and hence dependent for normal RVs).")

    # ── Tab 5 (MGF) ───────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("§7 Moment Generating Functions")
        defbox("MGF", "M_X(t) = E[eᵗˣ], for t where the expectation exists.")
        propbox("Properties",
                "1. Mˡ(n)_X(0) = E[Xⁿ] (n-th derivative at 0 = n-th moment). "
                "2. X,Y independent ⟹ M_{X+Y}(t) = M_X(t)·M_Y(t). "
                "3. Matching MGFs ⟹ same distribution.")
        st.subheader("§6 CDF Method (Transformations)")
        keybox("To find distribution of Y=g(X): compute F_Y(y)=P(g(X)≤y) from known F_X, then differentiate for density.")
        st.subheader("§6.2 Convolution")
        defbox("Convolution (sum of independent)",
               "Z=X+Y indep.: f_Z(z)=∫f_X(x)f_Y(z−x)dx [continuous]. "
               "P(X+Y=n) = Σ_k P(X=k)P(Y=n−k) [discrete].")
        exbox("Poisson additivity",
              "X~Poisson(λ), Y~Poisson(μ) independent ⟹ X+Y~Poisson(λ+μ).")

    # ── Tab 6 (Inequalities) ──────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("§8 Markov's and Chebyshev's Inequalities")
        propbox("Markov's Inequality", "X≥0, E[X]<∞: P(X≥a) ≤ E[X]/a for any a>0.")
        propbox("Chebyshev's Inequality", "P(|X−μ|≥a) ≤ σ²/a² for any a>0.")
        keybox("These convert moment information (mean/variance) into probability bounds. "
               "They apply to partial sums, hitting times, and time averages of stochastic processes.")

        st.subheader("§9 Limit Theorems")
        thmbox("SLLN (Strong LLN)", "X₁,X₂,… i.i.d. with finite mean μ: X̄ₙ = (1/n)ΣXᵢ → μ a.s.")
        thmbox("CLT", "(ΣXᵢ−nμ)/(σ√n) ⟹ N(0,1) in distribution.")

        st.subheader("🎯 CLT Demonstration")
        col1, col2 = st.columns([1, 2])
        with col1:
            src_dist = st.selectbox("Source distribution", ["Exponential(1)", "Uniform(0,1)", "Bernoulli(0.3)"])
            n_clt = st.slider("Sample size n", 1, 200, 30)
            n_trials = 3000
        with col2:
            np.random.seed(0)
            if src_dist == "Exponential(1)":
                raw = np.random.exponential(1, (n_trials, n_clt))
                mu_true, sig_true = 1.0, 1.0
            elif src_dist == "Uniform(0,1)":
                raw = np.random.uniform(0, 1, (n_trials, n_clt))
                mu_true, sig_true = 0.5, 1/np.sqrt(12)
            else:
                raw = np.random.binomial(1, 0.3, (n_trials, n_clt)).astype(float)
                mu_true, sig_true = 0.3, np.sqrt(0.3*0.7)
            means = raw.mean(axis=1)
            z_scores = (means - mu_true) / (sig_true / np.sqrt(n_clt))
            x_norm = np.linspace(-4, 4, 200)
            fig_clt = go.Figure()
            fig_clt.add_trace(go.Histogram(x=z_scores, histnorm='probability density', nbinsx=50,
                                            name=f"n={n_clt}", opacity=0.7,
                                            marker_color='steelblue'))
            fig_clt.add_trace(go.Scatter(x=x_norm, y=sps.norm.pdf(x_norm),
                                          name="N(0,1)", line=dict(color='crimson', width=2.5)))
            fig_clt.update_layout(title=f"Standardized sample means ({src_dist}, n={n_clt})",
                                   xaxis_title="z-score", yaxis_title="Density",
                                   height=320, margin=dict(t=40))
            st.plotly_chart(fig_clt, use_container_width=True)


def chapter3():
    st.header("Chapter 3 — Conditional Probability and Conditional Expectation")
    tabs = st.tabs(["Conditional Distributions", "Tower Property & Random Sums",
                    "Conditional Variance", "Conditioning on Events", "🎯 Interactive"])

    with tabs[0]:
        st.subheader("§2 Conditional Distributions — Discrete Case")
        defbox("Conditional PMF",
               "p_{X|Y}(x|y) = P(X=x, Y=y)/P(Y=y). This is a valid PMF in x for fixed y. "
               "Think of it as the distribution of X in the sub-population where Y=y.")
        defbox("Conditional Expectation (discrete)",
               "E[X|Y=y] = Σ_x x·P(X=x|Y=y)")

        st.subheader("§3 Conditional Densities — Continuous Case")
        defbox("Conditional Density",
               "f_{X|Y}(x|y) = f_{X,Y}(x,y)/f_Y(y), where f_Y(y) = ∫f_{X,Y}(x,y)dx")
        defbox("Conditional Expectation (continuous)",
               "E[X|Y=y] = ∫ x · f_{X|Y}(x|y) dx")
        keybox("E[X|Y=y] is a NUMBER (for fixed y). E[X|Y] is a RANDOM VARIABLE — a function of Y.")

    with tabs[1]:
        st.subheader("§4.1 Law of Total Expectation (Tower Property)")
        thmbox("Tower Property",
               "E[X] = E[E[X|Y]]. For discrete Y: E[X] = Σ_y E[X|Y=y]·P(Y=y).")
        keybox("Conditioning is a strategy: pick Y so E[X|Y] is easy, then average it. "
               "A good Y is often: first step, first trial, or a count that simplifies the problem.")

        st.subheader("§4.2 Random Sums")
        propbox("Mean of a Random Sum",
                "S = X₁+⋯+Xₙ with N independent of i.i.d. Xᵢ (mean μ): E[S] = E[N]·μ")
        exbox("Accident injuries", "E[N]=4 accidents/week, E[X₁]=2 injuries/accident → E[S]=8.")

        st.subheader("§4.3 Restart Trick / First-Step Recursion")
        keybox("Many expected values follow the pattern: condition on the first event. "
               "If failure occurs first, the problem restarts. This turns expectations into linear equations.")
        exbox("Geometric mean via restart",
              "N = #trials until first success. Conditioning: E[N] = p·1 + (1−p)·(1+E[N]) → E[N]=1/p.")
        exbox("Consecutive successes (E[N_k])",
              "Condition on when the first failure occurs in a run. Recursion: "
              "M_k = M_{k−1}/p^k + M_{k−1}/(1−p) + M_{k−1} relates M_k to M_{k−1}.")

    with tabs[2]:
        st.subheader("§5 Conditional Variance and Law of Total Variance")
        defbox("Conditional Variance",
               "Var(X|Y=y) = E[X²|Y=y] − (E[X|Y=y])²")
        propbox("Law of Total Variance",
                "Var(X) = E[Var(X|Y)] + Var(E[X|Y])")
        st.markdown("**Interpretation:**")
        st.latex(r"\underbrace{\text{Var}(X)}_{\text{total}} = \underbrace{E[\text{Var}(X|Y)]}_{\text{avg residual after Y}} + \underbrace{\text{Var}(E[X|Y])}_{\text{from randomness of Y}}")
        propbox("Variance of a Random Sum",
                "Var(S) = E[N]σ² + μ²·Var(N), where S=Σᵢ^N Xᵢ, Xᵢ i.i.d. mean μ, var σ², indep of N.")

    with tabs[3]:
        st.subheader("§6 Computing Probabilities by Conditioning")
        keybox("Probabilities are expectations of indicators: P(E) = E[1_E]. Conditioning: P(E) = E[P(E|Y)].")
        propbox("Total Probability via Conditioning",
                "P(E) = Σ_y P(E|Y=y)P(Y=y) [Y discrete] or ∫P(E|Y=y)f_Y(y)dy [Y continuous]")
        propbox("E[X] via Tail Sums",
                "X nonneg integer: E[X] = Σ_{n=0}^∞ P(X>n). "
                "X nonneg continuous: E[X] = ∫_0^∞ P(X>t)dt.")
        exbox("Mixture Poisson",
              "Λ~Exp(1), X|Λ=λ~Poisson(λ). Then P(X=n) = ∫₀^∞ e^{-λ}λⁿ/n! · e^{-λ}dλ = 1/2^{n+1}. "
              "(Geometric distribution on {0,1,2,…}.)")

    with tabs[4]:
        st.subheader("🎯 Tower Property: Visual Decomposition")
        st.write("Suppose Y~Bernoulli(q) and X|(Y=0)~N(0,1), X|(Y=1)~N(m,1).")
        col1, col2 = st.columns([1, 2])
        with col1:
            q_val = st.slider("q = P(Y=1)", 0.01, 0.99, 0.4, 0.01)
            m_val = st.slider("Mean shift m", -4.0, 4.0, 2.0, 0.1)
            e_x = q_val * m_val + (1-q_val)*0
            var_inner = 1.0  # E[Var(X|Y)] = 1 for both
            var_outer = q_val*(1-q_val)*m_val**2  # Var(E[X|Y]) = Var(q_val*m*Y)
            var_total = var_inner + var_outer
            st.metric("E[X] = q·m", f"{e_x:.3f}")
            st.metric("E[Var(X|Y)] (within)", f"{var_inner:.3f}")
            st.metric("Var(E[X|Y]) (between)", f"{var_outer:.3f}")
            st.metric("Var(X) = sum", f"{var_total:.3f}")
        with col2:
            xs = np.linspace(-5, m_val+4, 500)
            mix_pdf = (1-q_val)*sps.norm.pdf(xs, 0, 1) + q_val*sps.norm.pdf(xs, m_val, 1)
            fig_mix = go.Figure()
            fig_mix.add_trace(go.Scatter(x=xs, y=sps.norm.pdf(xs,0,1)*(1-q_val),
                                          fill='tozeroy', name=f"Y=0 component", opacity=0.5,
                                          line=dict(color='steelblue')))
            fig_mix.add_trace(go.Scatter(x=xs, y=sps.norm.pdf(xs,m_val,1)*q_val,
                                          fill='tozeroy', name=f"Y=1 component", opacity=0.5,
                                          line=dict(color='crimson')))
            fig_mix.add_trace(go.Scatter(x=xs, y=mix_pdf, name="Marginal f_X",
                                          line=dict(color='black', width=2, dash='dash')))
            fig_mix.update_layout(title="Mixture Distribution", height=320,
                                   xaxis_title="x", yaxis_title="f(x)",
                                   margin=dict(t=40))
            st.plotly_chart(fig_mix, use_container_width=True)
        keybox(f"Within-group variance = {var_inner:.2f}, Between-group variance = {var_outer:.3f}. "
               f"Total Var(X) = {var_total:.3f}. When m is large, between-group variance dominates.")


def chapter4():
    st.header("Chapter 4 — Markov Chains")
    tabs = st.tabs(["Definition & Transition Matrix", "Multi-Step Transitions",
                    "Classes & Recurrence", "Stationary Distributions",
                    "Gambler's Ruin", "Transient States", "Branching Processes",
                    "Time Reversibility & MDPs"])

    # ── Tab 0 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("§2 Markov Chains: Definition")
        defbox("Markov Chain (time-homogeneous)",
               "Stochastic process {Xₙ} on S is a Markov chain if "
               "P(Xₙ₊₁=j|Xₙ=i, Xₙ₋₁,…,X₀) = Pᵢⱼ (does not depend on n or history).")
        defbox("Transition Matrix P",
               "Pᵢⱼ = probability of going i→j. P is a stochastic matrix: Pᵢⱼ≥0, Σⱼ Pᵢⱼ=1.")
        keybox("Fully specified by: (1) state space S, (2) transition matrix P, (3) initial distribution.")

        st.subheader("🎯 Transition Matrix Explorer")
        st.write("Enter a 3-state transition matrix (rows must sum to 1):")
        col1, col2 = st.columns(2)
        with col1:
            default_P = [[0.5,0.4,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]]
            labels = ["State 0","State 1","State 2"]
            P_input = []
            for i in range(3):
                row = st.text_input(f"Row {i} (space-separated)", value=" ".join(map(str,default_P[i])),
                                     key=f"p_row_{i}")
                try:
                    vals = list(map(float, row.split()))
                    if len(vals)==3 and abs(sum(vals)-1)<0.01:
                        P_input.append(vals)
                    else:
                        P_input.append(default_P[i])
                        st.warning(f"Row {i} must have 3 non-negative numbers summing to 1.")
                except:
                    P_input.append(default_P[i])
        with col2:
            if len(P_input)==3:
                P_arr = np.array(P_input)
                df_p = pd.DataFrame(P_arr, index=labels, columns=labels)
                st.write("**Transition Matrix P:**")
                st.dataframe(df_p.style.format("{:.3f}"), use_container_width=True)
                # Visualize as heatmap
                fig_hm = go.Figure(go.Heatmap(z=P_arr, x=labels, y=labels,
                                               colorscale='Blues', zmin=0, zmax=1,
                                               text=[[f"{P_arr[i][j]:.2f}" for j in range(3)] for i in range(3)],
                                               texttemplate="%{text}"))
                fig_hm.update_layout(title="P matrix heatmap", height=250, margin=dict(t=40,b=20))
                st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("§4 Multi-Step Transitions & Chapman–Kolmogorov")
        defbox("n-step Transition", "P^(n)_ij = P(X_{k+n}=j|X_k=i). By time-homogeneity this doesn't depend on k.")
        propbox("Matrix Powers", "P^(n) = Pⁿ (n-th power of the one-step matrix).")
        thmbox("Chapman–Kolmogorov", "P^(n+m) = P^(n)·P^(m). Equivalently: Pᵢⱼ^(n+m) = Σ_k Pᵢₖ^(n) Pₖⱼ^(m).")

        st.subheader("🎯 Compute P^n and n-step Forecasts")
        col1, col2 = st.columns([1,2])
        with col1:
            n_steps = st.slider("Number of steps n", 1, 50, 5)
            init_state = st.selectbox("Initial state X₀", [0,1,2])
        with col2:
            try:
                P_n = np.linalg.matrix_power(P_arr, n_steps)
                df_pn = pd.DataFrame(P_n, index=labels, columns=labels)
                st.write(f"**P^{n_steps} (n-step transition matrix):**")
                st.dataframe(df_pn.style.format("{:.4f}"), use_container_width=True)
                st.write(f"Starting from state {init_state}, after {n_steps} steps:")
                for j, lbl in enumerate(labels):
                    st.write(f"  P(X_{n_steps}={j}|X_0={init_state}) = **{P_n[init_state,j]:.4f}**")
            except:
                st.error("Please fix transition matrix.")

        # Distribution over time
        if 'P_arr' in dir():
            st.subheader("Distribution Evolution Over Time")
            init_dist = np.zeros(3); init_dist[init_state] = 1.0
            T_max = 30
            dist_over_time = [init_dist.copy()]
            d = init_dist.copy()
            for _ in range(T_max):
                d = d @ P_arr
                dist_over_time.append(d.copy())
            dist_matrix = np.array(dist_over_time)
            fig_ev = go.Figure()
            colors = ['steelblue','crimson','green']
            for j in range(3):
                fig_ev.add_trace(go.Scatter(x=list(range(T_max+1)), y=dist_matrix[:,j],
                                             name=f"P(X_n={j})", line=dict(color=colors[j],width=2)))
            fig_ev.update_layout(title=f"Distribution P(Xₙ=j|X₀={init_state}) over time",
                                  xaxis_title="n", yaxis_title="Probability", height=300,
                                  margin=dict(t=40))
            st.plotly_chart(fig_ev, use_container_width=True)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("§5 Communicating Classes and Irreducibility")
        defbox("Accessibility", "j is accessible from i (i→j) if ∃n≥0: P^(n)_ij > 0.")
        defbox("Communication", "i↔j if i→j AND j→i. Communication is an equivalence relation → partitions S.")
        defbox("Irreducible", "All states communicate (single communicating class).")
        rmkbox("Recurrence and transience are CLASS properties: if i is recurrent/transient and i↔j, then j is also recurrent/transient.")

        st.subheader("§6 Recurrence and Transience")
        defbox("Return Probability", "f_ii = P(return to i | X₀=i). Recurrent: f_ii=1. Transient: f_ii<1.")
        propbox("Criterion via Return Probabilities",
                "i is recurrent iff Σₙ P^(n)_ii = ∞. i is transient iff Σₙ P^(n)_ii < ∞.")
        keybox("Every state is either recurrent or transient — no third option. "
               "Finite irreducible chain → all states recurrent.")
        exbox("Pólya's Theorem",
              "Simple symmetric random walk on ℤᵈ: recurrent for d=1,2 (drunk man finds home); "
              "transient for d≥3 (drunk bird gets lost forever!). "
              "Key: P^(2n)_{0,0} ≍ n^(−d/2), and Σn^(−d/2) diverges iff d≤2.")

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("§7 Stationary Distributions and Long-Run Behavior")
        defbox("Stationary Distribution π",
               "π = πP, π≥0, Σπⱼ=1. If X₀~π then Xₙ~π for all n. "
               "Long-run proportion of time in j: Mⱼ = 1/mⱼ (mⱼ = mean return time).")
        propbox("Positive vs. Null Recurrence",
                "Positive recurrent: mⱼ<∞ ↔ Mⱼ>0. Null recurrent: mⱼ=∞ ↔ Mⱼ=0. "
                "Symmetric 1D random walk on ℤ is null recurrent.")
        defbox("Ergodic Chain",
               "Irreducible + positive recurrent + aperiodic ⟹ ergodic. "
               "For ergodic chains: lim P^(n)_ij = πⱼ for all i,j.")
        keybox("Irreducible+pos.recurrent ⟹ unique π (long-run fractions). "
               "Add aperiodicity ⟹ transition probabilities converge to π.")
        defbox("Periodicity", "d(i) = gcd{n≥1: P^(n)_ii > 0}. Aperiodic: d(i)=1.")

        st.subheader("🎯 Stationary Distribution Solver")
        st.write("Solves π = πP (using eigenvalue method) for the 3-state matrix above.")
        try:
            evals, evecs = np.linalg.eig(P_arr.T)
            idx = np.argmin(np.abs(evals - 1.0))
            pi = np.real(evecs[:, idx])
            pi = pi / pi.sum()
            df_pi = pd.DataFrame({"State": labels, "π (stationary)": pi,
                                   "Mⱼ = 1/mⱼ": pi})
            st.dataframe(df_pi.style.format({"π (stationary)": "{:.4f}", "Mⱼ = 1/mⱼ": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            # Compare with P^100
            P100 = np.linalg.matrix_power(P_arr, 100)
            st.write("**Verification — P^100 rows (should ≈ π for ergodic chains):**")
            st.dataframe(pd.DataFrame(P100, index=labels, columns=labels).style.format("{:.4f}"),
                         use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Tab 4 (Gambler's Ruin) ─────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("§8 Gambler's Ruin Problem")
        propbox("Absorption Probability",
                "Pᵢ = P(hit N before 0 | X₀=i). P₀=0, Pₙ=1. For 1≤i≤N−1: Pᵢ = pPᵢ₊₁ + qPᵢ₋₁. "
                "Solution: Pᵢ = [1−(q/p)ⁱ]/[1−(q/p)^N] (p≠1/2), Pᵢ=i/N (p=1/2).")
        propbox("Mean Duration",
                "E[T|X₀=i]: for p=1/2, E[T]=i(N−i). For p≠1/2, closed-form involving (q/p)^i.")
        rmkbox("Infinitely rich adversary (N→∞): Pᵢ→0 if p≤1/2; Pᵢ→1−(q/p)ⁱ if p>1/2.")

        st.subheader("🎯 Gambler's Ruin Simulator")
        col1, col2 = st.columns([1, 2])
        with col1:
            N_gr = st.slider("Target fortune N", 5, 50, 20)
            p_gr = st.slider("Win probability p", 0.05, 0.95, 0.5, 0.01)
            q_gr = 1 - p_gr
        with col2:
            fortunes = np.arange(0, N_gr+1)
            if abs(p_gr - 0.5) < 1e-9:
                probs_win = fortunes / N_gr
                mean_time = fortunes * (N_gr - fortunes)
            else:
                r = q_gr / p_gr
                probs_win = (1 - r**fortunes) / (1 - r**N_gr)
                probs_win[0] = 0.0; probs_win[-1] = 1.0
                # Mean time (formula from §8)
                mean_time = np.zeros(N_gr+1)
                for i_f in range(1, N_gr):
                    mean_time[i_f] = (i_f/(p_gr-q_gr) - N_gr/(p_gr-q_gr) * (1-r**i_f)/(1-r**N_gr)
                                      if abs(p_gr-0.5)>1e-9 else i_f*(N_gr-i_f))

            fig_gr = make_subplots(rows=1, cols=2,
                                    subplot_titles=["P(win | fortune i)", "E[Duration | fortune i]"])
            fig_gr.add_trace(go.Scatter(x=fortunes, y=probs_win, mode='lines+markers',
                                         name="P(win)", line=dict(color='green',width=2)), row=1,col=1)
            fig_gr.add_trace(go.Scatter(x=fortunes[1:-1], y=mean_time[1:-1], mode='lines+markers',
                                         name="E[T]", line=dict(color='orange',width=2)), row=1,col=2)
            if abs(p_gr-0.5)<1e-9:
                fig_gr.add_trace(go.Scatter(x=fortunes, y=fortunes/N_gr,
                                             name="i/N (p=0.5)", line=dict(dash='dash',color='gray')), row=1,col=1)
            fig_gr.update_layout(height=320, showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig_gr, use_container_width=True)
            i0 = st.slider("Starting fortune i", 1, N_gr-1, N_gr//3)
            st.info(f"P(win | fortune={i0}) = **{probs_win[i0]:.4f}**  |  "
                    f"E[duration | fortune={i0}] ≈ **{mean_time[i0]:.1f}** steps")

    # ── Tab 5 (Transient States) ───────────────────────────────────────────────
    with tabs[5]:
        st.subheader("§9 Mean Time Spent in Transient States")
        defbox("Setup", "T = transient states, Pₜ = sub-matrix of P restricted to T×T.")
        propbox("Fundamental Matrix",
                "S = (I − Pₜ)⁻¹. Entry sᵢⱼ = expected number of time periods in state j "
                "before absorption, given X₀=i.")
        propbox("Hitting Probabilities",
                "fᵢⱼ = P(visit j before absorption | X₀=i) = (sᵢⱼ − δᵢⱼ)/sⱼⱼ")
        exbox("Gambler's Ruin (p=0.4, N=7, start at 3)",
              "The fundamental matrix gives expected time spent at each fortune level. "
              "E.g., s₃,₅ = expected #times fortune reaches 5 before bankruptcy/success.")

    # ── Tab 6 (Branching) ─────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("§10 Branching Processes (Galton–Watson)")
        defbox("Model", "Each individual produces i.i.d. offspring (distribution {pⱼ}). "
               "Xₙ = population at generation n. X₀=1.")
        propbox("Mean and Variance of Xₙ",
                "E[Xₙ]=μⁿ. Var(Xₙ)=σ²μⁿ⁻¹(1−μⁿ)/(1−μ) if μ≠1, else nσ².")
        defbox("PGF", "G(s) = E[sᶻ] = Σpⱼsʲ, 0≤s≤1.")
        thmbox("Extinction Criterion",
               "π = P(eventual extinction). If μ≤1: π=1. If μ>1: π<1 and π = smallest solution in [0,1] of s=G(s).")
        rmkbox("Iteration: qₙ₊₁=G(qₙ), q₀=0. Since G is increasing, qₙ↑π (the smallest fixed point of G).")

        st.subheader("🎯 Extinction Probability Calculator")
        col1, col2 = st.columns([1,2])
        with col1:
            p0 = st.slider("p₀ (P(0 offspring))", 0.0, 0.98, 0.3, 0.01)
            p1 = st.slider("p₁ (P(1 offspring))", 0.0, 1.0-p0, 0.3, 0.01)
            p2_val = round(1 - p0 - p1, 4)
            st.write(f"p₂ = 1−p₀−p₁ = **{max(0,p2_val):.4f}**")
            p2_val = max(0, p2_val)
            mu_bp = p1 + 2*p2_val
            st.metric("Offspring mean μ", f"{mu_bp:.4f}")
        with col2:
            s_range = np.linspace(0, 1, 300)
            G_vals = p0 + p1*s_range + p2_val*s_range**2
            # iterate to find fixed point
            s_iter = [0.0]
            for _ in range(200):
                s_iter.append(p0 + p1*s_iter[-1] + p2_val*s_iter[-1]**2)
            pi_est = s_iter[-1]

            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(x=s_range, y=G_vals, name="G(s)", line=dict(color='steelblue',width=2)))
            fig_bp.add_trace(go.Scatter(x=s_range, y=s_range, name="y=s", line=dict(color='gray',dash='dash')))
            fig_bp.add_vline(x=pi_est, line_dash='dot', line_color='crimson',
                              annotation_text=f"π≈{pi_est:.4f}")
            fig_bp.update_layout(title="G(s) vs s — find fixed point", xaxis_title="s",
                                  yaxis_title="G(s)", height=300, margin=dict(t=40))
            st.plotly_chart(fig_bp, use_container_width=True)
            if mu_bp <= 1:
                st.success(f"μ={mu_bp:.4f}≤1: extinction probability π = **1** (certain extinction).")
            else:
                st.warning(f"μ={mu_bp:.4f}>1: extinction probability π ≈ **{pi_est:.4f}** < 1.")

    # ── Tab 7 (TR & MDP) ──────────────────────────────────────────────────────
    with tabs[7]:
        t1, t2 = st.tabs(["Time Reversibility", "Markov Decision Processes"])
        with t1:
            st.subheader("§11 Time Reversibility")
            keybox("Running an ergodic chain in stationarity: the backward-time process is also a Markov chain. "
                   "The original is time reversible if the reversed chain has the SAME transition matrix.")
            defbox("Detailed Balance (DB)",
                   "πᵢPᵢⱼ = πⱼPⱼᵢ for all i,j. "
                   "Interpretation: rate i→j = rate j→i in steady state.")
            propbox("Useful Sufficient Condition",
                    "If ∃ xᵢ≥0, Σxᵢ=1 s.t. xᵢPᵢⱼ=xⱼPⱼᵢ for all i,j, then x=π (stationary) and chain is time-reversible.")
            propbox("Cycle Criterion",
                    "Chain is time-reversible iff for every cycle i→i₁→⋯→iₖ→i: "
                    "Pᵢᵢ₁Pᵢ₁ᵢ₂⋯Pᵢₖᵢ = Pᵢᵢₖ⋯Pᵢ₂ᵢ₁Pᵢ₁ᵢ")
            exbox("Birth–Death chains", "Every birth–death chain is time reversible. Detailed balance: πₙλₙ = πₙ₊₁μₙ₊₁.")
        with t2:
            st.subheader("§12 Markov Decision Processes")
            defbox("MDP", "Tuple (S, {A(s)}, P(·|s,a), r(s,a)). At time t: state Xₜ=s, action Aₜ=a, "
                   "next state Xₜ₊₁~P(·|s,a), reward r(s,a).")
            keybox("Fixing a policy π(a|s) turns an MDP into an ordinary Markov chain with Pᵖ(s,s')=Σₐπ(a|s)P(s'|s,a).")
            propbox("Bellman Optimality (discounted, γ∈(0,1))",
                    "V*(s) = max_a [r(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]. "
                    "Q*(s,a) = r(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a').")
            propbox("Bellman Optimality (finite horizon H)",
                    "V*_t(s) = max_a [r(s,a)+Σ_{s'}P(s'|s,a)V*_{t+1}(s')]. V*_H(s)=0.")
            st.subheader("Algorithms")
            col1, col2 = st.columns(2)
            with col1:
                box("thm-box","<b>Value Iteration:</b> V_{k+1}=T·V_k (Bellman operator T). "
                    "V_k→V* under standard conditions (finite S, bounded rewards, γ<1).")
            with col2:
                box("thm-box","<b>Policy Iteration:</b><br>"
                    "1. Eval: compute Vᵖ (solve Bellman equations for π).<br>"
                    "2. Improve: π_new(s) = argmax_a[r(s,a)+γΣP(s'|s,a)Vᵖ(s')].<br>"
                    "Converges in finitely many steps for finite MDPs.")
            rmkbox("MDP → RL: In RL, P and r are UNKNOWN and learned from data. "
                   "Slow mixing ⟹ correlated data ⟹ harder estimation.")


def chapter5():
    st.header("Chapter 5 — The Exponential Distribution and the Poisson Process")
    tabs = st.tabs(["Exponential Distribution", "Poisson Process",
                    "Splitting & Superposition", "Generalizations"])

    # ── Tab 0 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("§2 The Exponential Distribution")
        defbox("Exp(λ)", "f(x)=λe^{-λx}, x≥0. CDF: F(x)=1−e^{-λx}. "
               "E[X]=1/λ, Var(X)=1/λ², MGF=λ/(λ−t), t<λ.")
        keybox("λ is a rate (events/time). Larger λ = faster events = smaller mean wait.")

        st.subheader("§2.2 The Memoryless Property")
        propbox("Memoryless",
                "P(X>s+t|X>s) = P(X>t) for all s,t≥0. "
                "UNIQUE continuous memoryless distribution.")
        keybox("Interpretation: given you've waited s units with no event, "
               "the remaining wait has the SAME Exp(λ) distribution. The exponential has no memory.")

        st.subheader("§2.3 Racing Exponentials")
        propbox("Minimum & Winner",
                "X₁~Exp(λ₁),…,Xₙ~Exp(λₙ) independent. "
                "M=min Xᵢ ~ Exp(Σλᵢ). P(Xᵢ=M) = λᵢ/Σλⱼ.")
        propbox("Independence of Minimum and Rank Order",
                "M=min Xᵢ is INDEPENDENT of the rank ordering permutation Π=(i₁,…,iₙ). "
                "So knowing which clock finishes first tells us nothing extra about the remaining order.")

        st.subheader("🎯 Memoryless Property Demo")
        col1, col2 = st.columns([1, 2])
        with col1:
            lam_exp = st.slider("Rate λ", 0.1, 5.0, 1.0, 0.1)
            s_val = st.slider("Already waited s", 0.0, 5.0, 1.0, 0.1)
            t_show = st.slider("Extra time t", 0.1, 5.0, 1.0, 0.1)
            prob_orig = np.exp(-lam_exp * t_show)
            prob_cond = np.exp(-lam_exp * t_show)  # same by memoryless!
            st.metric("P(X>t)", f"{prob_orig:.4f}")
            st.metric("P(X>s+t | X>s)", f"{prob_cond:.4f}")
            st.success("They are equal! This is the memoryless property.")
        with col2:
            xs_e = np.linspace(0, max(8/lam_exp, s_val+t_show+2), 400)
            survival = np.exp(-lam_exp * xs_e)
            survival_cond = np.exp(-lam_exp * (xs_e - s_val))
            survival_cond[xs_e < s_val] = np.nan
            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(x=xs_e, y=survival, name="P(X>t) original",
                                          line=dict(color='steelblue',width=2)))
            fig_mem.add_trace(go.Scatter(x=xs_e, y=survival_cond, name=f"P(X>t|X>s={s_val:.1f})",
                                          line=dict(color='crimson',width=2,dash='dash')))
            fig_mem.add_vline(x=s_val, line_dash='dot', line_color='green',
                               annotation_text=f"s={s_val}")
            fig_mem.update_layout(title="Survival function — memoryless property",
                                   xaxis_title="time", yaxis_title="Probability",
                                   height=320, margin=dict(t=40))
            st.plotly_chart(fig_mem, use_container_width=True)

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("§3 The Poisson Process")
        defbox("Counting Process",
               "{N(t)}: N(t)∈{0,1,2,…}, N(0)=0, nondecreasing, jumps by +1 only.")
        defbox("Poisson Process (rate λ)",
               "1. Independent increments: counts in disjoint intervals are independent. "
               "2. P(N(t+h)−N(t)=1)=λh+o(h). P(N(t+h)−N(t)≥2)=o(h).")
        thmbox("Distribution",
               "N(t) ~ Poisson(λt). E[N(t)]=λt. Var[N(t)]=λt.")
        propbox("Interarrival Times",
                "T₁,T₂,… are i.i.d. Exp(λ). (S_n = T₁+⋯+Tₙ ~ Gamma(n,λ)).")
        keybox("Two views: (counting) N(t)~Poisson(λt) with indep.increments; "
               "(waiting-time) interarrivals are i.i.d. Exp(λ). Use whichever is easier.")

        st.subheader("🎯 Poisson Process Sample Path Simulator")
        col1, col2 = st.columns([1,2])
        with col1:
            lam_pp = st.slider("Rate λ", 0.5, 10.0, 3.0, 0.5)
            T_pp = st.slider("Time horizon T", 1.0, 20.0, 5.0, 0.5)
            seed_pp = st.number_input("Random seed", 0, 999, 42, 1)
        with col2:
            np.random.seed(int(seed_pp))
            n_events = np.random.poisson(lam_pp * T_pp)
            event_times = sorted(np.random.uniform(0, T_pp, n_events))
            # Build step function
            t_path = [0] + [t for t in event_times for _ in range(2)] + [T_pp]
            n_path = [0] + [j for i,_ in enumerate(event_times) for j in (i, i+1)] + [n_events]
            fig_pp = go.Figure()
            fig_pp.add_trace(go.Scatter(x=t_path, y=n_path, mode='lines',
                                         name=f"N(t), λ={lam_pp}",
                                         line=dict(color='steelblue', width=2.5, shape='hv')))
            fig_pp.add_trace(go.Scatter(x=event_times, y=list(range(1,len(event_times)+1)),
                                         mode='markers', name='Events',
                                         marker=dict(color='crimson', size=8)))
            fig_pp.add_trace(go.Scatter(x=[0, T_pp], y=[0, lam_pp*T_pp], mode='lines',
                                         name=f"E[N(t)]=λt", line=dict(dash='dash', color='gray')))
            fig_pp.update_layout(title=f"Poisson process: λ={lam_pp}, T={T_pp}, observed {n_events} events",
                                  xaxis_title="t", yaxis_title="N(t)", height=330, margin=dict(t=40))
            st.plotly_chart(fig_pp, use_container_width=True)

            col_a, col_b = st.columns(2)
            col_a.metric("Observed events", n_events)
            col_b.metric("Expected λT", f"{lam_pp*T_pp:.1f}")

    # ── Tab 2 (Splitting) ─────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("§3.4 Thinning / Splitting")
        propbox("Splitting",
                "PP(λ), label each event type I w.p. p, type II w.p. 1−p independently. "
                "Then N₁~PP(λp) and N₂~PP(λ(1−p)) are INDEPENDENT.")

        st.subheader("§3.5 Superposition and Bernoulli Labeling")
        propbox("Superposition",
                "N^(1)~PP(λ₁), N^(2)~PP(λ₂) independent. Merged: N=N^(1)+N^(2)~PP(λ₁+λ₂).")
        propbox("Bernoulli Labeling",
                "Given N(t)=n, N^(1)(t)|N(t)=n ~ Binomial(n, p=λ₁/(λ₁+λ₂)).")
        propbox("Order Statistics Property",
                "Given N(t)=n, arrival times (S₁,…,Sₙ) ~ order statistics of n i.i.d. Unif(0,t). "
                "Equivalently: f_{S₁,…,Sₙ|N(t)=n}(s₁,…,sₙ) = n!/tⁿ for 0<s₁<⋯<sₙ<t.")

        st.subheader("§3.6 Time-Varying Classification")
        propbox("Time-Dependent Thinning",
                "Event at time y classified as type i w.p. Pᵢ(y). "
                "Then {Nᵢ(t)} are INDEPENDENT Poisson with E[Nᵢ(t)] = λ∫₀ᵗ Pᵢ(s)ds.")

    # ── Tab 3 (Generalizations) ────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("§4.1 Nonhomogeneous Poisson Process")
        defbox("Nonhomogeneous PP",
               "Rate λ(t) varies with time. Independent increments still hold, but NOT stationary. "
               "m(t)=∫₀ᵗλ(s)ds (mean value function). N(t)~Poisson(m(t)).")
        rmkbox("Interarrival times are NEITHER independent NOR identically distributed.")

        st.subheader("§4.2 Compound Poisson Process")
        defbox("Compound PP",
               "S(t) = X₁+⋯+X_{N(t)}, Xᵢ i.i.d., independent of PP(λ). "
               "E[S(t)]=λt·E[X₁]. Var(S(t))=λt·E[X₁²].")
        rmkbox("Note: Var = λt·E[X²], NOT λt·Var(X)! The Poisson counting contributes an extra term.")

        st.subheader("🎯 Compound Poisson: Mean & Variance Calculator")
        col1, col2 = st.columns(2)
        with col1:
            lam_cp = st.slider("Poisson rate λ", 0.1, 10.0, 2.0, 0.1)
            t_cp = st.slider("Time t", 0.1, 50.0, 10.0, 0.1)
            jump_type = st.selectbox("Jump distribution", ["Exponential(1)", "Uniform(0,2)", "Constant 1"])
        with col2:
            if jump_type == "Exponential(1)":
                ex1, ex2 = 1.0, 2.0
            elif jump_type == "Uniform(0,2)":
                ex1, ex2 = 1.0, 4/3
            else:
                ex1, ex2 = 1.0, 1.0
            mean_st = lam_cp * t_cp * ex1
            var_st = lam_cp * t_cp * ex2
            st.metric("E[S(t)]", f"{mean_st:.3f}")
            st.metric("Var(S(t)) = λt·E[X²]", f"{var_st:.3f}")
            st.metric("√Var(S(t))", f"{np.sqrt(var_st):.3f}")

        st.subheader("§4.4 Conditional / Mixed Poisson Process")
        defbox("Mixed Poisson",
               "L≥0 random. {N(t)}: given L=ℓ, N(t)~Poisson(ℓt). "
               "E[N(t)] = t·E[L]. Var(N(t)) = t·E[L] + t²·Var(L) > E[N(t)] (overdispersion).")
        keybox("Overdispersion: Var(N(t)) > E[N(t)]. Mixing over a random rate creates more variability "
               "than a standard Poisson — useful for count data with heterogeneous populations.")


def chapter6():
    st.header("Chapter 6 — Continuous-Time Markov Chains (CTMCs)")
    tabs = st.tabs(["CTMC Basics", "Birth–Death Processes",
                    "Kolmogorov Equations", "Steady State & Reversibility"])

    # ── Tab 0 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("§2 Continuous-Time Markov Chains")
        defbox("CTMC (time-homogeneous)",
               "P(X(t+s)=j|X(s)=i, past) = P(X(t+s)=j|X(s)=i). Time-homogeneous: doesn't depend on s.")
        propbox("Holding Times are Exponential",
                "By Markov property + time-homogeneity: "
                "P(Tᵢ>s+t|Tᵢ>s)=P(Tᵢ>t) ⟹ Tᵢ ~ Exp(vᵢ). E[Tᵢ]=1/vᵢ.")
        keybox("Equivalent description: entering state i, (i) stay Exp(vᵢ) time, "
               "(ii) jump to j with probability Pᵢⱼ. Staying time and next state are INDEPENDENT.")
        defbox("Instantaneous Rates",
               "qᵢⱼ = vᵢ·Pᵢⱼ (i≠j). Meaning: when in i, rate of transitions to j is qᵢⱼ. "
               "vᵢ = Σⱼ₍≠ᵢ₎ qᵢⱼ (total rate out of i).")

    # ── Tab 1 (BD) ─────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("§3 Birth–Death Processes")
        defbox("Birth–Death Process",
               "CTMC on {0,1,2,…}. From n: births n→n+1 at rate λₙ, deaths n→n-1 at rate μₙ (μ₀=0). "
               "vₙ = λₙ+μₙ (n≥1). Pₙ,ₙ₊₁ = λₙ/(λₙ+μₙ).")
        st.subheader("Key Examples")
        col1, col2 = st.columns(2)
        with col1:
            box("def-box","<b>M/M/1 Queue:</b> λₙ=λ (all n), μₙ=μ (n≥1). "
                "Single server. ρ=λ/μ. Stable iff ρ&lt;1.")
            box("def-box","<b>M/M/s Queue:</b> λₙ=λ, μₙ=min(n,s)·μ. "
                "s servers. Stable iff λ&lt;sμ.")
        with col2:
            box("def-box","<b>Yule Process:</b> λₙ=nλ, μₙ=0. Pure birth. "
                "Starting from 1: X(t) ~ Geometric(e^{-λt}).")
            box("def-box","<b>Machine Repair (M machines, 1 repairman):</b> "
                "λₙ=(M−n)λ (failures), μₙ=μ (repairs, n≥1).")

        st.subheader("🎯 M/M/1 Queue Metrics")
        col1, col2 = st.columns([1,2])
        with col1:
            lam_mm1 = st.slider("Arrival rate λ", 0.1, 9.9, 3.0, 0.1)
            mu_mm1 = st.slider("Service rate μ", 0.1, 10.0, 4.0, 0.1)
            rho = lam_mm1 / mu_mm1
        with col2:
            if rho < 1:
                L = rho / (1 - rho)
                Lq = rho**2 / (1 - rho)
                W = 1 / (mu_mm1 - lam_mm1)
                Wq = lam_mm1 / (mu_mm1 * (mu_mm1 - lam_mm1))
                p0 = 1 - rho
                col_a, col_b = st.columns(2)
                col_a.metric("ρ = λ/μ", f"{rho:.3f}")
                col_a.metric("L (avg in system)", f"{L:.3f}")
                col_a.metric("Lq (avg in queue)", f"{Lq:.3f}")
                col_b.metric("P(empty) = 1−ρ", f"{p0:.3f}")
                col_b.metric("W (avg sojourn time)", f"{W:.3f}")
                col_b.metric("Wq (avg wait in queue)", f"{Wq:.3f}")
                # Steady-state distribution plot
                n_max = min(30, int(np.log(0.001)/np.log(rho))+5) if rho > 0.001 else 20
                ns = np.arange(0, n_max+1)
                pn = (1-rho) * rho**ns
                fig_mm1 = go.Figure(go.Bar(x=ns, y=pn, name="Pₙ=(1-ρ)ρⁿ", marker_color='steelblue'))
                fig_mm1.update_layout(title="M/M/1 Steady-State Distribution",
                                       xaxis_title="n (# in system)", yaxis_title="Pₙ",
                                       height=260, margin=dict(t=40))
                st.plotly_chart(fig_mm1, use_container_width=True)
            else:
                st.error(f"ρ = {rho:.3f} ≥ 1: Queue is UNSTABLE (no steady state exists)!")

    # ── Tab 2 (Kolmogorov) ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("§4 Transition Probability Functions and Kolmogorov Equations")
        defbox("Transition Probabilities", "Pᵢⱼ(t) = P(X(t+s)=j|X(s)=i). Pᵢⱼ(0)=δᵢⱼ.")
        thmbox("Kolmogorov Backward Equations",
               "P'ᵢⱼ(t) = −vᵢPᵢⱼ(t) + Σ_{k≠i} qᵢₖ Pₖⱼ(t).")
        thmbox("Kolmogorov Forward Equations",
               "P'ᵢⱼ(t) = −vⱼPᵢⱼ(t) + Σ_{k≠j} qₖⱼ Pᵢₖ(t).")
        thmbox("Matrix Form",
               "Let R be the generator matrix: Rᵢⱼ=qᵢⱼ (i≠j), Rᵢᵢ=−vᵢ. "
               "Then P(t) = e^{Rt} = Σₙ (Rt)ⁿ/n!.")
        defbox("Generator Matrix R",
               "Rᵢⱼ = qᵢⱼ for i≠j,  Rᵢᵢ = −vᵢ = −Σⱼ≠ᵢ qᵢⱼ. "
               "Each row of R sums to 0.")
        exbox("Two-state CTMC (machine up/down)",
              "State 0 = working (leaves at rate λ), state 1 = broken (repaired at rate μ). "
              "R = [[-λ, λ], [μ, -μ]]. "
              "P₀₀(t) = μ/(λ+μ) + λ/(λ+μ)·e^{-(λ+μ)t}.")

        st.subheader("🎯 Two-State CTMC Solution")
        col1, col2 = st.columns([1,2])
        with col1:
            lam_2s = st.slider("Failure rate λ", 0.1, 5.0, 1.0, 0.1)
            mu_2s = st.slider("Repair rate μ", 0.1, 5.0, 2.0, 0.1)
        with col2:
            ts = np.linspace(0, 6, 300)
            pi0 = mu_2s/(lam_2s+mu_2s); pi1 = lam_2s/(lam_2s+mu_2s)
            P00 = pi0 + pi1 * np.exp(-(lam_2s+mu_2s)*ts)
            P01 = pi1 * (1 - np.exp(-(lam_2s+mu_2s)*ts))
            fig_2s = go.Figure()
            fig_2s.add_trace(go.Scatter(x=ts, y=P00, name="P₀₀(t) (stays working)",
                                         line=dict(color='green',width=2)))
            fig_2s.add_trace(go.Scatter(x=ts, y=P01, name="P₀₁(t) (breaks down)",
                                         line=dict(color='crimson',width=2)))
            fig_2s.add_hline(y=pi0, line_dash='dot', line_color='green',
                              annotation_text=f"π₀={pi0:.3f}")
            fig_2s.update_layout(title="P(t) for two-state CTMC (start in working state)",
                                  xaxis_title="t", yaxis_title="Probability", height=300,
                                  margin=dict(t=40))
            st.plotly_chart(fig_2s, use_container_width=True)

    # ── Tab 3 (Steady-State) ──────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("§5 Limiting Probabilities and Balance Equations")
        defbox("Limiting Probabilities", "Pⱼ = lim_{t→∞} Pᵢⱼ(t) (independent of i for irreducible chain). "
               "Also: Pⱼ = long-run fraction of time in j.")
        thmbox("Global Balance",
               "vⱼPⱼ = Σ_{k≠j} Pₖqₖⱼ (rate out of j = rate into j). Σⱼ Pⱼ = 1.")
        propbox("Birth–Death: Local Balance",
                "λₙPₙ = μₙ₊₁Pₙ₊₁. Pₙ = (λ₀⋯λₙ₋₁)/(μ₁⋯μₙ)·P₀. "
                "P₀ = [1 + Σ_{n≥1} (λ₀⋯λₙ₋₁)/(μ₁⋯μₙ)]⁻¹")
        rmkbox("Stationary distribution EXISTS iff Σ_{n≥1}(λ₀⋯λₙ₋₁)/(μ₁⋯μₙ) < ∞.")
        propbox("M/M/1 Steady State", "Pₙ = (1−ρ)ρⁿ, ρ=λ/μ, provided ρ<1.")

        st.subheader("§6 Time Reversibility for CTMCs")
        defbox("Reversed CTMC",
               "Run chain in stationarity backward in time. Reversed rates: q̃ᵢⱼ = Pⱼqⱼᵢ/Pᵢ.")
        defbox("Detailed Balance (CTMC)",
               "Pᵢqᵢⱼ = Pⱼqⱼᵢ for all i≠j ↔ time reversible. "
               "Interpretation: in equilibrium, rate i→j = rate j→i.")
        propbox("Birth–Death is Time Reversible",
                "Every ergodic birth–death process satisfies detailed balance. "
                "Result: output process of M/M/s queue is a PP(λ) (Burke's theorem).")
        propbox("Truncation Preserves Reversibility",
                "Time-reversible chain truncated to A (irreducible on A): still reversible. "
                "PⱼA = Pⱼ/ΣPᵢ (i∈A).")

        st.subheader("🎯 Birth–Death Steady-State Calculator")
        st.write("Enter birth/death rates for a finite birth–death process (states 0,1,…,K):")
        K_bd = st.slider("Number of states K", 3, 10, 5)
        col1, col2 = st.columns(2)
        with col1:
            lambdas_str = st.text_input(f"Birth rates λ₀,λ₁,…,λ_{K_bd-1} (space-sep)", value=" ".join(["1.0"]*K_bd))
            mus_str = st.text_input(f"Death rates μ₁,μ₂,…,μ_{K_bd} (space-sep)", value=" ".join(["2.0"]*K_bd))
        with col2:
            try:
                lams = list(map(float, lambdas_str.split()))[:K_bd]
                mus = [0.0] + list(map(float, mus_str.split()))[:K_bd]
                while len(lams) < K_bd: lams.append(lams[-1])
                while len(mus) < K_bd+1: mus.append(mus[-1])
                # P_n = product(lams[0:n]) / product(mus[1:n+1]) * P_0
                products = [1.0]
                for n in range(1, K_bd+1):
                    prod = products[-1] * lams[n-1] / mus[n]
                    products.append(prod)
                C = sum(products)
                P_bd = [p/C for p in products]
                df_bd = pd.DataFrame({"State n": range(K_bd+1), "Pₙ (steady state)": P_bd})
                st.dataframe(df_bd.style.format({"Pₙ (steady state)": "{:.5f}"}),
                             use_container_width=True, hide_index=True)
                fig_bd = go.Figure(go.Bar(x=list(range(K_bd+1)), y=P_bd,
                                          marker_color='steelblue'))
                fig_bd.update_layout(title="Steady-State Distribution",
                                      xaxis_title="n", yaxis_title="Pₙ", height=250,
                                      margin=dict(t=40))
                st.plotly_chart(fig_bd, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_home():
    st.markdown('<div style="text-align:center;">'
                '<h1>🎲 ISE 5414 — Random Processes Study Hub</h1>'
                '<p style="font-size:1.1rem;color:#555;">Based on Ross: Chapters 1–6</p>'
                '</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📖 What is this app?")
        st.write("An interactive study companion for ISE 5414. Every definition, theorem, "
                 "proposition, and key idea from Chapters 1–6 is presented here with "
                 "interactive visualizations to build intuition.")
    with col2:
        st.markdown("### 🗺️ How to navigate")
        st.write("Use the **sidebar** to jump between chapters. Each chapter has **tabs** "
                 "for subtopics. Look for **🎯 Interactive** tabs for hands-on exploration.")
    with col3:
        st.markdown("### 🔍 Topic Finder")
        st.write("Can't find a concept? Use the **Topic Finder** below — it maps every "
                 "topic to its exact chapter, section, and page in the course notes.")

    st.divider()
    st.subheader("📚 Chapter Overview")
    chapters_info = [
        ("Ch 1", "Probability Theory Refresher", "Events, axioms, conditioning, Bayes, continuity", "8 pages"),
        ("Ch 2", "Random Variables", "Distributions, expectation, MGFs, CLT, SLLN", "18 pages"),
        ("Ch 3", "Conditional Expectation", "Tower property, random sums, conditional variance", "22 pages"),
        ("Ch 4", "Markov Chains", "Recurrence, stationary dist., Gambler's ruin, MDPs", "45 pages"),
        ("Ch 5", "Exponential & Poisson", "Memoryless property, Poisson process, splitting", "37 pages"),
        ("Ch 6", "Continuous-Time Markov Chains", "Kolmogorov eqs., steady state, reversibility", "22 pages"),
    ]
    for ch, title, topics, pages in chapters_info:
        with st.expander(f"**{ch}: {title}** ({pages})"):
            st.write(f"*Key topics:* {topics}")

    st.divider()

    # ── Topic Finder ──────────────────────────────────────────────────────────
    st.subheader("🔍 Topic Finder")
    search_query = st.text_input("Search for a concept, theorem, or formula:",
                                  placeholder="e.g. Bayes, stationary distribution, CLT, MGF…")
    if search_query:
        results = {k: v for k, v in TOPIC_INDEX.items()
                   if search_query.lower() in k.lower()}
        if results:
            df_res = pd.DataFrame([
                {"Topic": k, "Chapter": v[0], "Section": v[1], "Page(s)": v[2], "Tab to open": v[3]}
                for k, v in results.items()
            ])
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No results for '{search_query}'. Try synonyms or partial words.")
    else:
        st.caption("Type above to search through all topics in the course.")

    st.divider()

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.subheader("📄 Generate Study Notes PDF")
    st.write("This generates a comprehensive PDF of all key formulas, definitions, "
             "theorems, and notes from Chapters 1–6. Give it to a new AI chat session "
             "as context — it's formatted as a reference guide for an AI tutor.")
    if st.button("📥 Generate & Download Study PDF", type="primary"):
        with st.spinner("Generating PDF..."):
            pdf_buf = generate_study_pdf()
        st.download_button(
            label="⬇️ Download ISE5414_Study_Notes.pdf",
            data=pdf_buf,
            file_name="ISE5414_Study_Notes.pdf",
            mime="application/pdf",
        )
        st.success("PDF ready! Contains all definitions, theorems, and formulas from Ch 1–6.")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & ROUTING
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎲 ISE 5414 Study Hub")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠 Home",
        "📘 Ch 1: Probability Theory",
        "📗 Ch 2: Random Variables",
        "📙 Ch 3: Conditional Expectation",
        "📕 Ch 4: Markov Chains",
        "📓 Ch 5: Exponential & Poisson",
        "📔 Ch 6: CTMCs",
    ])
    st.markdown("---")
    st.caption("All content from course notes Chapters 1–6 (based on Ross textbook).")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
if "Home" in page:
    show_home()
elif "Ch 1" in page:
    chapter1()
elif "Ch 2" in page:
    chapter2()
elif "Ch 3" in page:
    chapter3()
elif "Ch 4" in page:
    chapter4()
elif "Ch 5" in page:
    chapter5()
elif "Ch 6" in page:
    chapter6()
