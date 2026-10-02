"""
Personal Finance / Budgeting Coach — App (Use Case #20)
Spend categorization, savings-goal tracking, AI monthly insights, disclaimer (not financial advice).
Gemini 1.5 Flash free key optional; rule-based engine covers full demo offline.
"""
import streamlit as st
import pandas as pd
import re
import requests
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent

st.set_page_config(page_title="Finance Coach", page_icon="💰", layout="wide")

RULES = {
    "Rent": ["rent", "landlord", "housing"],
    "Groceries": ["bigbasket", "dmart", "blinkit", "grocer", "supermarket", "milk"],
    "Dining": ["swiggy", "zomato", "restaurant", "dinner", "lunch", "cafe", "food"],
    "Transport": ["uber", "ola", "metro", "petrol", "fuel", "servicing", "parking", "cab"],
    "Shopping": ["amazon", "myntra", "flipkart", "shopping", "fashion", "gifts", "diwali"],
    "Utilities": ["electricity", "bescom", "gas", "airtel", "broadband", "mobile", "water", "bill"],
    "Entertainment": ["netflix", "hotstar", "pvr", "movies", "bookmyshow", "ott", "concert"],
    "Health": ["apollo", "pharmacy", "medplus", "fortis", "checkup", "hospital", "doctor"],
    "Income": ["salary", "payout", "freelance", "refund", "bonus"],
}

def auto_categorize(desc):
    d = desc.lower()
    for cat, kws in RULES.items():
        if any(k in d for k in kws):
            return cat
    return "Other"

def load_df(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(BASE / "sample_expenses.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])
    if "category" not in df.columns:
        df["category"] = df["description"].apply(auto_categorize)
    else:
        mask = df["category"].isna() | (df["category"].astype(str).str.strip() == "")
        df.loc[mask, "category"] = df.loc[mask, "description"].apply(auto_categorize)
    if "type" not in df.columns:
        df["type"] = df.apply(lambda r: "income" if r["category"] == "Income" or r["amount"] < 0 else "expense", axis=1)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df

def rule_insights(df):
    exp = df[df["type"] == "expense"]
    inc = df[df["type"] == "income"]["amount"].sum()
    tot = exp["amount"].sum()
    sav = inc - tot
    rate = (sav / inc * 100) if inc else 0
    by_cat = exp.groupby("category")["amount"].sum().sort_values(ascending=False)
    months = sorted(df["month"].unique())
    mom = {}
    if len(months) >= 2:
        for c in exp["category"].unique():
            a = exp[(exp["month"] == months[-2]) & (exp["category"] == c)]["amount"].sum()
            b = exp[(exp["month"] == months[-1]) & (exp["category"] == c)]["amount"].sum()
            if a > 0:
                mom[c] = round((b - a) / a * 100, 1)
    flags = []
    fun = by_cat.get("Dining", 0) + by_cat.get("Entertainment", 0) + by_cat.get("Shopping", 0)
    if tot and fun / tot > 0.35:
        flags.append(f"Lifestyle (Dining+Entertainment+Shopping) is {fun/tot:.0%} of spend — above 35% guardrail.")
    top = by_cat.head(1)
    if not top.empty and top.iloc[0] / tot > 0.4:
        flags.append(f"{top.index[0]} dominates at {top.iloc[0]/tot:.0%} — single-category concentration risk.")
    if rate < 10:
        flags.append(f"Savings rate {rate:.1f}% is below 10% floor — trim top discretionary category first.")
    tips = ["Cap Dining+Swiggy/Zomato to a weekly budget (e.g., Rs.800/week).",
            "Move 10% of salary to a separate account on payday (pay-yourself-first).",
            "Review Shopping spikes (festive) with a 48-hour waitlist rule."]
    return {"income": inc, "spend": tot, "saved": sav, "rate": rate,
            "by_cat": by_cat, "mom": mom, "flags": flags, "tips": tips, "months": months}

def gemini_insights(api_key, agg_text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
    prompt = f"""You are a budgeting coach (not a financial advisor). Given these monthly aggregates, write 4-6 short insight bullets + 3 practical saving tips under 180 words. Never recommend stocks/mutual funds/crypto. End with: 'Not financial advice — educational only.'
Data:
{agg_text[:4000]}"""
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}, timeout=30)
        if r.status_code != 200:
            return None, f"Gemini API error {r.status_code}: {r.text[:250]}"
        return r.json()["candidates"][0]["content"]["parts"][0]["text"], None
    except Exception as e:
        return None, f"Gemini call failed: {e}"

# ---------- UI ----------
st.title("💰 Personal Finance / Budgeting Coach")
st.caption("App • Auto-categorization + goal tracking + AI monthly insights • Educational only — NOT financial advice")
st.warning("⚠️ Disclaimer: educational budgeting helper only. Not financial advice. No stocks/MF/crypto recommendations. Verify with a qualified advisor.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini Free API Key (optional)", type="password")
    st.markdown("[Get free key](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("Works without key via rule engine.")

up = st.file_uploader("Upload expenses CSV (date,description,amount,type) — or use sample", type=["csv"])
df = load_df(up)
st.caption(f"Loaded {len(df)} rows • {df['month'].min()} → {df['month'].max()} • {'uploaded file' if up else 'sample_expenses.csv (Aug–Oct 2026)'}")

ins = rule_insights(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Income", f"Rs.{ins['income']:,.0f}")
k2.metric("Total Spend", f"Rs.{ins['spend']:,.0f}")
k3.metric("Saved", f"Rs.{ins['saved']:,.0f}")
k4.metric("Savings Rate", f"{ins['rate']:.1f}%")

t1, t2, t3 = st.tabs(["📊 Dashboard", "🎯 Savings Goal", "🤖 AI Insights"])
with t1:
    st.subheader("Spend by category")
    st.bar_chart(ins["by_cat"])
    st.dataframe(ins["by_cat"].reset_index().rename(columns={"amount": "spend"}), use_container_width=True)
    st.subheader("Monthly trend")
    trend = df[df["type"] == "expense"].groupby("month")["amount"].sum()
    st.line_chart(trend)
    if ins["mom"]:
        st.subheader("Month-on-month change (last 2 months)")
        mom_df = pd.DataFrame(list(ins["mom"].items()), columns=["category", "pct_change"]).sort_values("pct_change", ascending=False)
        st.dataframe(mom_df, use_container_width=True)
    with st.expander("Show categorized transactions"):
        st.dataframe(df.sort_values("date"), use_container_width=True)
with t2:
    g1, g2, g3 = st.columns(3)
    with g1: goal_name = st.text_input("Goal name", "Emergency fund")
    with g2: target = st.number_input("Target (Rs.)", 1000, 10000000, 50000, step=5000)
    with g3: already = st.number_input("Already saved (Rs.)", 0, 10000000, 10000, step=1000)
    monthly_surplus = max(ins["saved"] / max(len(ins["months"]), 1), 0)
    proj = already + monthly_surplus * 3
    pct = min(proj / target * 100 if target else 0, 100)
    st.progress(pct / 100)
    st.write(f"Projected in 3 months at current surplus (Rs.{monthly_surplus:,.0f}/mo): **Rs.{proj:,.0f} / Rs.{target:,.0f} ({pct:.1f}%)**")
    if proj >= target: st.success("On track ✅ — keep the surplus steady.")
    elif pct >= 60: st.warning("Borderline 🟡 — trim top category or extend timeline.")
    else: st.error("Off track 🔴 — cut discretionary 15% or raise target date.")
with t3:
    st.subheader("Monthly insights")
    agg = f"Months {ins['months']}; Income {ins['income']}; Spend {ins['spend']}; Saved {ins['saved']} ({ins['rate']:.1f}%); ByCat {ins['by_cat'].to_dict()}; MoM {ins['mom']}; Flags {ins['flags']}"
    if api_key:
        with st.spinner("Asking Gemini…"):
            txt, err = gemini_insights(api_key, agg)
        if txt:
            st.success("Generated with 🤖 Gemini AI")
            st.write(txt)
        else:
            st.warning(f"{err} — showing rule-based insights so demo continues.")
            txt = None
        if not txt:
            for f in ins["flags"]: st.write(f"• {f}")
            st.write("**Tips:**")
            for t in ins["tips"]: st.write(f"- {t}")
    else:
        st.info("Rule-based insights (add key for Gemini personalization):")
        for f in ins["flags"] or ["Spend looks balanced — no guardrail breached."]:
            st.write(f"• {f}")
        st.write("**Tips:**")
        for t in ins["tips"]: st.write(f"- {t}")
    st.caption("AI never recommends investments; tips are budgeting habits only.")
    md = f"# Budget Notes — {datetime.now().strftime('%b %Y')}\n\nIncome Rs.{ins['income']:,.0f} | Spend Rs.{ins['spend']:,.0f} | Saved Rs.{ins['saved']:,.0f} ({ins['rate']:.1f}%)\n\n## By category\n" + "\n".join(f"- {c}: Rs.{v:,.0f}" for c, v in ins["by_cat"].items()) + f"\n\n## Flags\n" + "\n".join(f"- {f}" for f in ins["flags"]) + "\n\n_Not financial advice._\n"
    st.download_button("⬇️ Export budget notes (Markdown)", md.encode(), "budget_notes.md", "text/markdown")
