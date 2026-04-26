import streamlit as st
import pandas as pd
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import IsolationForest
import shap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import plotly.express as px
import sqlite3
from datetime import datetime
import google.generativeai as genai

gemini_model = None

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("Gemini API key is missing. AI Copilot features are disabled.")


st.set_page_config(
    page_title="GPPA Advanced AI Procurement Risk System",
    layout="wide"
)

st.markdown("""
<style>
.kpi-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 10px;
}
.kpi-title {
    font-size: 13px;
    color: #777;
}
.kpi-value {
    font-size: 30px;
    font-weight: bold;
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered GPPA Procurement Compliance & Risk System")
st.write(
    "Advanced GovTech prototype combining GPPA compliance rules, AI-style risk scoring, "
    "SMOTE class balancing, model comparison, prediction confidence, and executive audit reporting."
)


DB_PATH = "gppa_procurement_data.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_name TEXT,
            upload_date TEXT,
            data TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_uploaded_dataset(df, upload_name):
    conn = sqlite3.connect(DB_PATH)

    data_json = df.to_json(orient="records")

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO uploaded_datasets (upload_name, upload_date, data)
        VALUES (?, ?, ?)
        """,
        (
            upload_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_json
        )
    )

    conn.commit()
    conn.close()


def load_latest_dataset():
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT upload_name, upload_date, data
        FROM uploaded_datasets
        ORDER BY id DESC
        LIMIT 1
    """

    result = pd.read_sql_query(query, conn)
    conn.close()

    if result.empty:
        return None, None, None

    df = pd.read_json(result.loc[0, "data"])
    return df, result.loc[0, "upload_name"], result.loc[0, "upload_date"]

init_database()

uploaded_file = st.file_uploader(
    "Upload procurement data file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    save_uploaded_dataset(df, uploaded_file.name)
    st.success("✅ Uploaded dataset saved permanently to database")

else:
    saved_df, saved_name, saved_date = load_latest_dataset()

    if saved_df is not None:
        df = saved_df
        st.info(f"ℹ️ Using saved dataset: {saved_name} uploaded on {saved_date}")
    else:
        try:
            df = pd.read_csv("gppa_large_dataset.csv")
            st.info("ℹ️ No saved upload found — using default sample dataset")
        except FileNotFoundError:
            st.warning("⚠️ No dataset available. Please upload a file.")
            st.stop()



def yes_no(value):
    return str(value).strip().lower() in ["yes", "true", "1", "y"]


def get_number(row, col, default=0):
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def check_compliance(row):
    flags = []
    total_rules = 0
    passed_rules = 0

    category = str(row.get("procurement_category", "")).strip().lower()
    method = str(row.get("procurement_method", "")).strip().lower()

    amount = get_number(row, "amount")
    quotes = int(get_number(row, "number_of_quotes"))
    tender_days = int(get_number(row, "tender_days"))
    variation = get_number(row, "variation_percentage")

    gppa_approval = yes_no(row.get("gppa_approval", "no"))
    supplier_registered = yes_no(row.get("supplier_registered", "no"))
    monthly_report = yes_no(row.get("monthly_report_submitted", "no"))

    def rule(condition, message):
        nonlocal total_rules, passed_rules
        total_rules += 1
        if condition:
            flags.append(message)
        else:
            passed_rules += 1

    # Universal rules
    rule(not supplier_registered, "Supplier is not registered")
    rule(not monthly_report, "Monthly procurement report not submitted")
    rule(variation > 5, "Contract variation above 5%")

    # Method and threshold rules
    if method == "rfq":
        rule(quotes < 3, "RFQ requires at least 3 quotations")

    if amount >= 1_000_000:
        rule(not gppa_approval, "GPPA approval missing for procurement ≥ D1,000,000")
        rule(tender_days < 30, "Tender period below 30 days for high-value procurement")

    if amount > 3_000_000:
        rule(method not in ["open_tender", "international_tender"], "Procurement above D3,000,000 may require open/international tendering")

    # Goods-specific rules
    if category == "goods":
        rule(
            yes_no(row.get("bid_security_required", "no")) and not yes_no(row.get("bid_security_submitted", "no")),
            "Goods: bid security required but not submitted"
        )
        rule(
            yes_no(row.get("inspection_certificate_required", "no")) and not yes_no(row.get("inspection_certificate_submitted", "no")),
            "Goods: inspection/manufacturer certificate required but missing"
        )

    # Services-specific rules
    if category == "services":
        rule(not yes_no(row.get("technical_proposal", "no")), "Services: technical proposal missing")
        rule(not yes_no(row.get("financial_proposal", "no")), "Services: financial proposal missing")
        rule(not yes_no(row.get("tor_attached", "no")), "Services: Terms of Reference missing")

    # Complex works-specific rules
    if category == "complex_works":
        rule(not yes_no(row.get("site_visit_done", "no")), "Complex works: site visit required but not completed")
        rule(not yes_no(row.get("performance_security", "no")), "Complex works: performance security missing")
        rule(not yes_no(row.get("technical_director_assigned", "no")), "Complex works: technical director not assigned")
        rule(not yes_no(row.get("essential_equipment_available", "no")), "Complex works: essential equipment not available")

    compliance_score = round((passed_rules / total_rules) * 100, 2) if total_rules else 100
    compliance_risk_score = 100 - compliance_score

    return compliance_score, compliance_risk_score, flags


def ai_risk_score(row, compliance_risk_score):
    score = compliance_risk_score
    reasons = []

    amount = get_number(row, "amount")
    method = str(row.get("procurement_method", "")).strip().lower()
    quotes = int(get_number(row, "number_of_quotes"))
    tender_days = int(get_number(row, "tender_days"))
    variation = get_number(row, "variation_percentage")

    gppa_approval = yes_no(row.get("gppa_approval", "no"))
    supplier_registered = yes_no(row.get("supplier_registered", "no"))
    monthly_report = yes_no(row.get("monthly_report_submitted", "no"))

    if amount >= 1_000_000 and not gppa_approval:
        score += 20
        reasons.append("High-value procurement without GPPA approval")

    if method == "rfq" and quotes < 3:
        score += 15
        reasons.append("Low competition: fewer than 3 quotations")

    if amount >= 1_000_000 and tender_days < 30:
        score += 10
        reasons.append("Short tender period for high-value procurement")

    if not supplier_registered:
        score += 10
        reasons.append("Unregistered supplier")

    if not monthly_report:
        score += 10
        reasons.append("Missing monthly report")

    if variation > 5:
        score += 10
        reasons.append("Contract variation above threshold")

    score = min(round(score, 2), 100)

    return score, reasons


def risk_category(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.duplicated()]

    if "procurement_category" not in df.columns:
        df["procurement_category"] = "goods"

    required_columns = [
        "institution",
        "procurement_category",
        "procurement_method",
        "amount",
        "number_of_quotes",
        "tender_days",
        "gppa_approval",
        "supplier_registered",
        "monthly_report_submitted",
        "variation_percentage",
    ]

    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    results = []

    for _, row in df.iterrows():
        compliance_score, compliance_risk_score, flags = check_compliance(row)
        risk_score, risk_reasons = ai_risk_score(row, compliance_risk_score)

        results.append({
            "Compliance Score": compliance_score,
            "AI Risk Score": risk_score,
            "AI Risk Category": risk_category(risk_score),
            "Compliance Flags": "; ".join(flags) if flags else "Compliant",
            "Risk Reasons": "; ".join(risk_reasons) if risk_reasons else "Low risk"
        })

    def generate_pdf_report(df):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
    
        width, height = A4
        y = height - 50
    
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "GPPA Procurement Risk & Compliance Report")
    
        y -= 35
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Total Procurements: {len(df)}")
    
        y -= 20
        high_risk = (df["AI Risk Category"] == "High").sum()
        c.drawString(50, y, f"High Risk Cases: {high_risk}")
    
        y -= 20
        if "Anomaly Flag" in df.columns:
            anomalies = (df["Anomaly Flag"] == "Anomaly").sum()
        else:
            anomalies = 0
        c.drawString(50, y, f"Anomalies Detected: {anomalies}")
    
        y -= 35
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Top 5 Riskiest Procurements")
    
        y -= 25
        c.setFont("Helvetica", 8)
    
        top5 = df.sort_values("AI Risk Score", ascending=False).head(5)
    
        for _, row in top5.iterrows():
            text = (
                f"{row.get('institution', '')} | "
                f"{row.get('procurement_category', '')} | "
                f"Risk Score: {row.get('AI Risk Score', '')} | "
                f"Risk: {row.get('AI Risk Category', '')}"
            )
    
            c.drawString(50, y, text[:110])
            y -= 18
    
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 8)
    
        c.save()
        buffer.seek(0)
        return buffer

    final_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    tab1, tab2, tab3 = st.tabs([
        "📋 Executive Compliance Dashboard",
        "🤖 ML Training & Testing",
        "🔮 Predict New Procurement"
    ])

    with tab1:
        st.markdown("## 📊 GPPA Executive Risk Intelligence Dashboard")
        st.write(
            "Designed for GPPA directors, auditors, and decision-makers. "
            "This section focuses on compliance status, risk exposure, audit priorities, and downloadable reports."
        )
    
        # -----------------------------
        # FILTERS FIRST
        # -----------------------------
        st.subheader("🔎 Filters")
    
        col_a, col_b, col_c = st.columns(3)
    
        with col_a:
            risk_filter = st.selectbox(
                "Filter by Risk Category",
                ["All", "High", "Medium", "Low"]
            )
    
        with col_b:
            category_filter = st.selectbox(
                "Filter by Procurement Category",
                ["All"] + sorted(final_df["procurement_category"].dropna().astype(str).unique().tolist())
            )
    
        with col_c:
            search = st.text_input("Search Institution")
    
        display_df = final_df.copy()
    
        if risk_filter != "All":
            display_df = display_df[display_df["AI Risk Category"] == risk_filter]
    
        if category_filter != "All":
            display_df = display_df[
                display_df["procurement_category"].astype(str) == category_filter
            ]
    
        if search:
            display_df = display_df[
                display_df["institution"].astype(str).str.contains(search, case=False, na=False)
            ]
    
        # -----------------------------
        # KPI CARDS
        # -----------------------------
        total = len(display_df)
        high_risk = (display_df["AI Risk Category"] == "High").sum()
        avg_risk = display_df["AI Risk Score"].mean() if total > 0 else 0
        avg_compliance = display_df["Compliance Score"].mean() if total > 0 else 0
    
        if "Anomaly Flag" in display_df.columns:
            anomalies = (display_df["Anomaly Flag"] == "Anomaly").sum()
        else:
            anomalies = 0
    
        st.subheader("📌 Executive Summary")
    
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Procurements</div>
                <div class="kpi-value">{total}</div>
            </div>
            """, unsafe_allow_html=True)
    
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">High Risk Cases</div>
                <div class="kpi-value">{high_risk}</div>
            </div>
            """, unsafe_allow_html=True)
    
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Anomalies</div>
                <div class="kpi-value">{anomalies}</div>
            </div>
            """, unsafe_allow_html=True)
    
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Risk Score</div>
                <div class="kpi-value">{avg_risk:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
    
        with kpi5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Compliance</div>
                <div class="kpi-value">{avg_compliance:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
        # -----------------------------
        # EXECUTIVE TEXT SUMMARY
        # -----------------------------
        if total > 0:
            high_pct = (high_risk / total) * 100
            medium_pct = (display_df["AI Risk Category"] == "Medium").mean() * 100
    
            highest_risk_inst = (
                display_df.groupby("institution")["AI Risk Score"]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
    
            st.subheader("🧠 Key Executive Insights")
            st.write(f"""
            - **{high_pct:.1f}%** of the selected procurement records are classified as **High Risk**.
            - **{medium_pct:.1f}%** of the selected procurement records are classified as **Medium Risk**.
            - The highest average risk is observed in **{highest_risk_inst}**.
            - Priority review should focus on missing approvals, low competition, short tender periods, unregistered suppliers, and contract variations.
            """)
        else:
            st.warning("No records match the selected filters.")
    
        st.divider()
    
        # -----------------------------
        # BI-STYLE CHARTS
        # -----------------------------
        if total > 0:
            st.markdown("## 📊 GPPA Executive Risk Intelligence Dashboard")
    
            risk_colors = {
                "High": "#d62728",
                "Medium": "#ff7f0e",
                "Low": "#2ca02c"
            }
    
            col1, col2 = st.columns(2)
    
            with col1:
                risk_counts = display_df["AI Risk Category"].value_counts().reset_index()
                risk_counts.columns = ["Risk Category", "Count"]
    
                fig_risk = px.pie(
                    risk_counts,
                    names="Risk Category",
                    values="Count",
                    title="Overall Risk Exposure",
                    hole=0.45,
                    color="Risk Category",
                    color_discrete_map=risk_colors
                )
                fig_risk.update_layout(height=420)
                st.plotly_chart(fig_risk, use_container_width=True)
    
            with col2:
                category_counts = display_df["procurement_category"].value_counts().reset_index()
                category_counts.columns = ["Procurement Category", "Count"]
    
                fig_category = px.pie(
                    category_counts,
                    names="Procurement Category",
                    values="Count",
                    title="Procurement Portfolio Breakdown",
                    hole=0.45
                )
                fig_category.update_layout(height=420)
                st.plotly_chart(fig_category, use_container_width=True)
    
            col3, col4 = st.columns(2)
    
            with col3:
                institution_risk = (
                    display_df.groupby("institution")["AI Risk Score"]
                    .mean()
                    .reset_index()
                    .sort_values("AI Risk Score", ascending=False)
                )
    
                fig_inst = px.bar(
                    institution_risk,
                    x="AI Risk Score",
                    y="institution",
                    orientation="h",
                    title="Average Risk Score by Institution"
                )
                fig_inst.update_layout(
                    height=500,
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig_inst, use_container_width=True)
    
            with col4:
                method_risk = (
                    display_df.groupby("procurement_method")["AI Risk Score"]
                    .mean()
                    .reset_index()
                    .sort_values("AI Risk Score", ascending=False)
                )
    
                fig_method = px.bar(
                    method_risk,
                    x="procurement_method",
                    y="AI Risk Score",
                    title="Average Risk by Procurement Method"
                )
                fig_method.update_layout(height=500)
                st.plotly_chart(fig_method, use_container_width=True)
    
            if "date" in display_df.columns:
                st.subheader("📈 Risk Trend Over Time")
    
                trend_df = display_df.copy()
                trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
    
                trend_df = (
                    trend_df.dropna(subset=["date"])
                    .groupby("date")["AI Risk Score"]
                    .mean()
                    .reset_index()
                    .sort_values("date")
                )
    
                if len(trend_df) > 0:
                    fig_trend = px.line(
                        trend_df,
                        x="date",
                        y="AI Risk Score",
                        title="Average Risk Score Trend Over Time",
                        markers=True
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
    
        # -----------------------------
        # RED FLAGS PANEL
        # -----------------------------
        st.subheader("🚨 Auto Red Flags Panel: Top 5 Riskiest Procurements")
    
        top5_risk = display_df.sort_values("AI Risk Score", ascending=False).head(5)
    
        red_flag_cols = [
            "institution",
            "procurement_category",
            "procurement_method",
            "amount",
            "AI Risk Score",
            "AI Risk Category",
            "Compliance Flags",
            "Risk Reasons",
        ]
    
        red_flag_cols = [c for c in red_flag_cols if c in top5_risk.columns]
    
        if len(top5_risk) > 0:
            st.dataframe(top5_risk[red_flag_cols], use_container_width=True)
        else:
            st.info("No red flags to display for the selected filters.")
    
        # -----------------------------
        # AUDIT RESULTS TABLE
        # -----------------------------
        audit_cols = [
            "institution",
            "procurement_category",
            "procurement_method",
            "amount",
            "number_of_quotes",
            "tender_days",
            "gppa_approval",
            "supplier_registered",
            "monthly_report_submitted",
            "variation_percentage",
            "Compliance Score",
            "AI Risk Score",
            "AI Risk Category",
            "Compliance Flags",
            "Risk Reasons",
        ]
    
        audit_cols = [c for c in audit_cols if c in display_df.columns]
    
        st.subheader("🧾 Compliance Audit Results")
        st.dataframe(display_df[audit_cols], use_container_width=True)
    
        with st.expander("🌍 Real-World Impact"):
            st.write("""
            This system supports:
            - Faster public procurement audit reviews
            - Early detection of risky procurement cases
            - Improved transparency and accountability
            - Evidence-based inspection planning
            - Digital transformation of procurement compliance monitoring
            """)

        
         # -----------------------------
        # 🎯 Explain Selected Procurement 
        # -----------------------------
        st.subheader("🎯 Explain Selected Procurement")
        
        if len(display_df) > 0:
            selected_index = st.selectbox(
                "Select procurement record",
                display_df.index,
                format_func=lambda x: f"{x} - {display_df.loc[x, 'institution']} | {display_df.loc[x, 'AI Risk Category']}"
            )
        
            st.dataframe(display_df.loc[[selected_index]], use_container_width=True)
        
            # ✅ Risk Badge
            risk = display_df.loc[selected_index, "AI Risk Category"]
        
            if risk == "High":
                st.markdown("### 🔴 High Risk Procurement")
            elif risk == "Medium":
                st.markdown("### 🟠 Medium Risk Procurement")
            else:
                st.markdown("### 🟢 Low Risk Procurement")
        
            # Divider
            st.markdown("---")
        
            if st.button("Explain this procurement"):
                selected_row = display_df.loc[selected_index].to_dict()
        
                prompt = f"""
        You are an expert GPPA procurement auditor.
        
        Analyze the procurement record below and produce a professional audit explanation.
        
        IMPORTANT:
        - Highlight critical risks using ⚠️ emoji where necessary
        - Be clear, structured, and concise
        - Use professional audit language
        
        Procurement Data:
        {selected_row}
        
        Provide the explanation in this format:
        
        ### 🧠 Overall Assessment
        
        ### 1. Why it is Risky or Compliant
        - Include strengths (if any)
        - Highlight major risks with ⚠️
        
        ### 2. The Most Important Compliance Issues
        - List key issues ranked by severity
        - Use ⚠️ for critical violations
        
        ### 3. What an Auditor Should Do Next
        - Provide clear investigation steps
        
        ### 4. A Short Recommendation for a GPPA Director
        - Executive-level recommendation (concise)
        """
        
                with st.spinner("Generating AI audit explanation..."):
                    try:
                        response = gemini_model.generate_content(prompt)
        
                        st.markdown("### 🧠 AI Audit Explanation")
        
                        st.markdown(f"""
                        <div style="
                            padding:18px;
                            border-radius:12px;
                            background-color:#111827;
                            border:1px solid #374151;
                            line-height:1.6;
                        ">
                        {response.text}
                        </div>
                        """, unsafe_allow_html=True)
        
                    except Exception as e:
                        st.error("❌ AI explanation failed.")
                        st.exception(e)
        
        else:
            st.info("No procurement records available to explain.")
                
        # -----------------------------
        # 🤖 AI COPILOT
        # -----------------------------
        st.subheader("🤖 GPPA AI Audit Copilot")
        
        if "copilot_messages" not in st.session_state:
            st.session_state.copilot_messages = []
        
        for msg in st.session_state.copilot_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        
        def run_ai_copilot(question, df):
            sample_data = df.sample(min(30, len(df))).to_dict(orient="records")
        
            prompt = f"""You are an expert AI procurement auditor for GPPA.
        
        Analyze the procurement dataset and answer questions about:
        - Compliance risks
        - High-risk procurements
        - Audit priorities
        - Anomalies
        - Recommendations for GPPA directors
        
        Dataset sample:
        {sample_data}
        
        Dataset summary:
        Total records: {len(df)}
        High risk cases: {(df["AI Risk Category"] == "High").sum() if "AI Risk Category" in df.columns else 0}
        Average risk score: {df["AI Risk Score"].mean() if "AI Risk Score" in df.columns else 0:.2f}
        Average compliance score: {df["Compliance Score"].mean() if "Compliance Score" in df.columns else 0:.2f}
        
        User question:
        {question}
        
        Answer clearly like a professional audit analyst.
        Keep the answer concise, executive-friendly, and no longer than 8 bullet points unless the user asks for details.
        """
        
            response = gemini_model.generate_content(prompt)
            return response.text
        
        
        user_question = st.chat_input(
            "Ask the AI copilot about compliance, risk, anomalies, or audit priorities"
        )
        
        if user_question:
            st.session_state.copilot_messages.append({
                "role": "user",
                "content": user_question
            })
        
            with st.chat_message("user"):
                st.markdown(user_question)
        
            with st.chat_message("assistant"):
                with st.spinner("Analyzing procurement data..."):
                    try:
                        answer = run_ai_copilot(user_question, display_df)
                        st.markdown(answer)
                    except Exception as e:
                        answer = f"Gemini error: {e}"
                        st.error(answer)
        
            st.session_state.copilot_messages.append({
                "role": "assistant",
                "content": answer
            })
        
        
        # -----------------------------
        # 📊 AUTO CHARTS FROM QUESTIONS
        # -----------------------------
        if user_question:
            q = user_question.lower()
        
            if "risk distribution" in q or "pie chart" in q:
                chart_df = display_df["AI Risk Category"].value_counts().reset_index()
                chart_df.columns = ["Risk Category", "Count"]
        
                fig = px.pie(
                    chart_df,
                    names="Risk Category",
                    values="Count",
                    title="Risk Distribution",
                    hole=0.45
                )
                st.plotly_chart(fig, use_container_width=True)
        
            elif "institution" in q and "risk" in q:
                chart_df = (
                    display_df.groupby("institution")["AI Risk Score"]
                    .mean()
                    .reset_index()
                    .sort_values("AI Risk Score", ascending=False)
                )
        
                fig = px.bar(
                    chart_df,
                    x="AI Risk Score",
                    y="institution",
                    orientation="h",
                    title="Average Risk by Institution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        
        # -----------------------------
        # 📄 AI REPORT
        # -----------------------------
        st.subheader("📄 AI-Written Executive Report")
        
        if st.button("Generate AI Executive Report"):
            report_sample = (
                display_df.sort_values("AI Risk Score", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )
        
            report_prompt = f"""You are an expert public procurement audit analyst.
        
        Write a concise executive report for GPPA directors based on this procurement data.
        
        Include:
        - Overall risk summary
        - Key compliance weaknesses
        - Top audit priorities
        - Recommended actions
        - Short conclusion
        
        Data sample:
        {report_sample}
        """
        
            with st.spinner("Writing executive report..."):
                try:
                    report_response = gemini_model.generate_content(report_prompt)
                    report_text = report_response.text
        
                    st.markdown(report_text)
        
                    st.download_button(
                        "Download AI Executive Report",
                        report_text,
                        "gppa_ai_executive_report.txt",
                        "text/plain"
                    )
                except Exception:
                    st.warning("AI report generation is temporarily unavailable.")
                
            
    
        # -----------------------------
        # DOWNLOADS
        # -----------------------------
        st.download_button(
            "Download Compliance Audit Report",
            display_df.to_csv(index=False),
            "gppa_compliance_audit_report.csv",
            "text/csv"
        )
    
        pdf_buffer = generate_pdf_report(display_df)
    
        st.download_button(
            label="📄 Download Professional PDF Report",
            data=pdf_buffer,
            file_name="gppa_procurement_risk_report.pdf",
            mime="application/pdf"
        )

    with tab2:
        st.subheader("🤖 Advanced Machine Learning Training & Testing")
    
        ml_features = [
            "procurement_category",
            "procurement_method",
            "amount",
            "number_of_quotes",
            "tender_days",
            "gppa_approval",
            "supplier_registered",
            "monthly_report_submitted",
            "variation_percentage",
        ]
    
        X = final_df[ml_features]
        y = final_df["AI Risk Category"]
    
        st.subheader("📊 Risk Category Distribution Before SMOTE")
        before = y.value_counts().reset_index()
        before.columns = ["Risk Category", "Count"]
        st.dataframe(before, use_container_width=True)
    
        if y.nunique() < 2:
            st.warning("At least two risk classes are required to train the ML model.")
        else:
            categorical_features = [
                "procurement_category",
                "procurement_method",
                "gppa_approval",
                "supplier_registered",
                "monthly_report_submitted",
            ]
    
            numeric_features = [
                "amount",
                "number_of_quotes",
                "tender_days",
                "variation_percentage",
            ]
    
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ("num", "passthrough", numeric_features),
                ]
            )
    
            X_processed = preprocessor.fit_transform(X)
    
            min_count = y.value_counts().min()
    
            if min_count > 1:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
                X_res, y_res = smote.fit_resample(X_processed, y)
    
                st.subheader("📊 Risk Category Distribution After SMOTE")
                after = pd.Series(y_res).value_counts().reset_index()
                after.columns = ["Risk Category", "Count"]
                st.dataframe(after, use_container_width=True)
            else:
                st.warning("SMOTE skipped because one class has only one record.")
                X_res, y_res = X_processed, y
    
            X_train, X_test, y_train, y_test = train_test_split(
                X_res,
                y_res,
                test_size=0.25,
                random_state=42,
                stratify=y_res if pd.Series(y_res).nunique() > 1 else None
            )
    
            st.subheader("🔧 Hyperparameter Tuning with GridSearchCV")
    
            rf_grid = {
                "n_estimators": [100, 300],
                "max_depth": [6, 8, None],
                "min_samples_leaf": [1, 2],
                "class_weight": ["balanced_subsample"],
            }
    
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1
            )
    
            grid_search.fit(X_train, y_train)
    
            best_rf = grid_search.best_estimator_
    
            st.write("Best Random Forest Parameters:")
            st.json(grid_search.best_params_)
    
            st.subheader("🏆 Model Comparison")
    
            models = {
                "Tuned Random Forest": best_rf,
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
            }
    
            comparison = []
    
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
    
                comparison.append({
                    "Model": name,
                    "Accuracy": round(acc, 4)
                })
    
            comparison_df = pd.DataFrame(comparison).sort_values("Accuracy", ascending=False)
    
            st.dataframe(comparison_df, use_container_width=True)
            st.bar_chart(comparison_df.set_index("Model"))
    
            best_model_name = comparison_df.iloc[0]["Model"]
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
    
            st.success(f"Best model selected: {best_model_name}")
    
            st.subheader("✅ Cross-Validation Score")
    
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
            cv_scores = cross_val_score(
                best_model,
                X_res,
                y_res,
                cv=cv,
                scoring="accuracy"
            )
    
            st.write(f"Cross-validation accuracy: **{cv_scores.mean() * 100:.2f}%**")
            st.write(f"Standard deviation: **{cv_scores.std() * 100:.2f}%**")
    
            pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, pred)
    
            st.metric("Test Accuracy", f"{acc * 100:.2f}%")
    
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, pred, labels=best_model.classes_)
            cm_df = pd.DataFrame(cm, index=best_model.classes_, columns=best_model.classes_)
            st.dataframe(cm_df, use_container_width=True)
    
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    
            final_df["ML Predicted Risk"] = best_model.predict(X_processed)
    
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(X_processed)
                prob_df = pd.DataFrame(
                    probs,
                    columns=[f"Probability_{label}" for label in best_model.classes_]
                )
    
                final_df = pd.concat(
                    [final_df.reset_index(drop=True), prob_df.reset_index(drop=True)],
                    axis=1
                )
    
                final_df["Prediction Confidence"] = probs.max(axis=1).round(3)
    
            st.subheader("🚨 Anomaly Detection")
    
            anomaly_model = IsolationForest(
                contamination=0.08,
                random_state=42
            )
    
            anomaly_labels = anomaly_model.fit_predict(X_processed)
    
            final_df["Anomaly Flag"] = [
                "Anomaly" if label == -1 else "Normal" for label in anomaly_labels
            ]
    
            anomaly_count = (final_df["Anomaly Flag"] == "Anomaly").sum()
    
            st.metric("Detected Anomalies", anomaly_count)
    
            st.dataframe(
                final_df[
                    [
                        "institution",
                        "procurement_category",
                        "procurement_method",
                        "amount",
                        "AI Risk Category",
                        "ML Predicted Risk",
                        "Anomaly Flag",
                    ]
                ],
                use_container_width=True
            )
    
            st.subheader("📈 Feature Importance")
    
            encoded_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
                categorical_features
            )
    
            feature_names = list(encoded_names) + numeric_features
    
            if hasattr(best_model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": best_model.feature_importances_
                }).sort_values("Importance", ascending=False)
    
                st.bar_chart(importance_df.set_index("Feature").head(10))
    
            st.subheader("💾 Model Persistence")
    
            model_package = {
                "model": best_model,
                "preprocessor": preprocessor,
                "features": ml_features,
                "classes": list(best_model.classes_),
            }
    
            joblib.dump(model_package, "gppa_saved_ml_pipeline.pkl")
    
            model_buffer = io.BytesIO()
            joblib.dump(model_package, model_buffer)
            model_buffer.seek(0)
    
            st.download_button(
                "Download Saved ML Pipeline",
                model_buffer,
                "gppa_saved_ml_pipeline.pkl",
                "application/octet-stream"
            )

            st.subheader("🎯 Filter ML Results")

            risk_filter = st.multiselect(
                "Select Risk Levels",
                options=final_df["AI Risk Category"].unique(),
                default=list(final_df["AI Risk Category"].unique())
            )
            
            anomaly_filter = st.selectbox(
                "Anomaly Filter",
                ["All", "Only Anomalies", "Only Normal"]
            )
            
            filtered_df = final_df[final_df["AI Risk Category"].isin(risk_filter)]
            
            if "Anomaly Flag" in filtered_df.columns:
                if anomaly_filter == "Only Anomalies":
                    filtered_df = filtered_df[filtered_df["Anomaly Flag"] == "Anomaly"]
                elif anomaly_filter == "Only Normal":
                    filtered_df = filtered_df[filtered_df["Anomaly Flag"] == "Normal"]
    
            st.subheader("🧠 ML Prediction Results")
    
            ml_cols = [
                "institution",
                "procurement_category",
                "procurement_method",
                "amount",
                "AI Risk Category",
                "ML Predicted Risk",
                "Prediction Confidence",
                "Anomaly Flag",
            ]
    
            ml_cols += [c for c in final_df.columns if c.startswith("Probability_")]
            ml_cols = [c for c in ml_cols if c in final_df.columns]
    
            st.dataframe(filtered_df[ml_cols], use_container_width=True)

            st.subheader("🔍 Inspect Individual Procurement")

            if len(filtered_df) > 0:
                selected_index = st.selectbox(
                    "Select a procurement to inspect",
                    filtered_df.index,
                    format_func=lambda x: f"{x} - {filtered_df.loc[x, 'institution']} ({filtered_df.loc[x, 'procurement_method']})"
                )
            
                st.dataframe(filtered_df.loc[[selected_index]], use_container_width=True)
            else:
                st.warning("No records match the selected filters.")
    
            st.download_button(
                "Download Advanced ML Prediction Report",
                final_df.to_csv(index=False),
                "gppa_advanced_ml_prediction_report.csv",
                "text/csv"
            )

    with tab3:
        st.subheader("🔮 Predict New Procurement Risk")
        st.write("Use this form to estimate risk for a new procurement case.")

        with st.form("new_procurement_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_institution = st.text_input("Institution", "Ministry of Health")
                new_category = st.selectbox("Procurement Category", ["goods", "services", "complex_works"])
                new_method = st.selectbox("Procurement Method", ["rfq", "open_tender", "restricted", "international_tender", "single_source"])
                new_amount = st.number_input("Amount", min_value=0.0, value=500000.0)
                new_quotes = st.number_input("Number of Quotations", min_value=0, value=3)
                new_tender_days = st.number_input("Tender Days", min_value=0, value=30)

            with col2:
                new_gppa = st.selectbox("GPPA Approval", ["yes", "no"])
                new_supplier = st.selectbox("Supplier Registered", ["yes", "no"])
                new_monthly = st.selectbox("Monthly Report Submitted", ["yes", "no"])
                new_variation = st.number_input("Variation Percentage", min_value=0.0, value=0.0)
                new_technical = st.selectbox("Technical Proposal Submitted", ["yes", "no"])
                new_financial = st.selectbox("Financial Proposal Submitted", ["yes", "no"])
                new_site_visit = st.selectbox("Site Visit Done", ["yes", "no"])
                new_perf_security = st.selectbox("Performance Security Submitted", ["yes", "no"])

            submitted = st.form_submit_button("Assess New Procurement Risk")

        if submitted:
            new_row = pd.Series({
                "institution": new_institution,
                "procurement_category": new_category,
                "procurement_method": new_method,
                "amount": new_amount,
                "number_of_quotes": new_quotes,
                "tender_days": new_tender_days,
                "gppa_approval": new_gppa,
                "supplier_registered": new_supplier,
                "monthly_report_submitted": new_monthly,
                "variation_percentage": new_variation,
                "technical_proposal": new_technical,
                "financial_proposal": new_financial,
                "site_visit_done": new_site_visit,
                "performance_security": new_perf_security,
                "tor_attached": new_technical,
                "technical_director_assigned": "yes",
                "essential_equipment_available": "yes",
                "bid_security_required": "no",
                "bid_security_submitted": "no",
                "inspection_certificate_required": "no",
                "inspection_certificate_submitted": "no"
            })

            comp_score, comp_risk, flags = check_compliance(new_row)
            risk_score, reasons = ai_risk_score(new_row, comp_risk)

            col1, col2, col3 = st.columns(3)
            col1.metric("Compliance Score", f"{comp_score:.2f}%")
            col2.metric("AI Risk Score", f"{risk_score:.2f}")
            col3.metric("Risk Category", risk_category(risk_score))

            st.write("Compliance Flags:", "; ".join(flags) if flags else "Compliant")
            st.write("Risk Reasons:", "; ".join(reasons) if reasons else "Low risk")

else:
    st.info("Upload a CSV or Excel file to begin.")

# ---------------- FOOTER ----------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:30px; padding:18px; border-radius:16px; background:linear-gradient(180deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9)); border:1px solid rgba(59,130,246,0.25);">

<div style="font-size:1.05rem; color:#e5e7eb; margin-bottom:6px;">
Built by <strong style="color:#3b82f6;">Abdoulie J Bah</strong> 🚀
</div>

<div style="font-size:0.9rem; color:#94a3b8; margin-bottom:12px;">
AI Engineer • Data Scientist • Business Intelligence Developer
</div>

<div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">

<a href="https://www.linkedin.com/in/abdoulie-j-bah-b71263244" target="_blank" style="text-decoration:none; padding:8px 14px; border-radius:10px; background:#0ea5e9; color:white; font-weight:600;">LinkedIn</a>

<a href="https://github.com/AbdoulieJBah/ai-business-intelligence-dashboard" target="_blank" style="text-decoration:none; padding:8px 14px; border-radius:10px; background:#1f2937; color:white; font-weight:600; border:1px solid rgba(255,255,255,0.1);">GitHub</a>

<a href="mailto:21722285bah@gmail.com" style="text-decoration:none; padding:8px 14px; border-radius:10px; background:#2563eb; color:white; font-weight:600;">Contact</a>

</div>
</div>
""", unsafe_allow_html=True)
