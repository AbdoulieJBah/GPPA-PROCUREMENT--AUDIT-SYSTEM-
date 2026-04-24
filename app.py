import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="GPPA Procurement Compliance & Audit System",
    layout="wide"
)

st.title("📊 AI-Powered Procurement Compliance & Risk Intelligence System")
st.write(
    "Upload procurement data and automatically detect compliance issues, risk patterns, and audit red flags based on GPPA procurement rules."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload procurement Excel/CSV file",
    type=["xlsx", "csv"]
)


def yes_no(value):
    return str(value).strip().lower() in ["yes", "true", "1"]


def check_compliance(row):
    flags = []
    total_rules = 0
    passed_rules = 0

    amount = float(row.get("amount", 0))
    method = str(row.get("procurement_method", "")).strip().lower()
    quotes = int(row.get("number_of_quotes", 0))
    tender_days = int(row.get("tender_days", 0))
    gppa_approval = yes_no(row.get("gppa_approval", "no"))
    supplier_registered = yes_no(row.get("supplier_registered", "no"))
    monthly_report = yes_no(row.get("monthly_report_submitted", "no"))
    variation_percent = float(row.get("variation_percentage", 0))

    total_rules += 1
    if method == "rfq" and quotes < 3:
        flags.append("RFQ has fewer than 3 quotations")
    else:
        passed_rules += 1

    total_rules += 1
    if amount >= 1_000_000 and not gppa_approval:
        flags.append("GPPA approval missing for procurement ≥ D1,000,000")
    else:
        passed_rules += 1

    total_rules += 1
    if amount > 3_000_000 and method not in ["open_tender", "international_tender"]:
        flags.append("High-value procurement may require open/international tendering")
    else:
        passed_rules += 1

    total_rules += 1
    if amount >= 1_000_000 and tender_days < 30:
        flags.append("Tender submission period is less than 30 days")
    else:
        passed_rules += 1

    total_rules += 1
    if not supplier_registered:
        flags.append("Supplier is not GPPA registered")
    else:
        passed_rules += 1

    total_rules += 1
    if not monthly_report:
        flags.append("Monthly procurement report not submitted")
    else:
        passed_rules += 1

    total_rules += 1
    if variation_percent > 5:
        flags.append("Contract variation above 5%")
    else:
        passed_rules += 1

    compliance_score = round((passed_rules / total_rules) * 100, 2)
    compliance_risk_score = 100 - compliance_score

    return compliance_score, compliance_risk_score, flags


def calculate_ai_risk(row):
    score = 0
    reasons = []

    amount = float(row.get("amount", 0))
    method = str(row.get("procurement_method", "")).strip().lower()
    quotes = int(row.get("number_of_quotes", 0))
    tender_days = int(row.get("tender_days", 0))
    gppa_approval = yes_no(row.get("gppa_approval", "no"))
    supplier_registered = yes_no(row.get("supplier_registered", "no"))
    monthly_report = yes_no(row.get("monthly_report_submitted", "no"))
    variation_percent = float(row.get("variation_percentage", 0))

    if amount >= 1_000_000 and not gppa_approval:
        score += 30
        reasons.append("Missing GPPA approval for high-value procurement")

    if method == "rfq" and quotes < 3:
        score += 20
        reasons.append("RFQ has fewer than 3 quotations")

    if amount >= 1_000_000 and tender_days < 30:
        score += 15
        reasons.append("Tender period below 30 days")

    if not supplier_registered:
        score += 15
        reasons.append("Supplier not registered")

    if not monthly_report:
        score += 10
        reasons.append("Monthly report not submitted")

    if variation_percent > 5:
        score += 10
        reasons.append("Contract variation above 5%")

    return min(score, 100), reasons


def final_risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.duplicated()]

    required_columns = [
        "institution",
        "procurement_method",
        "amount",
        "number_of_quotes",
        "tender_days",
        "gppa_approval",
        "supplier_registered",
        "monthly_report_submitted",
        "variation_percentage",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    results = []

    for _, row in df.iterrows():
        compliance_score, compliance_risk_score, flags = check_compliance(row)
        ai_risk_score, ai_reasons = calculate_ai_risk(row)

        final_score = round((ai_risk_score * 0.6) + (compliance_risk_score * 0.4), 2)
        final_level = final_risk_level(final_score)

        results.append({
            "compliance_score": compliance_score,
            "AI Risk Score": final_score,
            "AI Risk Category": final_level,
            "compliance_flags": "; ".join(flags) if flags else "Compliant",
            "ai_risk_reasons": "; ".join(ai_reasons) if ai_reasons else "Low risk"
        })

    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    st.subheader("📌 Dashboard Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Procurements", len(final_df))
    col2.metric("Average Compliance Score", f"{final_df['compliance_score'].mean():.2f}%")
    col3.metric("Average AI Risk Score", f"{final_df['AI Risk Score'].mean():.2f}")
    col4.metric("High Risk Cases", (final_df["AI Risk Category"] == "High").sum())

    st.divider()

    st.subheader("🔍 Filters")

    col_a, col_b = st.columns(2)

    with col_a:
        risk_filter = st.selectbox(
            "Filter by AI Risk Category",
            ["All", "High", "Medium", "Low"]
        )

    with col_b:
        search = st.text_input("Search institution")

    display_df = final_df.copy()

    if risk_filter != "All":
        display_df = display_df[display_df["AI Risk Category"] == risk_filter]

    if search:
        display_df = display_df[
            display_df["institution"].astype(str).str.contains(search, case=False, na=False)
        ]

    st.subheader("📁 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧠 Compliance & AI Risk Results")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("📊 Risk Distribution")
    risk_counts = final_df["AI Risk Category"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("📈 Compliance Score by Institution")
    compliance_by_inst = final_df.groupby("institution")["compliance_score"].mean().sort_values()
    st.bar_chart(compliance_by_inst)

    st.subheader("🚨 Top 10 Highest Risk Procurements")
    top_risk = final_df.sort_values(by="AI Risk Score", ascending=False).head(10)
    st.dataframe(top_risk, use_container_width=True)

    with st.expander("📌 What does the AI Risk Score mean?"):
        st.write("""
        The AI Risk Score is a hybrid risk indicator combining:
        
        - Rule-based compliance checks based on GPPA procurement logic
        - Weighted risk scoring for audit red flags
        
        Higher scores indicate higher procurement risk based on:
        - Missing GPPA approval
        - Low competition
        - Short tender periods
        - Unregistered suppliers
        - Missing monthly reports
        - Contract variations above 5%
        """)

    csv = display_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered Risk Report",
        data=csv,
        file_name="filtered_gppa_ai_risk_report.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV or Excel file to begin.")
