import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="GPPA Procurement Compliance & Audit System",
    layout="wide"
)

st.title("GPPA Procurement Compliance & Audit System")
st.write(
    "Upload procurement data and automatically check compliance risks based on GPPA procurement rules."
)

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

    # Rule 1: RFQ requires at least 3 quotations
    total_rules += 1
    if method == "rfq" and quotes < 3:
        flags.append("RFQ has fewer than 3 quotations")
    else:
        passed_rules += 1

    # Rule 2: GPPA approval required for procurement >= D1,000,000
    total_rules += 1
    if amount >= 1_000_000 and not gppa_approval:
        flags.append("GPPA approval missing for procurement ≥ D1,000,000")
    else:
        passed_rules += 1

    # Rule 3: High-value procurement should use open/international tendering
    total_rules += 1
    if amount > 3_000_000 and method not in ["open_tender", "international_tender"]:
        flags.append("High-value procurement may require open/international tendering")
    else:
        passed_rules += 1

    # Rule 4: Tender period should be at least 30 days for high-value procurement
    total_rules += 1
    if amount >= 1_000_000 and tender_days < 30:
        flags.append("Tender submission period is less than 30 days")
    else:
        passed_rules += 1

    # Rule 5: Supplier registration
    total_rules += 1
    if not supplier_registered:
        flags.append("Supplier is not GPPA registered")
    else:
        passed_rules += 1

    # Rule 6: Monthly report submission
    total_rules += 1
    if not monthly_report:
        flags.append("Monthly procurement report not submitted")
    else:
        passed_rules += 1

    # Rule 7: Contract variation above 5%
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

    score = min(score, 100)

    return score, reasons


def final_risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

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
            "ai_risk_score": ai_risk_score,
            "final_risk_score": final_score,
            "final_risk_level": final_level,
            "compliance_flags": "; ".join(flags) if flags else "Compliant",
            "ai_risk_reasons": "; ".join(ai_reasons) if ai_reasons else "Low risk"
        })

    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)

    st.subheader("Dashboard Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Procurements", len(final_df))
    col2.metric("Average Compliance Score", f"{final_df['compliance_score'].mean():.2f}%")
    col3.metric("Average Final Risk Score", f"{final_df['final_risk_score'].mean():.2f}")
    col4.metric("High Risk Cases", (final_df["final_risk_level"] == "High").sum())

    st.divider()

    st.subheader("Risk Filter")

    risk_filter = st.selectbox(
        "Filter by final risk level",
        ["All", "High", "Medium", "Low"]
    )

    if risk_filter != "All":
        display_df = final_df[final_df["final_risk_level"] == risk_filter]
    else:
        display_df = final_df

    st.subheader("Uploaded Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Compliance & AI Risk Results")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Risk Distribution")
    risk_counts = final_df["final_risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    st.bar_chart(risk_counts.set_index("Risk Level"))

    st.subheader("Top 10 Highest Risk Procurements")
    top_risk = final_df.sort_values(by="final_risk_score", ascending=False).head(10)
    st.dataframe(top_risk, use_container_width=True)

    csv = final_df.to_csv(index=False)

    st.download_button(
        label="Download Full Compliance Report",
        data=csv,
        file_name="gppa_compliance_ai_risk_report.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV or Excel file to begin.")
