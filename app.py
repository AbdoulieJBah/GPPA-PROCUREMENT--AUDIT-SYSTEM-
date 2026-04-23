import streamlit as st
import pandas as pd

st.set_page_config(page_title="GPPA Compliance Audit MVP", layout="wide")

st.title("GPPA Procurement Compliance & Audit System")
st.write("Upload procurement data and automatically check compliance risks based on GPPA rules.")

uploaded_file = st.file_uploader("Upload procurement Excel/CSV file", type=["xlsx", "csv"])

def check_compliance(row):
    flags = []
    total_rules = 0
    passed_rules = 0

    amount = row.get("amount", 0)
    method = str(row.get("procurement_method", "")).lower()
    quotes = row.get("number_of_quotes", 0)
    tender_days = row.get("tender_days", 0)
    gppa_approval = str(row.get("gppa_approval", "")).lower() == "yes"
    monthly_report = str(row.get("monthly_report_submitted", "")).lower() == "yes"
    supplier_registered = str(row.get("supplier_registered", "")).lower() == "yes"
    variation_percent = row.get("variation_percentage", 0)

    # Rule 1: Single source threshold
    total_rules += 1
    if method == "single_source" and amount > 10000:
        flags.append("Single source exceeds D10,000 threshold for goods/services")
    else:
        passed_rules += 1

    # Rule 2: RFQ requires 3 quotations
    total_rules += 1
    if method == "rfq" and quotes < 3:
        flags.append("RFQ has fewer than 3 quotations")
    else:
        passed_rules += 1

    # Rule 3: GPPA approval required from D1,000,000
    total_rules += 1
    if amount >= 1000000 and not gppa_approval:
        flags.append("GPPA approval missing for procurement ≥ D1,000,000")
    else:
        passed_rules += 1

    # Rule 4: International competitive bidding above D3,000,000
    total_rules += 1
    if amount > 3000000 and method not in ["international_tender", "open_tender"]:
        flags.append("High-value procurement may require international/open tendering")
    else:
        passed_rules += 1

    # Rule 5: Tender period should be at least 30 days for high-value tenders
    total_rules += 1
    if amount >= 1000000 and tender_days < 30:
        flags.append("Tender submission period is less than 30 days")
    else:
        passed_rules += 1

    # Rule 6: Supplier must be registered
    total_rules += 1
    if not supplier_registered:
        flags.append("Supplier is not GPPA registered")
    else:
        passed_rules += 1

    # Rule 7: Monthly report required
    total_rules += 1
    if not monthly_report:
        flags.append("Monthly procurement report not submitted")
    else:
        passed_rules += 1

    # Rule 8: Contract variation above 5%
    total_rules += 1
    if variation_percent > 5 and not gppa_approval:
        flags.append("Contract variation above 5% without required approval")
    else:
        passed_rules += 1

    score = round((passed_rules / total_rules) * 100, 2)
    risk_level = "Low" if score >= 80 else "Medium" if score >= 50 else "High"

    return pd.Series({
        "compliance_score": score,
        "risk_level": risk_level,
        "flags": "; ".join(flags) if flags else "Compliant"
    })

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    results = df.apply(check_compliance, axis=1)
    final_df = pd.concat([df, results], axis=1)

    st.subheader("Compliance Results")
    st.dataframe(final_df)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Procurements", len(final_df))
    col2.metric("Average Compliance Score", f"{final_df['compliance_score'].mean():.2f}%")
    col3.metric("High Risk Cases", (final_df["risk_level"] == "High").sum())

    st.download_button(
        "Download Compliance Report",
        final_df.to_csv(index=False),
        "gppa_compliance_report.csv",
        "text/csv"
    )
else:
    st.info("Upload a CSV or Excel file to begin.")