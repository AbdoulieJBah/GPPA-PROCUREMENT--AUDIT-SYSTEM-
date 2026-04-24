import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="GPPA Procurement Compliance & ML Risk System",
    layout="wide"
)

st.title("🤖 AI/ML-Powered GPPA Procurement Risk Detection System")
st.write(
    "Upload procurement data to detect compliance issues, calculate AI risk scores, "
    "and train a real Machine Learning model to predict procurement risk."
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

    st.subheader("🤖 Machine Learning Risk Model")

    ml_features = [
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

    if y.nunique() < 2:
        st.warning("ML model needs at least two risk categories to train.")
    else:
        categorical_features = [
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

        model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    max_depth=8
)

        ml_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() > 1 else None
        )

        ml_pipeline.fit(X_train, y_train)

        y_pred = ml_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        final_df["ML Predicted Risk"] = ml_pipeline.predict(X)

        st.metric("ML Model Accuracy", f"{accuracy * 100:.2f}%")

        st.write("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        fitted_preprocessor = ml_pipeline.named_steps["preprocessor"]
        fitted_model = ml_pipeline.named_steps["model"]

        encoded_cat_names = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        feature_names = list(encoded_cat_names) + numeric_features

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": fitted_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Top ML Feature Importance")
        st.bar_chart(feature_importance.set_index("Feature").head(10))

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

    st.subheader("🧠 Compliance, AI Risk & ML Results")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("📊 AI Risk Distribution")
    risk_counts = final_df["AI Risk Category"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("📈 Compliance Score by Institution")
    compliance_by_inst = final_df.groupby("institution")["compliance_score"].mean().sort_values()
    st.bar_chart(compliance_by_inst)

    st.subheader("🚨 Top 10 Highest Risk Procurements")
    top_risk = final_df.sort_values(by="AI Risk Score", ascending=False).head(10)
    st.dataframe(top_risk, use_container_width=True)

    with st.expander("📌 What does the ML model do?"):
        st.write("""
        The ML model is a Random Forest classifier trained on the uploaded procurement records.
        
        It learns patterns from procurement features such as:
        - Procurement method
        - Procurement amount
        - Number of quotations
        - Tender period
        - GPPA approval status
        - Supplier registration status
        - Monthly reporting status
        - Contract variation percentage
        
        The model predicts whether a procurement case is Low, Medium, or High risk.
        """)

    csv = display_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered ML Risk Report",
        data=csv,
        file_name="filtered_gppa_ml_risk_report.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV or Excel file to begin.")
