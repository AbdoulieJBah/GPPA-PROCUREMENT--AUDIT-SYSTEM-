import streamlit as st
import pandas as pd
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="GPPA Procurement Compliance & ML Risk System",
    layout="wide"
)

st.title("🤖 AI/ML-Powered GPPA Procurement Risk Detection System")
st.write(
    "A two-section system: one section for GPPA compliance audit reporting, "
    "and another section for ML model training, testing, and recruiter-facing technical evaluation."
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

    tab1, tab2 = st.tabs([
        "📋 Compliance Audit Report",
        "🤖 Machine Learning Model"
    ])

    with tab1:
        st.subheader("📋 GPPA Compliance Audit Report")
        st.write(
            "This section is designed for GPPA directors, auditors, and public-sector decision makers. "
            "It focuses on compliance status, risk flags, and downloadable audit reporting."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Procurements", len(final_df))
        col2.metric("Average Compliance Score", f"{final_df['compliance_score'].mean():.2f}%")
        col3.metric("Average Risk Score", f"{final_df['AI Risk Score'].mean():.2f}")
        col4.metric("High Risk Cases", (final_df["AI Risk Category"] == "High").sum())

        st.divider()

        st.subheader("🔍 Audit Filters")

        col_a, col_b = st.columns(2)

        with col_a:
            risk_filter = st.selectbox(
                "Filter by Risk Category",
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

        st.subheader("🧾 Compliance Results")
        audit_columns = [
            "institution",
            "procurement_method",
            "amount",
            "number_of_quotes",
            "tender_days",
            "gppa_approval",
            "supplier_registered",
            "monthly_report_submitted",
            "variation_percentage",
            "compliance_score",
            "AI Risk Score",
            "AI Risk Category",
            "compliance_flags",
            "ai_risk_reasons"
        ]

        available_audit_columns = [col for col in audit_columns if col in display_df.columns]
        st.dataframe(display_df[available_audit_columns], use_container_width=True)

        st.subheader("📊 Risk Distribution")
        risk_counts = final_df["AI Risk Category"].value_counts()
        st.bar_chart(risk_counts)

        st.subheader("📈 Compliance Score by Institution")
        compliance_by_inst = (
            final_df.groupby("institution")["compliance_score"]
            .mean()
            .sort_values()
        )
        st.bar_chart(compliance_by_inst)

        st.subheader("🚨 Top 10 Highest Risk Procurements")
        top_risk = final_df.sort_values(by="AI Risk Score", ascending=False).head(10)
        st.dataframe(top_risk[available_audit_columns], use_container_width=True)

        with st.expander("📌 How to interpret this audit report"):
            st.write("""
            This report helps identify procurement cases that may require further review.

            Key indicators:
            - **Compliance Score**: Percentage of GPPA-based checks passed.
            - **AI Risk Score**: Weighted risk score based on procurement red flags.
            - **Risk Category**: Low, Medium, or High.
            - **Compliance Flags**: Specific issues detected in the procurement record.

            This section is designed to be simple and useful for public-sector decision makers.
            """)

        csv = display_df.to_csv(index=False)

        st.download_button(
            label="Download Compliance Audit Report",
            data=csv,
            file_name="gppa_compliance_audit_report.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("🤖 Machine Learning Model Training & Testing")
        st.write(
            "This section is designed for technical reviewers and recruiters. "
            "It shows model training, class imbalance handling using SMOTE, evaluation metrics, "
            "feature importance, and downloadable trained model."
        )

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

        st.subheader("📊 Risk Category Distribution Before SMOTE")
        before_smote = y.value_counts().reset_index()
        before_smote.columns = ["Risk Category", "Count"]
        st.dataframe(before_smote, use_container_width=True)

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

            X_processed = preprocessor.fit_transform(X)

            min_class_count = y.value_counts().min()

            if min_class_count > 1:
                k_neighbors = min(5, min_class_count - 1)

                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors
                )

                X_resampled, y_resampled = smote.fit_resample(X_processed, y)

                st.subheader("📊 Risk Category Distribution After SMOTE")
                after_smote = pd.Series(y_resampled).value_counts().reset_index()
                after_smote.columns = ["Risk Category", "Count"]
                st.dataframe(after_smote, use_container_width=True)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled,
                    y_resampled,
                    test_size=0.25,
                    random_state=42,
                    stratify=y_resampled
                )
            else:
                st.warning(
                    "SMOTE was skipped because at least one risk category has only one record. "
                    "Add more examples for each risk class to use SMOTE."
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=y if y.nunique() > 1 else None
                )

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_depth=8
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.metric("ML Model Accuracy", f"{accuracy * 100:.2f}%")

            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            final_df["ML Predicted Risk"] = model.predict(X_processed)

            risk_probabilities = model.predict_proba(X_processed)
            prob_df = pd.DataFrame(
                risk_probabilities,
                columns=[f"Probability_{label}" for label in model.classes_]
            )

            final_df = pd.concat(
                [final_df.reset_index(drop=True), prob_df.reset_index(drop=True)],
                axis=1
            )

            final_df["Prediction Confidence"] = risk_probabilities.max(axis=1).round(3)

            encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
                categorical_features
            )

            feature_names = list(encoded_cat_names) + numeric_features

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.subheader("📈 Top ML Feature Importance")
            st.bar_chart(importance_df.set_index("Feature").head(10))

            st.subheader("🧠 ML Prediction Results")
            ml_result_columns = [
                "institution",
                "procurement_method",
                "amount",
                "AI Risk Category",
                "ML Predicted Risk",
                "Prediction Confidence",
            ]

            probability_cols = [col for col in final_df.columns if col.startswith("Probability_")]
            ml_result_columns = ml_result_columns + probability_cols
            available_ml_cols = [col for col in ml_result_columns if col in final_df.columns]

            st.dataframe(final_df[available_ml_cols], use_container_width=True)

            with st.expander("📌 What does this ML model do?"):
                st.write("""
                This model uses a Random Forest classifier trained on procurement records.

                It uses:
                - Procurement method
                - Procurement amount
                - Number of quotations
                - Tender period
                - GPPA approval status
                - Supplier registration status
                - Monthly reporting status
                - Contract variation percentage

                SMOTE is used to address class imbalance by synthetically increasing the minority risk categories.
                The model then predicts whether a procurement case is Low, Medium, or High risk.
                """)

            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)

            st.download_button(
                label="Download Trained ML Model",
                data=model_buffer,
                file_name="gppa_procurement_risk_model.pkl",
                mime="application/octet-stream"
            )

            ml_csv = final_df.to_csv(index=False)

            st.download_button(
                label="Download ML Prediction Report",
                data=ml_csv,
                file_name="gppa_ml_prediction_report.csv",
                mime="text/csv"
            )

else:
    st.info("Upload a CSV or Excel file to begin.")
