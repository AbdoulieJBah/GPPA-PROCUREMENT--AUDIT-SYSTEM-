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


st.set_page_config(
    page_title="GPPA Advanced AI Procurement Risk System",
    layout="wide"
)

st.title("🚀 Advanced AI-Powered GPPA Procurement Compliance & Risk System")
st.write(
    "Advanced GovTech prototype combining GPPA compliance rules, AI-style risk scoring, "
    "SMOTE class balancing, model comparison, prediction confidence, and executive audit reporting."
)

uploaded_file = st.file_uploader(
    "Upload procurement data file",
    type=["csv", "xlsx"]
)


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

    final_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    tab1, tab2, tab3 = st.tabs([
        "📋 Executive Compliance Dashboard",
        "🤖 ML Training & Testing",
        "🔮 Predict New Procurement"
    ])

    with tab1:
        st.subheader("📋 Executive Compliance Dashboard")
        st.write(
            "Designed for GPPA directors, auditors, and decision-makers. "
            "This section focuses on compliance, risk, and audit priorities."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Procurements", len(final_df))
        col2.metric("Average Compliance", f"{final_df['Compliance Score'].mean():.2f}%")
        col3.metric("Average AI Risk", f"{final_df['AI Risk Score'].mean():.2f}")
        col4.metric("High Risk Cases", (final_df["AI Risk Category"] == "High").sum())

        high_pct = (final_df["AI Risk Category"] == "High").mean() * 100
        medium_pct = (final_df["AI Risk Category"] == "Medium").mean() * 100
        highest_risk_inst = final_df.groupby("institution")["AI Risk Score"].mean().sort_values(ascending=False).index[0]

        st.subheader("🧠 Executive Summary")
        st.write(f"""
        - **{high_pct:.1f}%** of procurement records are classified as **High Risk**.
        - **{medium_pct:.1f}%** are classified as **Medium Risk**.
        - The highest average risk is observed in **{highest_risk_inst}**.
        - Priority review should focus on missing approvals, low competition, short tender periods, unregistered suppliers, and contract variations.
        """)

        st.divider()

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            risk_filter = st.selectbox("Filter by Risk Category", ["All", "High", "Medium", "Low"])

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
            display_df = display_df[display_df["procurement_category"].astype(str) == category_filter]

        if search:
            display_df = display_df[
                display_df["institution"].astype(str).str.contains(search, case=False, na=False)
            ]

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

        st.subheader("📊 Risk Distribution")
        st.bar_chart(final_df["AI Risk Category"].value_counts())

        st.subheader("📊 Average Risk by Procurement Category")
        st.bar_chart(final_df.groupby("procurement_category")["AI Risk Score"].mean().sort_values())

        st.subheader("📊 Average Risk by Procurement Method")
        st.bar_chart(final_df.groupby("procurement_method")["AI Risk Score"].mean().sort_values())

        st.subheader("📈 Average Compliance by Institution")
        st.bar_chart(final_df.groupby("institution")["Compliance Score"].mean().sort_values())

        st.subheader("🚨 Top 10 Highest Risk Procurements")
        top_risk = final_df.sort_values("AI Risk Score", ascending=False).head(10)
        st.dataframe(top_risk[audit_cols], use_container_width=True)

        with st.expander("🌍 Real-World Impact"):
            st.write("""
            This system supports:
            - Faster public procurement audit reviews
            - Early detection of risky procurement cases
            - Improved transparency and accountability
            - Evidence-based inspection planning
            - Digital transformation of procurement compliance monitoring
            """)

        st.download_button(
            "Download Compliance Audit Report",
            display_df.to_csv(index=False),
            "gppa_compliance_audit_report.csv",
            "text/csv"
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

        st.subheader("🧠 SHAP Explainability")

        try:
            if hasattr(best_model, "feature_importances_"):
                X_sample = X_processed[:100]

                if hasattr(X_sample, "toarray"):
                    X_sample_dense = X_sample.toarray()
                else:
                    X_sample_dense = X_sample

                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_sample_dense)

                if isinstance(shap_values, list):
                    shap_mean = abs(shap_values[0]).mean(axis=0)
                else:
                    shap_mean = abs(shap_values).mean(axis=0)

                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Mean SHAP Importance": shap_mean
                }).sort_values("Mean SHAP Importance", ascending=False)

                st.dataframe(shap_df.head(15), use_container_width=True)
                st.bar_chart(shap_df.set_index("Feature").head(10))
            else:
                st.info("SHAP tree explainability is only available for tree-based models.")
        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

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

        st.dataframe(final_df[ml_cols], use_container_width=True)

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
