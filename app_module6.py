from pathlib import Path
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Governance Simulation",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ AI Governance Simulation")

st.markdown(
    "You are a member of an organization's AI Governance Committee. "
    "Your responsibility is to evaluate an AI-powered lending model "
    "and recommend appropriate governance actions."
)

st.divider()


# ============================================================
# CSV DATA CONNECTION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_table(table_name):
    """
    Load governance data from CSV files.

    The CSV files contain synthetic model evaluation data
    previously generated in Databricks.

    Student decisions are not written to these files.
    """

    allowed_tables = {
        "model_performance",
        "model_fairness_summary",
        "governance_policies",
        "model_feature_importance",
        "model_fairness",
        "monitoring_policies",
        "monitoring_audit_log",
        "scenario_normal",
        "scenario_data_quality",
        "scenario_data_drift",
        "scenario_performance",
    }

    if table_name not in allowed_tables:
        raise ValueError(f"Table not permitted: {table_name}")

    file_path = DATA_DIR / f"{table_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required CSV file not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# GOVERNANCE POLICY DEFINITIONS
# ============================================================

POLICY_LABELS = {
    "auc": "Minimum AUC",
    "precision": "Minimum Precision",
    "recall": "Minimum Recall",
    "recall_gap": "Maximum Recall Disparity",
    "false_positive_rate_gap": "Maximum False Positive Rate Disparity",
}

POLICY_DIRECTIONS = {
    "auc": "min",
    "precision": "min",
    "recall": "min",
    "recall_gap": "max",
    "false_positive_rate_gap": "max",
}


def evaluate_policy(metric, actual, threshold):
    """
    Determine whether a metric satisfies its governance policy.
    """

    direction = POLICY_DIRECTIONS[metric]

    if direction == "min":
        return actual >= threshold

    return actual <= threshold


def overall_status(results):
    """
    Determine the overall automated governance decision.
    """

    if results and all(row["passed"] for row in results):
        return "ELIGIBLE FOR HUMAN REVIEW"

    return "BLOCKED"


# ============================================================
# PDF GENERATION
# ============================================================

def generate_decision_log_pdf(
    student_name,
    performance,
    fairness,
    policy_results,
    recommendation,
    primary_risk,
    policy_recommendations,
    safeguards,
    rationale,
):
    """
    Generate a PDF Governance Decision Log in memory.

    No student responses are written to the CSV files
    or permanently stored on the application server.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(8.5 * inch, 11 * inch),
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSection",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=13,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportCell",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
        )
    )

    story = []

    def safe_text(value):
        """
        Escape text for safe rendering in ReportLab paragraphs.
        """
        return escape(str(value)).replace("\n", "<br/>")

    def add_paragraph(value):
        story.append(
            Paragraph(value, styles["ReportBody"])
        )

    def add_section(title):
        story.append(
            Paragraph(safe_text(title), styles["ReportSection"])
        )

    def add_response(label, response):
        story.append(
            KeepTogether(
                [
                    Paragraph(
                        f"<b>{safe_text(label)}</b>",
                        styles["ReportBody"],
                    ),
                    Paragraph(
                        safe_text(response),
                        styles["ReportBody"],
                    ),
                ]
            )
        )

    def add_table(headers, rows, widths):

        data = [
            [
                Paragraph(safe_text(h), styles["ReportCell"])
                for h in headers
            ]
        ]

        for row in rows:
            data.append(
                [
                    Paragraph(safe_text(value), styles["ReportCell"])
                    for value in row
                ]
            )

        table = Table(
            data,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#E8EEF5"),
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#CCD5DF"),
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 9))

    # --------------------------------------------------------
    # REPORT HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI Governance Committee<br/>Decision Log",
            styles["ReportTitle"],
        )
    )

    add_paragraph("Module 4: Governance Framework")

    add_paragraph(
        f"<b>Student:</b> {safe_text(student_name)}<br/>"
        f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}"
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    add_section("1. Model Performance")

    add_table(
        ["Metric", "Value"],
        [
            ("AUC", f"{float(performance['auc']):.3f}"),
            ("Precision", f"{float(performance['precision']):.3f}"),
            ("Recall", f"{float(performance['recall']):.3f}"),
            ("Accuracy", f"{float(performance['accuracy']):.3f}"),
        ],
        [3.2 * inch, 3.2 * inch],
    )

    # --------------------------------------------------------
    # FAIRNESS EVALUATION
    # --------------------------------------------------------

    add_section("2. Fairness Evaluation")

    add_table(
        ["Metric", "Value"],
        [
            (
                "Recall Disparity",
                f"{float(fairness['recall_gap']):.3f}",
            ),
            (
                "False Positive Rate Disparity",
                f"{float(fairness['false_positive_rate_gap']):.3f}",
            ),
        ],
        [3.2 * inch, 3.2 * inch],
    )

    # --------------------------------------------------------
    # GOVERNANCE POLICY RESULTS
    # --------------------------------------------------------

    add_section("3. Governance Policy Evaluation")

    policy_rows = []

    for result in policy_results:
        policy_rows.append(
            (
                result["label"],
                f"{result['actual']:.3f}",
                f"{result['original_threshold']:.2f}",
                f"{result['adjusted_threshold']:.2f}",
                "PASS" if result["passed"] else "FAIL",
            )
        )

    add_table(
        [
            "Policy",
            "Actual",
            "Original",
            "Adjusted",
            "Result",
        ],
        policy_rows,
        [
            2.35 * inch,
            0.85 * inch,
            1.0 * inch,
            1.15 * inch,
            1.05 * inch,
        ],
    )

    status = overall_status(policy_results)

    add_paragraph(
        f"<b>Automated Policy Evaluation:</b> {safe_text(status)}"
    )

    add_paragraph(
        "This result reflects the student-adjusted policy "
        "thresholds. It does not constitute deployment approval."
    )

    # --------------------------------------------------------
    # COMMITTEE RECOMMENDATION
    # --------------------------------------------------------

    add_section("4. Governance Committee Recommendation")

    add_response(
        "Committee Recommendation",
        recommendation,
    )

    # --------------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------------

    add_section("5. Primary Risk Assessment")

    add_response(
        "Most Significant Performance or Fairness Concern",
        primary_risk,
    )

    # --------------------------------------------------------
    # POLICY RECOMMENDATIONS
    # --------------------------------------------------------

    add_section("6. Policy Recommendations")

    add_response(
        "Recommended Policy Changes",
        policy_recommendations,
    )

    # --------------------------------------------------------
    # REQUIRED SAFEGUARDS
    # --------------------------------------------------------

    add_section("7. Required Safeguards")

    add_response(
        "Monitoring, Oversight, and Additional Controls",
        safeguards,
    )

    # --------------------------------------------------------
    # DECISION RATIONALE
    # --------------------------------------------------------

    add_section("8. Decision Rationale")

    add_response(
        "Justification for the Committee Recommendation",
        rationale,
    )

    story.append(Spacer(1, 12))

    add_paragraph(
        "This report was generated as part of an educational "
        "AI governance simulation using synthetic lending data."
    )

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ============================================================
# LOAD GOVERNANCE DATA
# ============================================================

try:

    performance_df = load_table("model_performance")
    fairness_df = load_table("model_fairness_summary")
    policies_df = load_table("governance_policies")

except Exception as e:

    st.error(f"Unable to load governance data: {e}")
    st.stop()


if (
    performance_df.empty
    or fairness_df.empty
    or policies_df.empty
):

    st.error(
        "One or more governance CSV files contain no records."
    )
    st.stop()


performance = performance_df.iloc[0]
fairness = fairness_df.iloc[0]


def generate_module5_pdf(student_name, importance, fairness_groups, selections):
    """Build the student's Module 5 assessment in memory only."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(8.5 * inch, 11 * inch),
                            rightMargin=48, leftMargin=48,
                            topMargin=48, bottomMargin=48)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 9
    body.leading = 13
    story = []

    def paragraph(value):
        story.append(Paragraph(escape(str(value)).replace("\n", "<br/>"), body))
        story.append(Spacer(1, 7))

    def section(label):
        story.append(Spacer(1, 9))
        story.append(Paragraph(escape(label), styles["Heading2"]))

    story.append(Paragraph("Module 5 — Explainability & Fairness Assessment", styles["Title"]))
    paragraph(f"Student: {student_name} | Date: {datetime.now().strftime('%B %d, %Y')}")
    paragraph("Educational simulation using synthetic lending data. "
              "The app uses precomputed model evidence; it does not retrain the model.")

    section("1. Model evidence")
    paragraph("Global SHAP feature importance (mean absolute SHAP):")
    for _, row in importance.head(12).iterrows():
        paragraph(f"{row['feature']}: {float(row['mean_absolute_shap']):.4f}")
    section("2. Fairness evidence")
    paragraph("Group-level evaluation data as supplied by the simulation:")
    for _, row in fairness_groups.head(30).iterrows():
        paragraph(" | ".join(f"{column}: {row[column]}" for column in fairness_groups.columns))
    for title, key in [
        ("3. Explainability investigation", "feature_review"),
        ("Flagged features", "flagged_features"),
        ("4. Fairness investigation", "fairness_review"),
        ("Additional evidence requested", "evidence_requests"),
        ("5. Mitigation plan", "mitigations"),
        ("Safeguard justification", "mitigation_rationale"),
        ("6. Committee assessment", "risk_rating"),
        ("Committee recommendation", "recommendation"),
        ("Decision rationale", "decision_rationale"),
    ]:
        section(title)
        value = selections.get(key, "")
        paragraph(", ".join(value) if isinstance(value, list) else value)
    paragraph("This assessment is a student recommendation, not authorization to deploy a model.")
    doc.build(story)
    result = buffer.getvalue()
    buffer.close()
    return result



# ============================================================
# MODULE SELECTION
# ============================================================

module = st.selectbox(
    "Select a Governance Module",
    [
        "Module 4: Governance Framework",
        "Module 5: Performance & Explainability",
        "Module 6: Guardrails & Monitoring",
    ],
)

st.divider()


# ============================================================
# MODULE 4: GOVERNANCE FRAMEWORK
# ============================================================

if module == "Module 4: Governance Framework":

    st.header("Module 4: Model Governance Review")

    st.markdown(
        """
        ### Your Assignment

        Your organization has developed an AI model to predict
        whether a loan applicant will default.

        Before approving the model for operational use,
        the AI Governance Committee must determine whether
        it meets the organization's performance and fairness
        requirements.

        **Your responsibilities:**

        1. Review model performance and fairness.
        2. Evaluate the existing governance policies.
        3. Experiment with policy thresholds.
        4. Identify important governance risks.
        5. Recommend an appropriate course of action.
        6. Download your Governance Decision Log as a PDF.

        **Estimated time:** 20–30 minutes.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 1. MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("1. Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "AUC",
        f"{float(performance['auc']):.3f}",
    )

    col2.metric(
        "Precision",
        f"{float(performance['precision']):.3f}",
    )

    col3.metric(
        "Recall",
        f"{float(performance['recall']):.3f}",
    )

    col4.metric(
        "Accuracy",
        f"{float(performance['accuracy']):.3f}",
    )

    st.caption(
        "AUC measures how well the model distinguishes "
        "between applicants who default and those who do not. "
        "Precision measures how often predicted defaults are correct. "
        "Recall measures how many actual defaults are identified."
    )

    st.divider()

    # --------------------------------------------------------
    # 2. FAIRNESS EVALUATION
    # --------------------------------------------------------

    st.subheader("2. Fairness Evaluation")

    col1, col2 = st.columns(2)

    col1.metric(
        "Recall Disparity",
        f"{float(fairness['recall_gap']):.3f}",
    )

    col2.metric(
        "False Positive Rate Disparity",
        f"{float(fairness['false_positive_rate_gap']):.3f}",
    )

    st.info(
        "These metrics describe differences in model performance "
        "across demographic groups. They do not independently "
        "establish whether the model is fair or discriminatory."
    )

    st.divider()

    # --------------------------------------------------------
    # 3. GOVERNANCE POLICY CONFIGURATION
    # --------------------------------------------------------

    st.subheader("3. Configure Governance Policies")

    st.markdown(
        """
        Experiment with the policy thresholds below.

        **Challenge 1:** Identify which original policies the model fails.

        **Challenge 2:** Adjust the thresholds until the model
        becomes eligible for human review.

        **Challenge 3:** Consider whether relaxing those policies
        is justified. A passing result does not necessarily
        mean the model is safe or appropriate to deploy.

        Your adjusted thresholds will be included in your PDF.
        """
    )

    actual_metrics = {
        "auc": float(performance["auc"]),
        "precision": float(performance["precision"]),
        "recall": float(performance["recall"]),
        "recall_gap": float(fairness["recall_gap"]),
        "false_positive_rate_gap": float(
            fairness["false_positive_rate_gap"]
        ),
    }

    policy_results = []

    for _, policy in policies_df.sort_values(
        "policy_id"
    ).iterrows():

        policy_id = str(policy["policy_id"])
        metric = str(policy["metric"])
        threshold = float(policy["threshold"])

        if metric not in POLICY_DIRECTIONS:

            st.error(
                f"Unknown governance metric: {metric}"
            )
            st.stop()

        if not 0 <= threshold <= 1:

            st.error(
                f"Policy {policy_id} has a threshold "
                "outside the supported range of 0 to 1."
            )
            st.stop()

        label = POLICY_LABELS[metric]
        actual = actual_metrics[metric]

        adjusted_threshold = st.slider(
            label,
            min_value=0.0,
            max_value=1.0,
            value=threshold,
            step=0.01,
            key=f"policy_{policy_id}",
            help=str(policy["description"]),
        )

        passed = evaluate_policy(
            metric,
            actual,
            adjusted_threshold,
        )

        original_passed = evaluate_policy(
            metric,
            actual,
            threshold,
        )

        policy_results.append(
            {
                "policy_id": policy_id,
                "metric": metric,
                "label": label,
                "actual": actual,
                "original_threshold": threshold,
                "adjusted_threshold": adjusted_threshold,
                "original_passed": original_passed,
                "passed": passed,
            }
        )

    st.divider()

    # --------------------------------------------------------
    # 4. GOVERNANCE DECISION
    # --------------------------------------------------------

    st.subheader("4. Governance Decision")

    results_df = pd.DataFrame(
        [
            {
                "Policy": result["policy_id"],
                "Metric": result["label"],
                "Actual Value": round(
                    result["actual"], 4
                ),
                "Original Threshold": round(
                    result["original_threshold"], 4
                ),
                "Adjusted Threshold": round(
                    result["adjusted_threshold"], 4
                ),
                "Original Status": (
                    "PASS"
                    if result["original_passed"]
                    else "FAIL"
                ),
                "Adjusted Status": (
                    "PASS"
                    if result["passed"]
                    else "FAIL"
                ),
            }
            for result in policy_results
        ]
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    status = overall_status(policy_results)

    if status == "ELIGIBLE FOR HUMAN REVIEW":

        st.success(
            "✅ ELIGIBLE FOR HUMAN REVIEW — "
            "All adjusted governance policies are satisfied."
        )

    else:

        st.error(
            "🚫 BLOCKED — One or more adjusted "
            "governance policies are not satisfied."
        )

    original_status = (
        "ELIGIBLE FOR HUMAN REVIEW"
        if all(
            result["original_passed"]
            for result in policy_results
        )
        else "BLOCKED"
    )

    st.caption(
        f"Decision under original policies: {original_status}. "
        f"Decision under adjusted policies: {status}."
    )

    st.warning(
        "Changing a threshold changes the policy evaluation, "
        "not the underlying model's performance or fairness. "
        "A passing policy evaluation is not deployment approval."
    )

    st.divider()

    # --------------------------------------------------------
    # 5. GOVERNANCE COMMITTEE DECISION LOG
    # --------------------------------------------------------

    st.subheader("5. Governance Committee Decision Log")

    st.markdown(
        """
        Complete the following questions as a member
        of the AI Governance Committee.

        Your responses and selected policy thresholds
        will be included in your downloadable PDF.
        """
    )

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter your full name",
    )

    recommendation = st.radio(
        "What is your committee recommendation?",
        [
            "Approve for further review",
            "Reject the model",
            "Request additional evidence before deciding",
        ],
        index=None,
    )

    primary_risk = st.text_area(
        "What is the most significant performance or fairness risk?",
        placeholder=(
            "Identify the concern and explain why it matters "
            "to the organization."
        ),
        height=110,
    )

    policy_recommendations = st.text_area(
        "Would you recommend changing any governance thresholds? Why?",
        placeholder=(
            "Explain which policies you would retain or modify "
            "and justify your recommendation."
        ),
        height=110,
    )

    safeguards = st.text_area(
        "What additional safeguards or oversight would you require?",
        placeholder=(
            "Consider monitoring, human review, additional testing, "
            "documentation, or escalation procedures."
        ),
        height=110,
    )

    rationale = st.text_area(
        "Explain the rationale for your committee recommendation.",
        placeholder=(
            "Summarize how the evidence, risks, and governance "
            "requirements support your recommendation."
        ),
        height=140,
    )

    # --------------------------------------------------------
    # 6. PDF DOWNLOAD
    # --------------------------------------------------------

    st.divider()

    st.subheader("6. Download Your Governance Decision Log")

    required_responses = {
        "Student name": student_name,
        "Committee recommendation": recommendation,
        "Primary risk assessment": primary_risk,
        "Policy recommendations": policy_recommendations,
        "Required safeguards": safeguards,
        "Decision rationale": rationale,
    }

    missing_fields = [
        label
        for label, value in required_responses.items()
        if not value or not str(value).strip()
    ]

    if missing_fields:

        st.info(
            "Complete all Decision Log fields to enable "
            "the PDF download."
        )

    else:

        try:

            pdf_data = generate_decision_log_pdf(
                student_name=student_name,
                performance=performance,
                fairness=fairness,
                policy_results=policy_results,
                recommendation=recommendation,
                primary_risk=primary_risk,
                policy_recommendations=policy_recommendations,
                safeguards=safeguards,
                rationale=rationale,
            )

            safe_name = "".join(
                character
                for character in student_name.strip().replace(
                    " ", "_"
                )
                if character.isalnum()
                or character in "_-"
            ) or "Student"

            st.download_button(
                label="📄 Download Governance Decision Log (PDF)",
                data=pdf_data,
                file_name=(
                    f"Module4_Governance_Decision_Log_"
                    f"{safe_name}.pdf"
                ),
                mime="application/pdf",
                type="primary",
            )

            st.success(
                "Your Governance Decision Log is ready. "
                "Download the PDF and submit it according "
                "to your instructor's directions."
            )

        except Exception as e:

            st.error(
                f"Unable to generate the PDF: {e}"
            )

    st.caption(
        "Your responses are not permanently saved in this "
        "application. Download your PDF before closing "
        "your session."
    )


# ============================================================
# MODULE 5: EXPLAINABILITY, FAIRNESS, AND RISK MITIGATION
# ============================================================

elif module == "Module 5: Performance & Explainability":
    st.header("Module 5: Explainability, Fairness & Risk Mitigation")
    st.markdown(
        "**Your role:** You are an AI Governance Committee member investigating "
        "the same synthetic lending model from Module 4. Review existing evidence, "
        "identify gaps, and recommend safeguards. **Estimated time:** 25–35 minutes."
    )
    st.info("The evidence shown here was calculated before deployment of this "
            "teaching app. Selecting features or safeguards does not change the model.")

    try:
        importance_df = load_table("model_feature_importance")
        group_df = load_table("model_fairness")
    except Exception as exc:
        st.error(f"Module 5 data could not be loaded: {exc}")
        st.stop()

    if importance_df.empty or group_df.empty:
        st.error("Module 5 requires nonempty feature-importance and fairness CSV files.")
        st.stop()

    required_importance = {"feature", "mean_absolute_shap"}
    if not required_importance.issubset(importance_df.columns):
        st.error("model_feature_importance.csv must include 'feature' and "
                 "'mean_absolute_shap'. Available columns: "
                 + ", ".join(importance_df.columns))
        st.stop()

    importance_df = importance_df.copy()
    importance_df["mean_absolute_shap"] = pd.to_numeric(
        importance_df["mean_absolute_shap"], errors="coerce")
    importance_df = importance_df.dropna(subset=["feature", "mean_absolute_shap"])
    importance_df = importance_df.sort_values("mean_absolute_shap", ascending=False)
    if importance_df.empty:
        st.error("No usable SHAP importance values were found.")
        st.stop()

    explain_tab, fairness_tab, mitigation_tab, submit_tab = st.tabs([
        "1. Explore Explanations", "2. Investigate Fairness",
        "3. Develop Mitigation Plan", "4. Submit Assessment"
    ])

    with explain_tab:
        st.subheader("Global feature importance")
        st.caption("Mean absolute SHAP indicates the average magnitude of a feature's "
                   "contribution across the evaluated predictions. It does not indicate "
                   "direction, causality, or explain an individual applicant's outcome.")
        st.bar_chart(importance_df.set_index("feature")["mean_absolute_shap"])
        chosen_feature = st.selectbox("Investigate a feature", importance_df["feature"].astype(str).tolist(),
                                      key="m5_feature")
        chosen_row = importance_df.loc[importance_df["feature"].astype(str) == chosen_feature].iloc[0]
        st.metric("Mean absolute SHAP importance", f"{chosen_row['mean_absolute_shap']:.4f}")
        st.write("Consider whether this variable is understandable, appropriately documented, "
                 "or might act as a proxy for sensitive information. Importance alone does not "
                 "establish that a feature is inappropriate.")
        flagged_features = st.multiselect(
            "Flag features for governance investigation (select any that warrant review)",
            importance_df["feature"].astype(str).tolist(), key="m5_flagged_features")
        feature_review = st.text_area(
            "Explain your main explainability concern and why the selected feature(s) merit review.",
            height=130, key="m5_feature_review")

    with fairness_tab:
        st.subheader("Group-level fairness evaluation")
        st.caption("These are precomputed evaluation results. Group disparities can indicate "
                   "areas for investigation but do not independently establish discrimination.")
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        numeric_columns = [col for col in group_df.columns
                           if pd.api.types.is_numeric_dtype(group_df[col])]
        group_columns = [col for col in group_df.columns if col not in numeric_columns]
        if numeric_columns and group_columns:
            metric_choice = st.selectbox("Compare a group-level measure", numeric_columns,
                                         key="m5_fairness_metric")
            label_choice = st.selectbox("Group label", group_columns,
                                        key="m5_fairness_group")
            chart_df = group_df[[label_choice, metric_choice]].dropna()
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index(label_choice)[metric_choice])
        st.markdown("**Overall disparities from the original evaluation**")
        fc1, fc2 = st.columns(2)
        fc1.metric("Recall disparity", f"{float(fairness['recall_gap']):.3f}")
        fc2.metric("False positive rate disparity",
                   f"{float(fairness['false_positive_rate_gap']):.3f}")
        fairness_review = st.text_area(
            "Which fairness findings concern you most? Discuss implications and limitations of the evidence.",
            height=130, key="m5_fairness_review")
        evidence_requests = st.multiselect(
            "What additional evidence would you request?",
            ["Group sample sizes and uncertainty intervals",
             "Performance across additional relevant groups and intersections",
             "Review of feature proxies and data collection practices",
             "Outcome and error analysis by lending use case",
             "Independent legal/compliance review",
             "Local explanations for representative and adverse cases"],
            key="m5_evidence_requests")

    with mitigation_tab:
        st.subheader("Develop a governance mitigation plan")
        st.write("Choose safeguards that address the concerns you identified. "
                 "Each choice has costs and limitations; selecting every option is not necessarily best.")
        mitigations = st.multiselect(
            "Recommended safeguards",
            ["Targeted fairness testing before further review",
             "Independent explainability and proxy-feature audit",
             "Review feature inclusion and data provenance",
             "Human review of adverse lending decisions",
             "Document model limitations and intended-use restrictions",
             "Require additional model development and re-evaluation",
             "Establish fairness and performance monitoring with escalation"],
            key="m5_mitigations")
        mitigation_rationale = st.text_area(
            "Explain why your selected safeguards are appropriate and what risks remain.",
            height=130, key="m5_mitigation_rationale")
        risk_rating = st.radio("Overall risk rating", ["Low", "Moderate", "High"],
                               index=None, key="m5_risk_rating", horizontal=True)
        recommendation = st.radio(
            "Committee recommendation",
            ["Proceed to further review with safeguards",
             "Require remediation and additional evidence before proceeding",
             "Reject the model for the proposed use"],
            index=None, key="m5_recommendation")

    with submit_tab:
        st.subheader("Model Explainability & Fairness Assessment")
        student_name_m5 = st.text_input("Student name", key="m5_student_name")
        decision_rationale = st.text_area(
            "Summarize your recommendation using the model evidence, explainability "
            "findings, fairness risks, and proposed safeguards.",
            height=160, key="m5_decision_rationale")
        st.write("**Assessment checklist**")
        required = {
            "Student name": student_name_m5,
            "Explainability investigation": feature_review,
            "Fairness investigation": fairness_review,
            "Additional evidence requested": evidence_requests,
            "Safeguard selection": mitigations,
            "Safeguard justification": mitigation_rationale,
            "Risk rating": risk_rating,
            "Committee recommendation": recommendation,
            "Decision rationale": decision_rationale,
        }
        missing = [name for name, value in required.items() if not value or not str(value).strip()]
        if missing:
            st.info("Complete these items to enable your PDF: " + ", ".join(missing))
        else:
            answers = {
                "feature_review": feature_review,
                "flagged_features": flagged_features,
                "fairness_review": fairness_review,
                "evidence_requests": evidence_requests,
                "mitigations": mitigations,
                "mitigation_rationale": mitigation_rationale,
                "risk_rating": risk_rating,
                "recommendation": recommendation,
                "decision_rationale": decision_rationale,
            }
            try:
                pdf_data_m5 = generate_module5_pdf(
                    student_name_m5, importance_df, group_df, answers)
                safe_name_m5 = "".join(c for c in student_name_m5.strip().replace(" ", "_")
                                       if c.isalnum() or c in "_-") or "Student"
                st.download_button("📄 Download Module 5 Assessment (PDF)",
                                   data=pdf_data_m5,
                                   file_name=f"Module5_Explainability_Fairness_{safe_name_m5}.pdf",
                                   mime="application/pdf", type="primary")
            except Exception as exc:
                st.error(f"Unable to generate Module 5 PDF: {exc}")
        st.caption("Responses are held in your browser session, not permanently saved. "
                   "Download the PDF before closing the app. "
                   "This assessment is not model deployment approval.")


# ============================================================
# MODULE 6: MONITORING AND GUARDRAILS
# ============================================================

elif module == "Module 6: Guardrails & Monitoring":

    st.header("Module 6: Model Monitoring & Guardrails")

    st.markdown(
        """
        **Your role:** You are a member of the AI Governance
        Committee responsible for overseeing a deployed
        lending model.

        Your task is to investigate monitoring alerts,
        evaluate governance guardrails, and recommend
        an appropriate operational response.
        """
    )

    st.info(
        "This simulation uses synthetic lending data and "
        "precomputed monitoring scenarios. Your decisions "
        "do not affect a real AI system."
    )

    # --------------------------------------------------------
    # LOAD MONITORING DATA
    # --------------------------------------------------------

    try:
        monitoring_policies = load_table("monitoring_policies")
        monitoring_audit = load_table("monitoring_audit_log")

        monitoring_scenarios = {
            "Normal Operations": load_table("scenario_normal"),
            "Data Quality Issue": load_table("scenario_data_quality"),
            "Data Drift": load_table("scenario_data_drift"),
            "Performance Deterioration": load_table(
                "scenario_performance"
            ),
        }

    except Exception as e:
        st.error(f"Unable to load monitoring data: {e}")
        st.stop()

    st.success("Monitoring evidence loaded successfully.")

    # --------------------------------------------------------
    # MODULE 6 NAVIGATION
    # --------------------------------------------------------

    (
        tab_alerts,
        tab_guardrails,
        tab_response,
        tab_submit,
    ) = st.tabs(
        [
            "1. Investigate Alerts",
            "2. Evaluate Guardrails",
            "3. Decide & Escalate",
            "4. Submit Incident Assessment",
        ]
    )

    # --------------------------------------------------------
    # STUDENT SESSION STATE
    # --------------------------------------------------------

    if "m6_scenario" not in st.session_state:
        st.session_state["m6_scenario"] = "Normal Operations"

    if "m6_risk_rating" not in st.session_state:
        st.session_state["m6_risk_rating"] = "Moderate"

    if "m6_response" not in st.session_state:
        st.session_state["m6_response"] = "Investigate Further"

    # --------------------------------------------------------
    # TAB 1: INVESTIGATE MONITORING ALERTS
    # --------------------------------------------------------

    with tab_alerts:

        st.subheader("Investigate Monitoring Alerts")

        st.markdown(
            """
            Review the monitoring scenarios below.

            Select one scenario for your formal governance
            assessment. Examine the evidence and consider
            whether the model is operating within acceptable
            risk limits.
            """
        )

        scenario_names = list(monitoring_scenarios.keys())

        selected_scenario = st.selectbox(
            "Select a monitoring scenario",
            scenario_names,
            key="m6_scenario",
        )

        scenario_df = monitoring_scenarios[selected_scenario]
        # ----------------------------------------------------
        # CONSOLIDATED MONITORING GUARDRAIL SUMMARY
        # ----------------------------------------------------

        st.markdown("### Monitoring Guardrail Summary")

        st.caption(
            "This dashboard compares the selected scenario "
            "with established monitoring thresholds. "
            "Alerts indicate issues requiring governance review, "
            "not automatic decisions to suspend the model."
        )

        baseline_summary = monitoring_scenarios["Normal Operations"]
        current_summary = scenario_df

        # Classroom validation assumptions.
        summary_rules = {
            "age": (18, 100),
            "annual_income": (0, None),
            "credit_score": (300, 850),
            "loan_amount": (0, None),
            "debt_to_income": (0, 1),
            "employment_years": (0, 80),
        }

        # M001: Missing data
        summary_fields = [
            "age", "annual_income", "credit_score",
            "loan_amount", "debt_to_income",
            "employment_years", "demographic_group",
        ]

        available_summary_fields = [
            c for c in summary_fields
            if c in current_summary.columns
        ]

        missing_value = None

        if available_summary_fields and not current_summary.empty:
            summary_data = current_summary[available_summary_fields]
            missing_value = (
                summary_data.isna().sum().sum()
                / summary_data.size
            )

        # M002: Invalid records
        invalid_value = None

        if not current_summary.empty:
            invalid_mask = pd.Series(
                False, index=current_summary.index
            )

            for field, (minimum, maximum) in summary_rules.items():
                if field not in current_summary.columns:
                    continue

                numeric_values = pd.to_numeric(
                    current_summary[field], errors="coerce"
                )

                invalid_field = (
                    numeric_values.notna()
                    & (numeric_values < minimum)
                )

                if maximum is not None:
                    invalid_field = invalid_field | (
                        numeric_values.notna()
                        & (numeric_values > maximum)
                    )

                invalid_mask = invalid_mask | invalid_field

            invalid_value = invalid_mask.mean()

        # M003: Population Stability Index
        # Calculate PSI locally so this summary can appear
        # before the detailed evidence dashboard.

        import numpy as np

        def summary_psi(reference, observed, bins=10):
            reference = pd.to_numeric(
                reference, errors="coerce"
            ).dropna()

            observed = pd.to_numeric(
                observed, errors="coerce"
            ).dropna()

            if len(reference) < 2 or len(observed) < 2:
                return None

            edges = np.unique(
                np.quantile(
                    reference,
                    np.linspace(0, 1, bins + 1),
                )
            )

            if len(edges) < 2:
                return None

            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_counts, _ = np.histogram(
                reference, bins=edges
            )
            obs_counts, _ = np.histogram(
                observed, bins=edges
            )

            epsilon = 0.0001

            ref_pct = np.maximum(
                ref_counts / ref_counts.sum(), epsilon
            )
            obs_pct = np.maximum(
                obs_counts / obs_counts.sum(), epsilon
            )

            return float(np.sum(
                (obs_pct - ref_pct)
                * np.log(obs_pct / ref_pct)
            ))

        psi_values = []

        for feature in [
            "credit_score",
            "annual_income",
            "debt_to_income",
            "loan_amount",
            "employment_years",
        ]:
            if (
                feature in baseline_summary.columns
                and feature in current_summary.columns
            ):
                value = summary_psi(
                    baseline_summary[feature],
                    current_summary[feature],
                )
                if value is not None:
                    psi_values.append(value)

        max_psi_value = max(psi_values) if psi_values else None

        def guardrail_status(value, threshold, direction="max"):
            if value is None:
                return "Not Evaluated"

            if direction == "max":
                return (
                    "ALERT" if value > threshold
                    else "Within Threshold"
                )

            return (
                "ALERT" if value < threshold
                else "Within Threshold"
            )

        guardrail_summary = pd.DataFrame([
            {
                "Policy": "M001 - Missing Data",
                "Observed": (
                    f"{missing_value:.2%}"
                    if missing_value is not None
                    else "Unavailable"
                ),
                "Threshold": "5% maximum",
                "Status": guardrail_status(
                    missing_value, 0.05
                ),
            },
            {
                "Policy": "M002 - Invalid Data",
                "Observed": (
                    f"{invalid_value:.2%}"
                    if invalid_value is not None
                    else "Unavailable"
                ),
                "Threshold": "2% maximum",
                "Status": guardrail_status(
                    invalid_value, 0.02
                ),
            },
            {
                "Policy": "M003 - Data Drift (Max PSI)",
                "Observed": (
                    f"{max_psi_value:.4f}"
                    if max_psi_value is not None
                    else "Unavailable"
                ),
                "Threshold": "0.20 maximum",
                "Status": guardrail_status(
                    max_psi_value, 0.20
                ),
            },
            {
                "Policy": "M004 - Model AUC",
                "Observed": "Unavailable",
                "Threshold": "0.75 minimum",
                "Status": "Not Evaluated",
            },
        ])

        st.dataframe(
            guardrail_summary,
            use_container_width=True,
            hide_index=True,
        )

        alert_count = (
            guardrail_summary["Status"] == "ALERT"
        ).sum()

        if alert_count:
            st.error(
                f"{alert_count} monitoring guardrail(s) "
                "exceeded their thresholds."
            )
        else:
            st.info(
                "No evaluated guardrails exceeded their "
                "thresholds. Review unevaluated metrics "
                "before drawing conclusions."
            )

        st.markdown(
            """
            **Governance investigation:** Which alerts require
            action? What additional evidence would you need?
            Does the situation warrant continued operation,
            increased monitoring, or restrictions?
            """
        )
        st.markdown("### Scenario Evidence")

        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True,
        )
        
        # ----------------------------------------------------
        # COMPARATIVE MONITORING DASHBOARD
        # ----------------------------------------------------

        st.markdown("### Monitoring Evidence Dashboard")

        st.caption(
            "Compare the selected scenario against the "
            "Normal Operations reference dataset. "
            "Differences are descriptive indicators, "
            "not automatically policy violations."
        )

        baseline_df = monitoring_scenarios["Normal Operations"]
        current_df = monitoring_scenarios[
            st.session_state["m6_scenario"]
        ]

        monitoring_columns = {
            "credit_score": "Average Credit Score",
            "annual_income": "Average Annual Income",
            "debt_to_income": "Average Debt-to-Income Ratio",
            "loan_amount": "Average Loan Amount",
            "employment_years": "Average Employment Years",
        }

        comparison_rows = []

        for column, label in monitoring_columns.items():

            if (
                column in baseline_df.columns
                and column in current_df.columns
            ):
                baseline_values = pd.to_numeric(
                    baseline_df[column], errors="coerce"
                )
                current_values = pd.to_numeric(
                    current_df[column], errors="coerce"
                )

                baseline_mean = baseline_values.mean()
                current_mean = current_values.mean()

                comparison_rows.append({
                    "Metric": label,
                    "Normal Operations": round(baseline_mean, 3),
                    "Selected Scenario": round(current_mean, 3),
                    "Change": round(
                        current_mean - baseline_mean, 3
                    ),
                })

        if comparison_rows:
            st.dataframe(
                pd.DataFrame(comparison_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Applicant Demographic Distribution")

        if (
            "demographic_group" in baseline_df.columns
            and "demographic_group" in current_df.columns
        ):
            baseline_dist = (
                baseline_df["demographic_group"]
                .value_counts(normalize=True)
                .mul(100)
            )

            current_dist = (
                current_df["demographic_group"]
                .value_counts(normalize=True)
                .mul(100)
            )

            demographic_comparison = pd.DataFrame({
                "Normal Operations (%)": baseline_dist,
                "Selected Scenario (%)": current_dist,
            }).fillna(0)

            demographic_comparison["Change (percentage points)"] = (
                demographic_comparison["Selected Scenario (%)"]
                - demographic_comparison["Normal Operations (%)"]
            )

            st.dataframe(
                demographic_comparison.round(2),
                use_container_width=True,
            )
        # ----------------------------------------------------
        # CLASSROOM DATA VALIDATION RULES
        # ----------------------------------------------------

        st.markdown("### Data Validation Assessment")

        st.caption(
            "These validation rules are assumptions for this "
            "educational simulation. They are not regulatory "
            "requirements or established lending standards."
        )

        validation_rules = {
            "age": (18, 100),
            "annual_income": (0, None),
            "credit_score": (300, 850),
            "loan_amount": (0, None),
            "debt_to_income": (0, 1),
            "employment_years": (0, 80),
        }

        validation_results = []

        for column, (minimum, maximum) in validation_rules.items():

            if column not in current_df.columns:
                continue

            values = pd.to_numeric(
                current_df[column],
                errors="coerce",
            )

            invalid = values.notna() & (values < minimum)

            if maximum is not None:
                invalid = invalid | (
                    values.notna() & (values > maximum)
                )

            validation_results.append({
                "Field": column,
                "Invalid Records": int(invalid.sum()),
                "Invalid Rate (%)": round(
                    100 * invalid.mean(), 2
                ) if len(invalid) else 0,
            })

        st.dataframe(
            pd.DataFrame(validation_results),
            use_container_width=True,
            hide_index=True,
        )

        # Count records with at least one invalid field.
        invalid_record_mask = pd.Series(
            False,
            index=current_df.index,
        )

        for column, (minimum, maximum) in validation_rules.items():

            if column not in current_df.columns:
                continue

            values = pd.to_numeric(
                current_df[column],
                errors="coerce",
            )

            invalid = values.notna() & (values < minimum)

            if maximum is not None:
                invalid = invalid | (
                    values.notna() & (values > maximum)
                )

            invalid_record_mask = invalid_record_mask | invalid

        invalid_rate = (
            invalid_record_mask.mean()
            if len(current_df) else 0
        )

        invalid_threshold = 0.02

        if invalid_rate > invalid_threshold:
            st.error(
                f"M002 ALERT: Invalid record rate is "
                f"{invalid_rate:.2%}, exceeding the "
                f"2% monitoring threshold."
            )
        else:
            st.success(
                f"M002: Invalid record rate is "
                f"{invalid_rate:.2%}, within the "
                f"2% monitoring threshold."
            )
                    # ----------------------------------------------------
        # M001: MISSING DATA GUARDRAIL
        # ----------------------------------------------------

        st.markdown("### Missing Data Guardrail (M001)")

        # Evaluate missingness across the required applicant
        # fields, excluding identifiers and target outcomes.
        required_fields = [
            "age",
            "annual_income",
            "credit_score",
            "loan_amount",
            "debt_to_income",
            "employment_years",
            "demographic_group",
        ]

        available_fields = [
            col for col in required_fields
            if col in current_df.columns
        ]

        missing_threshold = 0.05

        if not available_fields or current_df.empty:
            st.warning(
                "M001: Insufficient data to evaluate missingness."
            )
        else:
            evaluated_df = current_df[available_fields]

            missing_rate = (
                evaluated_df.isna().sum().sum()
                / evaluated_df.size
            )

            missing_summary = pd.DataFrame({
                "Field": available_fields,
                "Missing Values": [
                    int(evaluated_df[col].isna().sum())
                    for col in available_fields
                ],
                "Missing Rate (%)": [
                    round(
                        100 * evaluated_df[col].isna().mean(), 2
                    )
                    for col in available_fields
                ],
            })

            st.dataframe(
                missing_summary,
                use_container_width=True,
                hide_index=True,
            )

            if missing_rate > missing_threshold:
                st.error(
                    f"M001 ALERT: Missing data rate is "
                    f"{missing_rate:.2%}, exceeding the "
                    f"5% monitoring threshold."
                )
            else:
                st.success(
                    f"M001: Missing data rate is "
                    f"{missing_rate:.2%}, within the "
                    f"5% monitoring threshold."
                )
         # ----------------------------------------------------
        # M003: POPULATION STABILITY INDEX (PSI)
        # ----------------------------------------------------

        st.markdown("### Population Stability Index (M003)")

        st.caption(
            "PSI compares the distribution of each applicant "
            "feature with the Normal Operations reference data. "
            "A PSI above 0.20 triggers the simulation's "
            "monitoring guardrail."
        )

        import numpy as np

        def calculate_psi(reference, observed, bins=10):

            reference = pd.to_numeric(
                reference, errors="coerce"
            ).dropna()

            observed = pd.to_numeric(
                observed, errors="coerce"
            ).dropna()

            if len(reference) < 2 or len(observed) < 2:
                return None

            # Use reference quantiles to define bins.
            edges = np.unique(
                np.quantile(
                    reference,
                    np.linspace(0, 1, bins + 1),
                )
            )

            if len(edges) < 2:
                return None

            # Include observations outside reference range.
            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_counts, _ = np.histogram(
                reference, bins=edges
            )

            obs_counts, _ = np.histogram(
                observed, bins=edges
            )

            # Smooth empty bins to avoid log(0).
            epsilon = 0.0001

            ref_pct = np.maximum(
                ref_counts / ref_counts.sum(), epsilon
            )

            obs_pct = np.maximum(
                obs_counts / obs_counts.sum(), epsilon
            )

            psi = np.sum(
                (obs_pct - ref_pct)
                * np.log(obs_pct / ref_pct)
            )

            return float(psi)

        psi_features = [
            "credit_score",
            "annual_income",
            "debt_to_income",
            "loan_amount",
            "employment_years",
        ]

        psi_results = []

        for feature in psi_features:

            if (
                feature not in baseline_df.columns
                or feature not in current_df.columns
            ):
                continue

            psi_value = calculate_psi(
                baseline_df[feature],
                current_df[feature],
            )

            if psi_value is not None:
                psi_results.append({
                    "Feature": feature,
                    "PSI": round(psi_value, 4),
                    "Threshold": 0.20,
                    "Status": (
                        "ALERT"
                        if psi_value > 0.20
                        else "Within Threshold"
                    ),
                })

        if psi_results:

            psi_df = pd.DataFrame(psi_results)

            st.dataframe(
                psi_df,
                use_container_width=True,
                hide_index=True,
            )

            alert_features = psi_df[
                psi_df["Status"] == "ALERT"
            ]

            if not alert_features.empty:
                st.error(
                    "M003 ALERT: One or more applicant "
                    "features exceed the PSI threshold of 0.20."
                )
            else:
                st.success(
                    "M003: All evaluated features are "
                    "within the PSI threshold of 0.20."
                )

        else:
            st.warning(
                "M003: Insufficient data to calculate PSI."
            )
            
        st.markdown("### Data Quality Indicators")

        quality_rows = []

        for label, df in [
            ("Normal Operations", baseline_df),
            ("Selected Scenario", current_df),
        ]:
            quality_rows.append({
                "Dataset": label,
                "Observations": len(df),
                "Missing Values": int(df.isna().sum().sum()),
                "Duplicate Applicant IDs": (
                    int(df["applicant_id"].duplicated().sum())
                    if "applicant_id" in df.columns
                    else None
                ),
            })

        st.dataframe(
            pd.DataFrame(quality_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The evidence shown above was generated from "
            "synthetic monitoring scenarios. It does not "
            "represent live production data."
        )

        st.markdown("### Monitoring Investigation")

        st.markdown(
            """
            As a governance committee member, consider:

            - What has changed compared with normal operations?
            - Which monitoring signals deserve attention?
            - Could the issue affect model reliability or fairness?
            - What additional information would help you
              determine the severity of the situation?
            """
        )

        st.text_area(
            "What are the most important monitoring findings?",
            key="m6_monitoring_findings",
            height=120,
            placeholder=(
                "Summarize the monitoring evidence and "
                "explain why it matters."
            ),
        )

        st.text_area(
            "What additional evidence would you request?",
            key="m6_additional_evidence",
            height=100,
            placeholder=(
                "Identify information needed to understand "
                "the potential impact of this scenario."
            ),
        )

        st.markdown(
            """
            **Next:** Open the **Evaluate Guardrails** tab to
            compare your findings with the organization's
            monitoring policies.
            """
        )

    # --------------------------------------------------------
    # TAB 2: EVALUATE GOVERNANCE GUARDRAILS
    # --------------------------------------------------------

    with tab_guardrails:

        st.subheader("Evaluate Governance Guardrails")
        # ----------------------------------------------------
        # CURRENT MONITORING RESULTS
        # ----------------------------------------------------

        st.markdown("### Current Monitoring Guardrail Results")

        st.caption(
            "These results are calculated from the selected "
            "monitoring scenario. Compare them with the "
            "organization's monitoring policies before "
            "making your governance assessment."
        )

        st.write(
            "**Selected scenario:**",
            st.session_state["m6_scenario"],
        )

        st.dataframe(
            guardrail_summary,
            use_container_width=True,
            hide_index=True,
        )

        triggered_guardrails = guardrail_summary[
            guardrail_summary["Status"] == "ALERT"
        ]

        if not triggered_guardrails.empty:
            st.warning(
                f"{len(triggered_guardrails)} monitoring "
                "guardrail(s) exceeded their thresholds. "
                "Review the evidence and determine the "
                "appropriate governance response."
            )
        else:
            st.info(
                "No evaluated guardrails exceeded their "
                "thresholds. Some metrics may still require "
                "additional evidence."
            )

        st.markdown("---")
        
        st.markdown(
            """
            Monitoring guardrails establish the conditions
            under which an AI model can continue operating,
            requires investigation, or must be escalated.

            Review the organization's monitoring policies
            and compare them with the evidence from your
            selected scenario.
            """
        )

        # ----------------------------------------------------
        # MONITORING POLICIES
        # ----------------------------------------------------

        st.markdown("### Organizational Monitoring Policies")

        st.dataframe(
            monitoring_policies,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These policies were established as part of the "
            "organization's simulated AI governance framework."
        )

        # ----------------------------------------------------
        # HISTORICAL MONITORING AUDIT LOG
        # ----------------------------------------------------

        st.markdown("### Monitoring Audit History")

        st.markdown(
            """
            The audit log provides a record of monitoring
            observations and governance actions.

            Review the history to understand how monitoring
            findings have been documented and escalated.
            """
        )

        st.dataframe(
            monitoring_audit,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # POLICY ASSESSMENT
        # ----------------------------------------------------

        st.markdown("### Your Guardrail Assessment")

        st.markdown(
            f"**Selected scenario:** "
            f"{st.session_state['m6_scenario']}"
        )

        st.multiselect(
            "Which monitoring areas require governance attention?",
            [
                "Data Quality",
                "Data Drift",
                "Model Performance",
                "Model Fairness",
                "Operational Reliability",
                "Additional Investigation Needed",
            ],
            key="m6_guardrail_concerns",
        )

        st.radio(
            "How would you characterize the monitoring evidence?",
            [
                "Within Established Guardrails",
                "Potential Policy Violation",
                "Confirmed Policy Violation",
                "Insufficient Evidence",
            ],
            key="m6_policy_assessment",
        )

        st.text_area(
            "Explain your assessment of the monitoring policies.",
            key="m6_guardrail_rationale",
            height=130,
            placeholder=(
                "Identify the relevant monitoring policies, "
                "explain how the scenario evidence relates to "
                "those policies, and discuss any uncertainty."
            ),
        )

        st.markdown(
            """
            **Next:** Open the **Decide & Escalate** tab to
            recommend an operational response and safeguards.
            """
        )

    # --------------------------------------------------------
    # TAB 3: DECIDE AND ESCALATE
    # --------------------------------------------------------

    with tab_response:

        st.subheader("Decide & Escalate")

        # ----------------------------------------------------
        # GOVERNANCE DECISION SUPPORT
        # ----------------------------------------------------

        st.markdown("### Monitoring Evidence for Your Decision")

        st.caption(
            "Review the monitoring results before selecting "
            "an operational response. A triggered guardrail "
            "requires evaluation but does not automatically "
            "determine the appropriate action."
        )

        st.write(
            "**Selected scenario:**",
            st.session_state["m6_scenario"],
        )

        st.dataframe(
            guardrail_summary,
            use_container_width=True,
            hide_index=True,
        )

        triggered = guardrail_summary[
            guardrail_summary["Status"] == "ALERT"
        ]

        unevaluated = guardrail_summary[
            guardrail_summary["Status"] == "Not Evaluated"
        ]

        if not triggered.empty:
            st.warning(
                f"{len(triggered)} guardrail(s) exceeded "
                "their monitoring thresholds. Consider "
                "the severity, potential consequences, "
                "and appropriate safeguards."
            )
        else:
            st.info(
                "No evaluated guardrails exceeded their "
                "thresholds in this scenario."
            )

        if not unevaluated.empty:
            st.info(
                f"{len(unevaluated)} guardrail(s) could not "
                "be evaluated with the available evidence. "
                "Consider whether additional investigation "
                "is necessary."
            )

        st.markdown("---")

        st.markdown(
            """
            As a member of the AI Governance Committee,
            recommend an appropriate operational response
            to the monitoring scenario you investigated.

            Consider the severity of the issue, the
            available evidence, and the potential impact
            on borrowers and the organization.
            """
        )

        st.markdown(
            f"**Selected scenario:** "
            f"{st.session_state['m6_scenario']}"
        )

        # ----------------------------------------------------
        # OPERATIONAL RESPONSE
        # ----------------------------------------------------

        st.markdown("### Operational Response")

        st.radio(
            "What action should the organization take?",
            [
                "Continue Operations",
                "Investigate Further",
                "Continue with Additional Controls",
                "Temporarily Suspend Model",
            ],
            key="m6_response",
        )

        st.text_area(
            "Explain your recommended operational response.",
            key="m6_response_rationale",
            height=120,
            placeholder=(
                "Explain why your selected response is "
                "appropriate based on the monitoring evidence."
            ),
        )

        # ----------------------------------------------------
        # RISK MITIGATION
        # ----------------------------------------------------

        st.markdown("### Risk Mitigation and Safeguards")

        st.multiselect(
            "Which safeguards or corrective actions would you recommend?",
            [
                "Investigate Data Quality",
                "Conduct Data Drift Analysis",
                "Reevaluate Model Performance",
                "Conduct Additional Fairness Testing",
                "Increase Monitoring Frequency",
                "Require Human Review",
                "Restrict Automated Decisions",
                "Initiate Model Retraining Review",
                "Escalate to Risk and Compliance Teams",
                "Document and Audit the Incident",
            ],
            key="m6_safeguards",
        )

        st.text_area(
            "How would these safeguards address the identified risks?",
            key="m6_safeguard_rationale",
            height=120,
            placeholder=(
                "Connect your proposed safeguards to "
                "specific monitoring findings."
            ),
        )

        # ----------------------------------------------------
        # OPERATIONAL RISK RATING
        # ----------------------------------------------------

        st.markdown("### Operational Risk Rating")

        st.radio(
            "Assign an overall risk rating.",
            [
                "Low",
                "Moderate",
                "High",
            ],
            key="m6_risk_rating",
        )

        st.text_area(
            "Explain your risk rating.",
            key="m6_risk_rationale",
            height=100,
            placeholder=(
                "Consider potential harm, severity, "
                "uncertainty, and existing controls."
            ),
        )

        # ----------------------------------------------------
        # ESCALATION RESPONSIBILITIES
        # ----------------------------------------------------

        st.markdown("### Escalation and Accountability")

        st.multiselect(
            "Who should be notified or involved?",
            [
                "Model Owner",
                "Data Science Team",
                "Model Risk Management",
                "Compliance and Legal",
                "Business Operations",
                "Executive Risk Committee",
                "Internal Audit",
            ],
            key="m6_escalation",
        )

        st.text_area(
            "Describe the responsibilities of the selected stakeholders.",
            key="m6_escalation_rationale",
            height=110,
            placeholder=(
                "Explain who should investigate, "
                "approve corrective actions, and "
                "authorize resumption if needed."
            ),
        )

        # ----------------------------------------------------
        # RECOVERY CRITERIA
        # ----------------------------------------------------

        st.markdown("### Recovery and Return to Normal Operations")

        st.text_area(
            "What conditions must be met before normal operations can resume?",
            key="m6_recovery",
            height=130,
            placeholder=(
                "Identify the evidence, validation, "
                "approvals, and monitoring conditions "
                "required to close the incident."
            ),
        )

        st.markdown(
            """
            **Next:** Open **Submit Incident Assessment**
            to document and download your governance decision.
            """
        )

    # --------------------------------------------------------
    # TAB 4: SUBMIT INCIDENT ASSESSMENT
    # --------------------------------------------------------

    with tab_submit:

        st.subheader("Submit Incident Assessment")

        st.markdown(
            """
            Review your findings and recommendations before
            generating your Module 6 assessment.

            Your report will document the monitoring evidence,
            governance policy assessment, operational response,
            safeguards, escalation responsibilities, and
            recovery criteria.
            """
        )

        student_name = st.text_input(
            "Student Name",
            key="m6_student_name",
        )

        st.markdown("### Assessment Summary")

        st.write(
            "**Scenario:**",
            st.session_state.get("m6_scenario", ""),
        )

        st.write(
            "**Policy Assessment:**",
            st.session_state.get("m6_policy_assessment", ""),
        )

        st.write(
            "**Operational Response:**",
            st.session_state.get("m6_response", ""),
        )

        st.write(
            "**Risk Rating:**",
            st.session_state.get("m6_risk_rating", ""),
        )

        st.markdown("### Final Governance Rationale")

        st.text_area(
            "Summarize your overall governance recommendation.",
            key="m6_final_rationale",
            height=150,
            placeholder=(
                "Explain how the monitoring evidence, "
                "organizational guardrails, and potential "
                "risks support your final recommendation."
            ),
        )

        def report_text(value):
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(item) for item in value)
            return str(value or "Not provided")

        def generate_module6_pdf():
            buffer = BytesIO()

            doc = SimpleDocTemplate(
                buffer,
                pagesize=(8.5 * inch, 11 * inch),
                rightMargin=48,
                leftMargin=48,
                topMargin=48,
                bottomMargin=48,
            )

            styles = getSampleStyleSheet()
            story = []

            def add_heading(value):
                story.append(
                    Paragraph(
                        escape(str(value)),
                        styles["Heading2"],
                    )
                )

            def add_text(value):
                safe = escape(report_text(value)).replace(
                    "\n", "<br/>"
                )
                story.append(
                    Paragraph(safe, styles["BodyText"])
                )
                story.append(Spacer(1, 8))

            story.append(
                Paragraph(
                    "Module 6: Monitoring and Guardrails",
                    styles["Title"],
                )
            )

            add_text("AI Governance Committee Incident Assessment")
            add_text(f"Student: {student_name}")
            add_text(
                f"Date: {datetime.now().strftime('%B %d, %Y')}"
            )
            
            # ------------------------------------------------
            # MONITORING GUARDRAIL EVIDENCE
            # ------------------------------------------------

            add_heading("Monitoring Guardrail Summary")

            add_text(
                "The following results were calculated "
                "from the selected monitoring scenario "
                "using the simulation's monitoring rules."
            )

            from reportlab.platypus import (
                Table as PDFTable,
                TableStyle as PDFTableStyle,
            )
            from reportlab.lib import colors as pdf_colors

            evidence_rows = [
                ["Policy", "Observed", "Threshold", "Status"]
            ]

            for _, evidence in guardrail_summary.iterrows():
                evidence_rows.append([
                    str(evidence["Policy"]),
                    str(evidence["Observed"]),
                    str(evidence["Threshold"]),
                    str(evidence["Status"]),
                ])

            evidence_table = PDFTable(
                evidence_rows,
                colWidths=[155, 75, 90, 105],
                repeatRows=1,
                hAlign="LEFT",
            )

            evidence_table.setStyle(
                PDFTableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        pdf_colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        pdf_colors.grey,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ])
            )

            story.append(evidence_table)
            story.append(Spacer(1, 14))

            add_text(
                "M001 evaluates missing applicant data; "
                "M002 evaluates invalid applicant records "
                "using classroom validation assumptions; "
                "M003 reports the maximum evaluated PSI; "
                "M004 requires prediction scores and is "
                "not evaluated with the current data. "
                "An alert does not automatically require "
                "suspending model operations."
            )
            
            fields = [
                ("Monitoring Scenario", "m6_scenario"),
                ("Monitoring Findings", "m6_monitoring_findings"),
                ("Additional Evidence Requested", "m6_additional_evidence"),
                ("Guardrail Concerns", "m6_guardrail_concerns"),
                ("Policy Assessment", "m6_policy_assessment"),
                ("Policy Rationale", "m6_guardrail_rationale"),
                ("Operational Response", "m6_response"),
                ("Response Rationale", "m6_response_rationale"),
                ("Recommended Safeguards", "m6_safeguards"),
                ("Safeguard Rationale", "m6_safeguard_rationale"),
                ("Operational Risk Rating", "m6_risk_rating"),
                ("Risk Rationale", "m6_risk_rationale"),
                ("Escalation Stakeholders", "m6_escalation"),
                ("Escalation Responsibilities", "m6_escalation_rationale"),
                ("Recovery Criteria", "m6_recovery"),
                ("Final Governance Rationale", "m6_final_rationale"),
            ]

            for label, key in fields:
                add_heading(label)
                add_text(st.session_state.get(key, ""))

            add_text(
                "Educational simulation using synthetic lending data. "
                "This assessment does not authorize deployment."
            )

            doc.build(story)
            result = buffer.getvalue()
            buffer.close()
            return result

        required_fields = [
            "m6_monitoring_findings",
            "m6_guardrail_rationale",
            "m6_response_rationale",
            "m6_safeguard_rationale",
            "m6_risk_rationale",
            "m6_recovery",
            "m6_final_rationale",
        ]

        missing_fields = [
            field for field in required_fields
            if not str(st.session_state.get(field, "")).strip()
        ]

        if not student_name.strip():
            st.warning("Enter your name before downloading.")

        elif missing_fields:
            st.warning(
                "Complete all required written assessments "
                "before downloading your report."
            )

        else:
            pdf_bytes = generate_module6_pdf()

            st.download_button(
                "Download Module 6 Incident Assessment PDF",
                data=pdf_bytes,
                file_name="Module_6_Incident_Assessment.pdf",
                mime="application/pdf",
                key="m6_pdf_download",
            )

        st.info(
            "Download your completed PDF before closing the "
            "application. Submit the PDF through Canvas."
        )

