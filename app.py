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
# MODULE 6 PLACEHOLDER
# ============================================================

elif module == "Module 6: Guardrails & Monitoring":

    st.header("Module 6: Guardrails & Monitoring")

    st.info(
        "Coming next: Production monitoring dashboards, "
        "data drift detection, simulated incidents, "
        "and governance response recommendations."
    )
