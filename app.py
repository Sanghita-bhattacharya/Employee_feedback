    """
    AI-Powered Anonymous Employee Feedback & Team Retention Intelligence System.
    Streamlit Application with Executive Overview, RAG Framework, Anonymization Testing,
    Team Attrition Prediction, Model Benchmarks, Human Review Queue, and Privacy Governance.
    """

    import os
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    # Import internal modules
    from src.anonymizer import AnonymizerEngine
    from src.classifier import get_classifier
    from src.attrition_predictor import get_attrition_predictor, FEATURE_COLS
    from src.evaluation import evaluate_attrition_models, get_synthetic_data_limitations
    from src.rag_framework import RAGRiskFramework
    from src.rag_retrieval import get_rag_retriever
    from src.llm_engine import get_llm_engine, LLMEngine
    from src.utils import (
        load_feedback_data,
        load_team_features,
        compute_live_team_features,
        enforce_k_anonymity_filter,
        call_llm_if_available,
        MIN_K_ANONYMITY
    )

    # Page configuration
    st.set_page_config(
        page_title="EchoGuard AI | Anonymous Feedback & Team Retention",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Styling (Dark theme, glassmorphism, modern typography)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
            padding: 1.75rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.25);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border-color: rgba(99, 102, 241, 0.4);
        }
        
        .badge-red {
            background-color: rgba(239, 68, 68, 0.2);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        .badge-amber {
            background-color: rgba(245, 158, 11, 0.2);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        .badge-green {
            background-color: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        .badge-blue {
            background-color: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        .privacy-callout {
            background: rgba(15, 23, 42, 0.85);
            border-left: 4px solid #6366F1;
            padding: 1rem 1.25rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "escalation_notes" not in st.session_state:
        st.session_state["escalation_notes"] = {}
    if "escalation_status_override" not in st.session_state:
        st.session_state["escalation_status_override"] = {}
    if "custom_feedbacks" not in st.session_state:
        st.session_state["custom_feedbacks"] = []
    if "last_submission_impact" not in st.session_state:
        st.session_state["last_submission_impact"] = None
    if "portal_role" not in st.session_state:
        st.session_state["portal_role"] = "👤 Employee Portal"
    if "emp_submitted_receipt" not in st.session_state:
        st.session_state["emp_submitted_receipt"] = None
    if "emp_feedback_input" not in st.session_state:
        st.session_state["emp_feedback_input"] = ""

    # Load Data
    @st.cache_data
    def get_datasets():
        df_fb = load_feedback_data()
        df_tm = load_team_features()
        return df_fb, df_tm

    df_feedback_base, df_teams_base = get_datasets()

    # Incorporate live custom submissions into active feedback dataframe and dynamically update team features
    if st.session_state.get("custom_feedbacks"):
        df_custom = pd.DataFrame(st.session_state["custom_feedbacks"])
        df_feedback = pd.concat([df_custom, df_feedback_base], ignore_index=True)
        df_teams = compute_live_team_features(df_feedback, df_teams_base)
    else:
        df_feedback = df_feedback_base.copy()
        df_teams = df_teams_base.copy()

    anonymizer_engine = AnonymizerEngine()
    classifier_engine = get_classifier()
    attrition_engine = get_attrition_predictor()
    rag_retriever = get_rag_retriever(df_feedback)

    # Sidebar Setup
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=120&q=80", width=64)
        st.markdown("### **EchoGuard AI**")
        st.caption("Ethical Anonymous Feedback & Retention Intelligence")
        st.markdown("---")

        # Portal Role Selector (Employee vs HR)
        portal_role = st.radio(
            "Current Portal View",
            [
                "👤 Employee Portal",
                "🏢 HR & Leadership Suite"
            ],
            index=0 if "Employee" in st.session_state.get("portal_role", "Employee") else 1,
            key="portal_role_radio"
        )
        st.session_state["portal_role"] = portal_role

        if portal_role == "👤 Employee Portal":
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 0.8rem; color: #34D399;">
                    🔒 <b>Safe Harbor Channel Active</b><br>
                    <span style="color: #CBD5E1; font-size: 0.74rem;">
                        Submit feedback with 100% confidentiality. Your identity is stripped before HR receives it.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
            emp_menu = st.radio(
                "Employee Portal Menu",
                [
                    "📝 Submit Anonymous Feedback",
                    "🛡️ Employee Privacy & Whistleblower Rights"
                ],
                index=0,
                key="emp_nav_radio"
            )
            
            # Default internal HR thresholds so backend calculations run smoothly
            red_thresh = 0.65
            amber_thresh = 0.35
            neg_sent_thresh = 0.45
            model_code = "hybrid"
            llm_inst = get_llm_engine(provider="auto")
            prov_info = llm_inst.get_active_provider_info()
            menu = None

        else:
            st.markdown(
                """
                <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; border-radius: 8px; padding: 10px; margin-top: 8px; font-size: 0.8rem; color: #C7D2FE;">
                    👔 <b>HR & Executive Analytics Suite</b><br>
                    <span style="color: #94A3B8; font-size: 0.74rem;">Full organizational health metrics, RAG risk matrix, and ML attrition predictions.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
            menu = st.radio(
                "HR Suite Navigation",
                [
                    "📊 Executive Overview",
                    "🎯 RAG Risk Matrix & Teams",
                    "🔍 Anonymised Feedback Explorer",
                    "🧪 AI Testing & Redaction Sandbox",
                    "📈 Team Attrition Prediction Lab",
                    "📉 Model Evaluation & Benchmarks",
                    "🚨 Escalation & Human Review Queue",
                    "🛡️ Privacy, Ethics & Methodology"
                ],
                index=0,
                key="main_nav_radio"
            )

            if st.session_state.get("custom_feedbacks"):
                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 8px; padding: 8px 10px; margin-top: 8px; font-size: 0.78rem; color: #C7D2FE;">
                        <b>📥 Live Submissions Active:</b> {len(st.session_state['custom_feedbacks'])} reviews
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("🔄 Reset to Baseline", use_container_width=True):
                    st.session_state["custom_feedbacks"] = []
                    st.session_state["last_submission_impact"] = None
                    st.rerun()

            st.markdown("---")
            st.markdown("#### **⚙️ RAG Risk Thresholds**")
            red_thresh = st.slider("Red (Critical) Attrition Prob", 0.50, 0.90, 0.65, 0.05, help="Cutoff to trigger Critical RED team alert")
            amber_thresh = st.slider("Amber (Elevated) Prob", 0.20, 0.50, 0.35, 0.05, help="Cutoff for Moderate AMBER warning")
            neg_sent_thresh = st.slider("Negative Sentiment Alert (%)", 30, 70, 45, 5) / 100.0

            st.markdown("#### **🤖 Prediction Model**")
            selected_model = st.selectbox(
                "Active Predictor",
                [
                    "Hybrid AI (LLM Reasoner + Random Forest)",
                    "Random Forest (Ensemble ML)",
                    "AI LLM Predictive Reasoner",
                    "Gradient Boosting (Boosted Trees ML)",
                    "Logistic Regression (Linear ML)",
                    "Dummy Classifier (Baseline ML)"
                ],
                index=0
            )
            model_code = (
                "hybrid" if "Hybrid AI" in selected_model else (
                    "llm" if "AI LLM" in selected_model else (
                        "rf" if "Random Forest" in selected_model else (
                            "gb" if "Gradient Boosting" in selected_model else (
                                "logreg" if "Logistic" in selected_model else "dummy"
                            )
                        )
                    )
                )
            )

            st.markdown("---")
            st.markdown("#### **🧠 AI LLM Engine**")
            llm_provider_choice = st.selectbox(
                "LLM Provider",
                [
                    "Auto-Detect (Best Available)",
                    "Groq (Llama 3.3 70B)",
                    "OpenAI (GPT-4o Mini)",
                    "Google Gemini (1.5 Flash)",
                    "Local Ollama (Offline Llama 3)",
                    "Local AI Reasoner (Zero-Key Simulation)"
                ],
                index=0,
                help="Select LLM provider for generative reasoning, root-cause analysis, and executive briefings."
            )
            provider_code_map = {
                "Auto-Detect (Best Available)": "auto",
                "Groq (Llama 3.3 70B)": "groq",
                "OpenAI (GPT-4o Mini)": "openai",
                "Google Gemini (1.5 Flash)": "gemini",
                "Local Ollama (Offline Llama 3)": "ollama",
                "Local AI Reasoner (Zero-Key Simulation)": "simulation"
            }
            provider_code = provider_code_map[llm_provider_choice]
            custom_api_key = st.text_input(
                "API Key (Optional)",
                type="password",
                placeholder="Enter Groq / OpenAI / Gemini Key...",
                help="Leave blank to use environment variables or local offline simulation reasoner."
            )
            llm_inst = get_llm_engine(provider=provider_code, api_key=custom_api_key.strip() if custom_api_key.strip() else None)
            prov_info = llm_inst.get_active_provider_info()

            if not prov_info["is_simulation"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; color: #34D399; margin-top: 4px;">
                        🟢 <b>Active:</b> {prov_info['name']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; color: #FCD34D; margin-top: 4px;">
                        🟡 <b>Mode:</b> {prov_info['name']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 8px; padding: 10px; font-size: 0.8rem;">
                    <b>🔒 Privacy Safeguards Active</b><br>
                    • Minimum group size: <b>k ≥ {MIN_K_ANONYMITY}</b><br>
                    • Strict Team-Level ML Only<br>
                    • Zero PII Storage
                </div>
                """,
                unsafe_allow_html=True
            )

    # Instantiate RAG framework with active thresholds
    rag_framework = RAGRiskFramework(
        red_prob_threshold=red_thresh,
        amber_prob_threshold=amber_thresh,
        red_neg_sentiment_threshold=neg_sent_thresh
    )
    df_rag_evaluated = rag_framework.evaluate_all_teams(df_teams, model_type=model_code)


    # ==============================================================================
    # ROUTER: EMPLOYEE SAFE HARBOR PORTAL VS HR & LEADERSHIP SUITE
    # ==============================================================================
    if portal_role == "👤 Employee Portal":
        if emp_menu == "📝 Submit Anonymous Feedback":
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                            border: 1.5px solid rgba(16, 185, 129, 0.4); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.5rem;
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #34D399; font-weight: 800;">
                                🛡️ ECHO GUARD ANONYMOUS EMPLOYEE SAFE HARBOR
                            </span>
                            <h1 style="color: #FFFFFF; margin: 4px 0 6px 0; font-size: 2.1rem; font-weight: 800;">
                                Submit Anonymous Employee Feedback
                            </h1>
                            <p style="color: #94A3B8; margin: 0; font-size: 0.95rem; max-width: 800px; line-height: 1.5;">
                                Speak your mind openly and without fear. Your feedback is processed through real-time privacy cloaking 
                                so your name, email, employee ID, and contact details are permanently scrubbed before reaching HR or leadership.
                            </p>
                        </div>
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 10px; padding: 8px 14px; text-align: center;">
                            <span style="color: #10B981; font-size: 1.4rem;">🔒</span>
                            <div style="color: #34D399; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-top: 2px;">Zero-PII Storage</div>
                            <div style="color: #A7F3D0; font-size: 0.68rem;">100% Identity Guarded</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Three trust pill banners
            c_tr1, c_tr2, c_tr3 = st.columns(3)
            with c_tr1:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; color: #E2E8F0;">
                        🛡️ <b>Anti-Retaliation Guard:</b> Non-retaliation policy strictly enforced under corporate whistleblower governance.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_tr2:
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; color: #E2E8F0;">
                        👥 <b>k-Anonymity (k ≥ {MIN_K_ANONYMITY}):</b> Data is only shared in aggregated team pulses with at least 5 people.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_tr3:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; color: #E2E8F0;">
                        🚫 <b>No Surveillance:</b> No IP addresses, device signatures, or browser fingerprints are ever logged.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

            # Submission Confirmation Receipt
            if st.session_state.get("emp_submitted_receipt"):
                receipt = st.session_state["emp_submitted_receipt"]
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(15, 23, 42, 0.98) 100%);
                                border: 2px solid #10B981; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.5rem;
                                box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);">
                        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.75rem;">
                            <span style="font-size: 2.4rem;">🎉</span>
                            <div>
                                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6EE7B7; font-weight: 800;">
                                    SUBMISSION SUCCESSFUL & CONFIDENTIALITY GUARANTEED
                                </span>
                                <h3 style="color: #FFFFFF; margin: 2px 0 0 0; font-size: 1.35rem; font-weight: 800;">
                                    Your Feedback Has Been Anonymously Recorded
                                </h3>
                            </div>
                        </div>
                        <div style="background: rgba(0,0,0,0.4); border-left: 3px solid #10B981; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 0.88rem; color: #E2E8F0;">
                            <b>🔒 Anonymized Record Forwarded to HR Pulse (Zero Personal Names):</b><br>
                            <i>"{receipt['scrubbed_text']}"</i>
                        </div>
                        <div style="display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.82rem; color: #94A3B8;">
                            <span><b>Anonymous Receipt Token:</b> <code style="color: #6EE7B7; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">{receipt['id']}</code></span>
                            <span>•</span>
                            <span><b>Department:</b> {receipt['dept']}</span>
                            <span>•</span>
                            <span><b>Target Team:</b> {receipt['team']}</span>
                            <span>•</span>
                            <span><b>Submitted:</b> {receipt['time']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col_rec1, col_rec2 = st.columns([1, 3])
                with col_rec1:
                    if st.button("📝 Submit Another Feedback", use_container_width=True, key="btn_emp_sub_another"):
                        st.session_state["emp_submitted_receipt"] = None
                        st.session_state["emp_feedback_input"] = ""
                        st.rerun()
                with col_rec2:
                    if st.button("✖️ Dismiss Confirmation", use_container_width=True, key="btn_emp_dismiss_rec"):
                        st.session_state["emp_submitted_receipt"] = None
                        st.rerun()

            # Step 1: Affiliation
            st.markdown("#### **1. Select Your Affiliation (Aggregated Team Pulse)**")
            col_emp_dept, col_emp_team = st.columns(2)
            with col_emp_dept:
                available_depts = sorted(df_feedback["department"].dropna().unique())
                emp_dept = st.selectbox("Your Department", available_depts, index=0, key="emp_dept_select")
            with col_emp_team:
                available_teams = sorted(df_feedback[df_feedback["department"] == emp_dept]["team_id"].dropna().unique())
                emp_team = st.selectbox("Your Team / Group ID", available_teams, index=0, key="emp_team_select")

            # Step 2: Privacy Cloaking Style
            st.markdown("#### **2. Privacy Redaction Cloaking Style**")
            emp_cloaking_style = st.radio(
                "Select how colleague names or personal references should be redacted before reaching HR:",
                [
                    "👤 Natural Colleague Cloaking (e.g. 'Ram is bad' becomes 'The colleague is bad')",
                    "🏷️ Cryptographic Safeguard Tokens (e.g. 'Ram is bad' becomes '[NAME] is bad')"
                ],
                index=0,
                key="emp_cloak_style_radio"
            )
            is_natural_emp = "Natural Colleague" in emp_cloaking_style

            # Step 3: Write Feedback
            st.markdown("#### **3. Write Your Feedback**")
            def set_emp_quick_text(txt: str):
                st.session_state["emp_feedback_input"] = txt

            emp_text = st.text_area(
                "Share your thoughts, suggestions, workload concerns, or management feedback freely:",
                value=st.session_state.get("emp_feedback_input", ""),
                placeholder="Type your feedback here freely... e.g. 'Ram is bad and team lead does not listen' or 'Great support from my team lead and collaborative sprint culture'...",
                height=130,
                key="emp_text_area_input"
            )

            st.caption("💡 *Quick Topic Starters (Click any chip to populate example feedback):*")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.button("🌟 Great Culture", on_click=set_emp_quick_text, args=("Great support from my team lead and manager. I really enjoy the collaborative culture!",), use_container_width=True, key="emp_chip1")
            with c2:
                st.button("💸 Pay & Bonuses", on_click=set_emp_quick_text, args=("Salary is too low for the cost of living and we received no bonuses this year.",), use_container_width=True, key="emp_chip2")
            with c3:
                st.button("💔 Burnout", on_click=set_emp_quick_text, args=("Our manager does not care about team burnout and micromanages all daily tasks.",), use_container_width=True, key="emp_chip3")
            with c4:
                st.button("⚙️ Slow Tools", on_click=set_emp_quick_text, args=("The internal development tools are completely outdated, slow, and frustrating to use.",), use_container_width=True, key="emp_chip4")
            with c5:
                st.button("🚨 Misconduct", on_click=set_emp_quick_text, args=("A senior director subjected me to inappropriate treatment and threatened retaliation.",), use_container_width=True, key="emp_chip5")

            # Step 4: Real-time Privacy Shield Preview
            active_emp_text = emp_text.strip()
            if active_emp_text:
                live_scrubbed = anonymizer_engine.anonymize(active_emp_text, natural=is_natural_emp)
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #10B981; border-radius: 10px; padding: 12px 16px; margin-top: 14px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.78rem; text-transform: uppercase; color: #34D399; font-weight: 700;">
                                🛡️ Live Identity Protection Shield (What HR Will Receive)
                            </span>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #A7F3D0; font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                                ✓ 100% PII-Scrubbed
                            </span>
                        </div>
                        <div style="background: rgba(0, 0, 0, 0.4); border-left: 3px solid #10B981; padding: 8px 12px; border-radius: 6px; font-size: 0.92rem; color: #FFFFFF; font-style: italic;">
                            "{live_scrubbed['anonymised_text']}"
                        </div>
                        <div style="color: #6EE7B7; font-size: 0.75rem; margin-top: 6px;">
                            🔒 All personal names (e.g. Ram, Alex), phone numbers, email addresses, and employee badge IDs are scrubbed automatically before saving.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            col_sub1, col_sub2 = st.columns([2.5, 1])
            with col_sub1:
                submit_emp_btn = st.button("📤 Submit Anonymous Feedback", type="primary", use_container_width=True, key="btn_emp_submit_final")
            with col_sub2:
                st.button("🧹 Clear Box", on_click=set_emp_quick_text, args=("",), use_container_width=True, key="btn_emp_clear_final")

            if submit_emp_btn:
                if not active_emp_text:
                    st.warning("⚠️ Please enter feedback text before submitting.")
                else:
                    anon_res = anonymizer_engine.anonymize(active_emp_text, natural=is_natural_emp)
                    clf_res = classifier_engine.classify(active_emp_text, anon_res["anonymised_text"])
                    new_id = f"FB-CUSTOM-{len(df_feedback) + 1}"
                    new_rec = {
                        "feedback_id": new_id,
                        "department": emp_dept,
                        "team_id": emp_team,
                        "theme": clf_res["theme"],
                        "sentiment": clf_res["sentiment"],
                        "severity": clf_res["severity"],
                        "privacy_risk": anon_res["privacy_risk"],
                        "escalation_status": clf_res["escalation_status"],
                        "feedback_text": anon_res["anonymised_text"],  # 100% Zero-PII stored!
                        "anonymised_text": anon_res["anonymised_text"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state["custom_feedbacks"].insert(0, new_rec)

                    # Pre & Post metrics snapshot for HR Suite
                    cur_slice = df_teams[df_teams["team_id"] == emp_team]
                    pre_t = cur_slice.iloc[0].to_dict() if not cur_slice.empty else {}
                    df_new_fbs = pd.concat([pd.DataFrame([new_rec]), df_feedback], ignore_index=True)
                    df_new_tms = compute_live_team_features(df_new_fbs, df_teams_base)
                    post_slice = df_new_tms[df_new_tms["team_id"] == emp_team]
                    post_t = post_slice.iloc[0].to_dict() if not post_slice.empty else {}

                    st.session_state["last_submission_impact"] = {
                        "feedback_id": new_id,
                        "team_id": emp_team,
                        "department": emp_dept,
                        "sentiment": clf_res["sentiment"],
                        "theme": clf_res["theme"],
                        "severity": clf_res["severity"],
                        "raw_text": active_emp_text,
                        "anonymised_text": anon_res["anonymised_text"],
                        "timestamp": new_rec["timestamp"],
                        "pre_sentiment": pre_t.get("avg_sentiment_score", 0.0),
                        "post_sentiment": post_t.get("avg_sentiment_score", 0.0),
                        "pre_neg_pct": pre_t.get("pct_negative", 0.0),
                        "post_neg_pct": post_t.get("pct_negative", 0.0),
                        "pre_pos_pct": pre_t.get("pct_positive", 0.0),
                        "post_pos_pct": post_t.get("pct_positive", 0.0),
                        "pre_risk": pre_t.get("continuous_risk_score", 0.0),
                        "post_risk": post_t.get("continuous_risk_score", 0.0),
                        "pre_count": int(pre_t.get("feedback_count", 0)),
                        "post_count": int(post_t.get("feedback_count", 0)),
                        "pre_total": len(df_feedback),
                        "post_total": len(df_new_fbs),
                        "valence_score": clf_res.get("valence_score", 0.0)
                    }
                    st.session_state["emp_submitted_receipt"] = {
                        "id": new_id,
                        "dept": emp_dept,
                        "team": emp_team,
                        "scrubbed_text": anon_res["anonymised_text"],
                        "time": new_rec["timestamp"]
                    }
                    st.session_state["emp_feedback_input"] = ""
                    st.session_state["feedback_just_submitted"] = True
                    st.balloons()
                    st.rerun()

            # Switch to HR suite link at the bottom
            st.markdown("---")
            st.markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; padding: 12px 16px; margin-top: 1.5rem;">
                    <div style="font-size: 0.85rem; color: #94A3B8;">
                        👔 <b>HR Administrator or Leader?</b> Switch to the full analytics, RAG risk matrix, and governance suite.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Switch to HR & Leadership Suite ➔", key="btn_switch_to_hr_bottom"):
                st.session_state["portal_role"] = "🏢 HR & Leadership Suite"
                st.rerun()

        elif emp_menu == "🛡️ Employee Privacy & Whistleblower Rights":
            st.markdown(
                """
                <div class="main-header">
                    <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                        🛡️ Employee Privacy, Safety & Whistleblower Rights
                    </h1>
                    <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                        Our cryptographic, non-retaliation, and anonymization commitments to every employee.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            ep1, ep2 = st.columns(2)
            with ep1:
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem;">
                        <h3 style="color: #34D399; margin: 0 0 8px 0; font-size: 1.15rem;">🔒 Zero-PII Redaction Guarantee</h3>
                        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.5;">
                            EchoGuard AI employs advanced Natural Language Processing and Named Entity Recognition (NER) to detect and redact all:
                        </p>
                        <ul style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 10px 18px;">
                            <li>Personal names (e.g. "Ram", "Alex Johnson", "Pooja")</li>
                            <li>Phone numbers and personal emails</li>
                            <li>Employee badge numbers (e.g. EMP-4821)</li>
                            <li>Confidential internal project codes</li>
                        </ul>
                        <p style="color: #34D399; font-size: 0.85rem; font-weight: 600; margin: 0;">
                            Raw un-anonymized text is never written to disk or database.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with ep2:
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem;">
                        <h3 style="color: #A5B4FC; margin: 0 0 8px 0; font-size: 1.15rem;">⚖️ Strict Whistleblower Protection</h3>
                        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.5;">
                            Retaliation against any employee reporting workplace concerns is illegal and strictly prohibited by company policy.
                        </p>
                        <ul style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 10px 18px;">
                            <li><b>Automatic Escalation:</b> Severe harassment, ethical fraud, or safety violations are automatically flagged for independent confidential compliance review.</li>
                            <li><b>k-Anonymity (k ≥ {MIN_K_ANONYMITY}):</b> HR only views aggregate sentiment for teams with at least 5 people.</li>
                            <li><b>Zero Surveillance:</b> No IP logging, no device tracking, and no browser cookies tied to feedback.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if st.button("📝 Return to Submit Feedback", type="primary"):
                st.session_state["emp_nav_radio"] = "📝 Submit Anonymous Feedback"
                st.rerun()

        # Employee portal execution ends here! Prevent any HR dashboards from rendering for employees.
        st.stop()

    # ==============================================================================
    # HR & LEADERSHIP INTELLIGENCE SUITE
    # ==============================================================================
    # Top bar showing HR mode status + quick switcher to Employee portal
    col_hr_banner, col_hr_sw = st.columns([3.5, 1.5])
    with col_hr_banner:
        st.markdown(
            """
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #C7D2FE;">
                🏢 <b>HR & Executive Mode Active:</b> You are viewing confidential organizational health analytics.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_hr_sw:
        if st.button("👤 Switch to Employee Portal ➔", key="btn_hr_to_emp_top", use_container_width=True):
            st.session_state["portal_role"] = "👤 Employee Portal"
            st.rerun()

    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # PAGE 1: EXECUTIVE OVERVIEW
    # ==============================================================================
    if menu == "📊 Executive Overview":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🛡️ Executive Feedback & Retention Overview
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Real-time organizational health pulse, anonymous feedback sentiment, and proactive retention intelligence.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Top KPI Metrics Calculation
        total_fb = len(df_feedback)
        n_teams = len(df_teams)
        n_red = (df_rag_evaluated["status"] == "RED").sum()
        n_amber = (df_rag_evaluated["status"] == "AMBER").sum()
        n_green = (df_rag_evaluated["status"] == "GREEN").sum()
        total_escalations = (df_feedback["escalation_status"] == "Escalated - Human Review Required").sum()

        # Urgent Critical Crisis Alert Banner (If RED teams or escalations exist)
        if n_red > 0 or total_escalations > 0:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(127, 29, 29, 0.85) 0%, rgba(69, 10, 10, 0.95) 100%); 
                            border: 2px solid #EF4444; border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; 
                            box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <span style="font-size: 2.2rem;">🚨</span>
                            <div>
                                <div style="color: #FEE2E2; font-size: 1.2rem; font-weight: 800; letter-spacing: 0.02em;">
                                    CRITICAL ORGANIZATIONAL CRISIS DETECTED
                                </div>
                                <div style="color: #FCA5A5; font-size: 0.92rem; margin-top: 3px;">
                                    <b>{n_red} Teams in Severe Red Alert</b> • <b>{total_escalations} Critical Escalations</b> 
                                    (Harassment, Medical Burnout, Retaliation & Safety Concealment)
                                </div>
                            </div>
                        </div>
                        <div>
                            <span style="background: #EF4444; color: #FFFFFF; padding: 6px 14px; border-radius: 20px; font-weight: 800; font-size: 0.82rem; letter-spacing: 0.05em; display: inline-block;">
                                ⚡ IMMEDIATE HR & EXEC ACTION REQUIRED
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Top KPI Metrics Columns
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">TOTAL FEEDBACK</div>
                <div style="color: #F8FAFC; font-size: 1.8rem; font-weight: 700;">{total_fb}</div>
                <div style="color: #10B981; font-size: 0.75rem;">Verified Anonymous</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">TEAMS MONITORED</div>
                <div style="color: #F8FAFC; font-size: 1.8rem; font-weight: 700;">{n_teams}</div>
                <div style="color: #6366F1; font-size: 0.75rem;">Independent Units</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="border: 1px solid rgba(239, 68, 68, 0.4); background: rgba(127, 29, 29, 0.2);">
                <div style="color: #FCA5A5; font-size: 0.8rem; font-weight: 700;">🚨 RED (CRITICAL)</div>
                <div style="color: #EF4444; font-size: 1.8rem; font-weight: 800;">{n_red}</div>
                <div style="color: #F87171; font-size: 0.75rem; font-weight: 600;">Severe Flight Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border: 1px solid rgba(245, 158, 11, 0.3);">
                <div style="color: #FCD34D; font-size: 0.8rem; font-weight: 600;">⚠️ AMBER (ELEVATED)</div>
                <div style="color: #F59E0B; font-size: 1.8rem; font-weight: 700;">{n_amber}</div>
                <div style="color: #FBBF24; font-size: 0.75rem;">Under Observation</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">GREEN (HEALTHY)</div>
                <div style="color: #10B981; font-size: 1.8rem; font-weight: 700;">{n_green}</div>
                <div style="color: #34D399; font-size: 0.75rem;">Optimal Culture</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card" style="border: 1px solid rgba(236, 72, 153, 0.4); background: rgba(131, 24, 67, 0.2);">
                <div style="color: #F9A8D4; font-size: 0.8rem; font-weight: 700;">⚡ ESCALATIONS</div>
                <div style="color: #EC4899; font-size: 1.8rem; font-weight: 800;">{total_escalations}</div>
                <div style="color: #F472B6; font-size: 0.75rem; font-weight: 600;">Immediate Human Review</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top Emergency Crisis Teams Section
        red_teams_df = df_rag_evaluated[df_rag_evaluated["status"] == "RED"].sort_values(
            by=["critical_count", "attrition_probability"], ascending=False
        ).head(3)

        if not red_teams_df.empty:
            st.markdown("#### **🔥 Highest-Severity Teams Requiring Emergency Intervention:**")
            em_cols = st.columns(len(red_teams_df))
            for i, (_, r_team) in enumerate(red_teams_df.iterrows()):
                with em_cols[i]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(127, 29, 29, 0.35); border: 2px solid #EF4444; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b style="color: #FFFFFF; font-size: 1.05rem;">{r_team['team_id']}</b>
                                <span class="badge-red">CRITICAL RISK</span>
                            </div>
                            <div style="color: #FCA5A5; font-size: 0.85rem; margin: 6px 0;"><b>Dept:</b> {r_team['department']}</div>
                            <div style="color: #EF4444; font-size: 1.25rem; font-weight: 800;">{r_team['attrition_probability']:.1%} Flight Risk</div>
                            <div style="color: #FEE2E2; font-size: 0.82rem; margin-top: 6px; line-height: 1.3;">
                                <b>Trigger:</b> {r_team['key_drivers'][0] if r_team['key_drivers'] else 'Multiple severe risk drivers'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Strategic AI Executive Pulse Briefing
        with st.expander("🤖 **AI Executive Strategic Pulse Briefing**", expanded=False):
            col_pulse_btn, col_pulse_meta = st.columns([1, 2])
            with col_pulse_btn:
                generate_pulse_btn = st.button("✨ Synthesize Executive Briefing", type="primary", use_container_width=True)
            with col_pulse_meta:
                st.markdown(
                    f"<div style='margin-top: 6px; font-size: 0.85rem; color: #94A3B8;'>Active Provider: <b>{prov_info['name']}</b> • Aggregated team metrics only • Zero PII exposure</div>",
                    unsafe_allow_html=True
                )

            if "exec_pulse_content" not in st.session_state:
                st.session_state["exec_pulse_content"] = None

            if generate_pulse_btn:
                with st.spinner("🤖 Synthesizing macro cultural health and systemic risk drivers..."):
                    top_theme_str = ", ".join(df_feedback["theme"].value_counts().head(3).index.tolist())
                    exec_stats = {
                        "total_feedback": total_fb,
                        "total_teams": n_teams,
                        "n_red": n_red,
                        "n_amber": n_amber,
                        "n_green": n_green,
                        "total_escalations": total_escalations,
                        "top_themes": top_theme_str
                    }
                    pulse_res = llm_inst.generate_executive_pulse(exec_stats)
                    st.session_state["exec_pulse_content"] = pulse_res

            if st.session_state.get("exec_pulse_content"):
                p_data = st.session_state["exec_pulse_content"]
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1.5px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 1.25rem; margin-top: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                            <span style="font-weight: 700; color: #818CF8; font-size: 0.9rem; letter-spacing: 0.05em; text-transform: uppercase;">Executive Intelligence Synthesis</span>
                            <span style="font-size: 0.75rem; background: rgba(99,102,241,0.25); color: #C7D2FE; padding: 2px 8px; border-radius: 4px;">{p_data['provider']}</span>
                        </div>
                        <div style="color: #F1F5F9; font-size: 0.95rem; line-height: 1.6;">
                            {p_data['text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Charts Row 1
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("🎯 Team Risk Distribution (RAG Framework)")
            rag_counts = pd.DataFrame({
                "Status": ["Red (Critical)", "Amber (Elevated)", "Green (Healthy)"],
                "Count": [n_red, n_amber, n_green],
                "Color": ["#EF4444", "#F59E0B", "#10B981"]
            })
            fig_donut = px.pie(
                rag_counts,
                values="Count",
                names="Status",
                color="Status",
                color_discrete_map={
                    "Red (Critical)": "#EF4444",
                    "Amber (Elevated)": "#F59E0B",
                    "Green (Healthy)": "#10B981"
                },
                hole=0.55
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_right:
            st.subheader("📊 Sentiment Breakdown by Department")
            dept_sent = df_feedback.groupby(["department", "sentiment"]).size().reset_index(name="count")
            fig_dept = px.bar(
                dept_sent,
                x="department",
                y="count",
                color="sentiment",
                barmode="stack",
                color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"}
            )
            fig_dept.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Department", gridcolor="#334155"),
                yaxis=dict(title="Submissions", gridcolor="#334155"),
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        # Department RAG Summary Table
        st.subheader("🏢 Department Health & Serious Concern Matrix")
        dept_summary_df = rag_framework.get_department_summary(df_rag_evaluated)
        st.dataframe(
            dept_summary_df,
            hide_index=True,
            use_container_width=True
        )


    # ==============================================================================
    # PAGE 2: RAG RISK MATRIX & TEAMS
    # ==============================================================================
    elif menu == "🎯 RAG Risk Matrix & Teams":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🎯 Red-Amber-Green (RAG) Risk Framework
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Continuous attrition risk scoring, root cause drivers, and tailored HR action playbooks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            status_filter = st.multiselect(
                "Filter by RAG Status",
                ["RED", "AMBER", "GREEN"],
                default=["RED", "AMBER", "GREEN"]
            )
        with f_col2:
            dept_filter = st.multiselect(
                "Filter by Department",
                df_rag_evaluated["department"].unique(),
                default=df_rag_evaluated["department"].unique()
            )

        filtered_rag = df_rag_evaluated[
            (df_rag_evaluated["status"].isin(status_filter)) &
            (df_rag_evaluated["department"].isin(dept_filter))
        ]

        st.markdown(f"**Displaying {len(filtered_rag)} teams matching criteria:**")

        # Team Cards Display
        for _, team_item in filtered_rag.iterrows():
            status = team_item["status"]
            prob_pct = team_item["attrition_probability"] * 100
            
            card_style = (
                "background: rgba(127, 29, 29, 0.35); border: 2px solid #EF4444; box-shadow: 0 0 18px rgba(239, 68, 68, 0.25);"
                if status == "RED" else (
                    "background: rgba(120, 53, 15, 0.25); border: 1.5px solid #F59E0B;"
                    if status == "AMBER" else
                    "background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);"
                )
            )
            badge_class = "badge-red" if status == "RED" else ("badge-amber" if status == "AMBER" else "badge-green")
            status_prefix = "🚨 " if status == "RED" else ("⚠️ " if status == "AMBER" else "✅ ")
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="{card_style} border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <div>
                                <h3 style="margin: 0; color: #F8FAFC; display: inline;">{status_prefix}{team_item['team_id']}</h3>
                                <span style="color: #94A3B8; margin-left: 10px;">({team_item['department']})</span>
                            </div>
                            <span class="{badge_class}" style="font-weight: 800;">{status} — {team_item['status_label'].upper()}</span>
                        </div>
                        <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 20px;">
                            <span style="color: #E2E8F0; font-size: 0.95rem;">
                                Flight Risk Probability: <b style="color: {'#EF4444' if status=='RED' else ('#F59E0B' if status=='AMBER' else '#10B981')}; font-size: 1.1rem;">{prob_pct:.1f}%</b>
                            </span>
                            <span style="color: #94A3B8; font-size: 0.9rem;">Negative Feedback: <b>{team_item['pct_negative']:.1%}</b></span>
                            <span style="color: #94A3B8; font-size: 0.9rem;">Burnout Rate: <b>{team_item['pct_burnout']:.1%}</b></span>
                            <span style="color: #94A3B8; font-size: 0.9rem;">Critical Escalations: <b>{team_item['critical_count']}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                with st.expander(f"🔍 Detailed Drivers & Playbook: {team_item['team_id']}"):
                    col_d, col_a = st.columns([1, 1])
                    with col_d:
                        st.markdown("**📌 Key Risk Drivers:**")
                        for d in team_item["key_drivers"]:
                            st.markdown(f"- {d}")
                        if team_item["mandatory_human_review"]:
                            st.error("⚠️ Mandatory Human Review Triggered: Unresolved critical severity or ethics incident.")
                    with col_a:
                        st.markdown("**📋 Recommended HR Interventions:**")
                        for a in team_item["recommended_actions"]:
                            st.markdown(f"- {a}")

        st.markdown("---")
        # Export Team Risk Report
        st.subheader("📥 Export Team Risk Intelligence Report")
        export_df = filtered_rag[[
            "team_id", "department", "status", "attrition_probability",
            "pct_negative", "pct_burnout", "critical_count", "mandatory_human_review"
        ]].copy()
        
        # Enforce minimum group size safeguard on export
        safe_export_df, is_safe, msg = enforce_k_anonymity_filter(export_df, group_col=None, min_k=MIN_K_ANONYMITY)
        if is_safe:
            csv_data = safe_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download RAG Risk Report (CSV)",
                data=csv_data,
                file_name=f"rag_team_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning(msg)


    # ==============================================================================
    # PAGE 3: ANONYMISED FEEDBACK EXPLORER
    # ==============================================================================
    elif menu == "🔍 Anonymised Feedback Explorer":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🔍 Privacy-Preserving Feedback Explorer
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Search, filter, and inspect feedback with strict k-anonymity (k ≥ 5) cohort masking.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Filter Bar
        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        with c_f1:
            sel_dept = st.selectbox("Department", ["All Departments"] + list(df_feedback["department"].unique()))
        with c_f2:
            sel_theme = st.selectbox("Theme", ["All Themes"] + list(df_feedback["theme"].unique()))
        with c_f3:
            sel_sent = st.selectbox("Sentiment", ["All Sentiments", "Positive", "Neutral", "Negative"])
        with c_f4:
            sel_sev = st.selectbox("Severity", ["All Severities", "Low", "Medium", "High", "Critical"])

        search_kw = st.text_input("Search anonymised text by keyword...", "")

        # Apply Filters
        filtered_df = df_feedback.copy()
        if sel_dept != "All Departments":
            filtered_df = filtered_df[filtered_df["department"] == sel_dept]
        if sel_theme != "All Themes":
            filtered_df = filtered_df[filtered_df["theme"] == sel_theme]
        if sel_sent != "All Sentiments":
            filtered_df = filtered_df[filtered_df["sentiment"] == sel_sent]
        if sel_sev != "All Severities":
            filtered_df = filtered_df[filtered_df["severity"] == sel_sev]
        if search_kw:
            filtered_df = filtered_df[filtered_df["anonymised_text"].str.contains(search_kw, case=False, na=False)]

        # Enforce k-anonymity on the query slice!
        safe_view_df, is_cohort_safe, privacy_msg = enforce_k_anonymity_filter(
            filtered_df, group_col="team_id", min_k=MIN_K_ANONYMITY
        )

        if not is_cohort_safe:
            st.markdown(
                f"""
                <div class="privacy-callout" style="border-left-color: #EF4444;">
                    <h4 style="color: #EF4444; margin: 0;">🛡️ Privacy Protection Threshold Triggered</h4>
                    <p style="color: #E2E8F0; margin: 5px 0 0 0;">{privacy_msg}</p>
                    <small style="color: #94A3B8;">Broaden your search or filter selection to view aggregate cohorts of 5 or more employees.</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info(privacy_msg)
            st.markdown(f"**Showing {len(safe_view_df)} anonymised records:**")

            # Display toggle for inspection
            allow_side_by_side = st.checkbox("Show Redaction Diff View (Compare Original vs Redacted)", value=False)

            for _, row in safe_view_df.head(25).iterrows():
                sev_badge = "badge-red" if row['severity'] in ["Critical", "High"] else "badge-green"
                sent_badge = "badge-green" if row['sentiment'] == "Positive" else ("badge-amber" if row['sentiment'] == "Neutral" else "badge-red")
                
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 0.85rem; color: #94A3B8;">ID: <b>{row['feedback_id']}</b> | Team: <b>{row['team_id']}</b> ({row['department']})</span>
                                <div>
                                    <span class="{sent_badge}">{row['sentiment']}</span>
                                    <span class="{sev_badge}">{row['severity']}</span>
                                    <span class="badge-blue">{row['theme']}</span>
                                </div>
                            </div>
                            <div style="color: #F1F5F9; font-size: 0.95rem; line-height: 1.5;">
                                {row['anonymised_text']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if allow_side_by_side and row['feedback_text'] != row['anonymised_text']:
                        with st.expander("Compare with Original Raw Text"):
                            col_raw, col_anon = st.columns(2)
                            with col_raw:
                                st.caption("Raw Submitted Text (Contains PII):")
                                st.text_area("Original", row['feedback_text'], height=70, key=f"raw_{row['feedback_id']}", disabled=True)
                            with col_anon:
                                st.caption("Cleaned & Redacted Text:")
                                st.text_area("Anonymised", row['anonymised_text'], height=70, key=f"anon_{row['feedback_id']}", disabled=True)


    # ==============================================================================
    # PAGE 4: HR AI TESTING & REDACTION SANDBOX
    # ==============================================================================
    elif menu in ["🧪 AI Testing & Redaction Sandbox", "🧪 AI Playground & Privacy Sandbox"]:
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🧪 HR & Compliance AI Testing & Redaction Sandbox
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    HR & Compliance Testing Lab: Test real-time PII anonymization, multi-target classification, safety overrides, RAG policies, and prompt-injection defenses.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Live Submission Notification & Impact Card (Renders when a review is submitted)
        if st.session_state.get("feedback_just_submitted"):
            st.balloons()
            if st.session_state.get("last_submission_impact"):
                imp = st.session_state["last_submission_impact"]
                st.toast(f"🎉 Success! Feedback {imp['feedback_id']} submitted & {imp['team_id']} sentiment updated!", icon="✅")
            st.session_state["feedback_just_submitted"] = False

        if st.session_state.get("last_submission_impact"):
            impact = st.session_state["last_submission_impact"]
            sent_delta = impact["post_sentiment"] - impact["pre_sentiment"]
            neg_delta = impact["post_neg_pct"] - impact["pre_neg_pct"]
            pos_delta = impact.get("post_pos_pct", 0.0) - impact.get("pre_pos_pct", 0.0)
            risk_delta = impact["post_risk"] - impact["pre_risk"]

            sent_delta_str = f"{sent_delta:+.2f}"
            neg_delta_str = f"{neg_delta:+.1%}"
            pos_delta_str = f"{pos_delta:+.1%}"
            risk_delta_str = f"{risk_delta:+.1%}"

            sent_color = "#10B981" if sent_delta >= 0 else "#EF4444"
            neg_color = "#EF4444" if neg_delta > 0 else "#10B981"
            pos_color = "#10B981" if pos_delta >= 0 else "#EF4444"
            risk_color = "#EF4444" if risk_delta > 0 else "#10B981"

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(15, 23, 42, 0.98) 100%);
                            border: 2px solid #10B981; border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
                            box-shadow: 0 0 30px rgba(16, 185, 129, 0.28);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 0.75rem;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 2.2rem;">🎉</span>
                            <div>
                                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6EE7B7; font-weight: 800;">
                                    ✅ FEEDBACK SUCCESSFULLY SUBMITTED & COMMITTED TO LIVE DATASET
                                </span>
                                <h3 style="color: #FFFFFF; margin: 2px 0 0 0; font-size: 1.4rem; font-weight: 800;">
                                    Team {impact['team_id']} ({impact['department']}) Metrics Updated Live
                                </h3>
                            </div>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="background: rgba(16,185,129,0.25); border: 1px solid #10B981; color: #A7F3D0; font-size: 0.82rem; padding: 4px 12px; border-radius: 20px; font-weight: 700;">
                                ID: {impact['feedback_id']}
                            </span>
                            <span style="background: rgba(99,102,241,0.25); border: 1px solid #6366F1; color: #C7D2FE; font-size: 0.82rem; padding: 4px 12px; border-radius: 20px; font-weight: 600;">
                                {impact['sentiment'].upper()} • {impact['severity'].upper()}
                            </span>
                        </div>
                    </div>
                    <div style="background: rgba(0,0,0,0.4); border-left: 3px solid #10B981; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; font-size: 0.88rem; color: #E2E8F0;">
                        <b>🔒 Scrubbed Record Stored in Active Dataset (Zero Names):</b> "{impact['anonymised_text']}"<br>
                        <span style="color: #94A3B8; font-size: 0.8rem;">Raw Input: <i>"{impact.get('raw_text', '')}"</i> • Theme: <b>{impact['theme']}</b> • Timestamp: <b>{impact.get('timestamp', '')}</b></span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px;">
                        <div style="background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600;">TEAM AVG SENTIMENT</div>
                            <div style="color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">
                                {impact['post_sentiment']:+.2f} 
                                <span style="color: {sent_color}; font-size: 0.82rem; font-weight: 600;">({sent_delta_str})</span>
                            </div>
                            <div style="color: #64748B; font-size: 0.7rem;">Baseline was: {impact['pre_sentiment']:+.2f}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600;">NEGATIVE RATIO</div>
                            <div style="color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">
                                {impact['post_neg_pct']:.1%}
                                <span style="color: {neg_color}; font-size: 0.82rem; font-weight: 600;">({neg_delta_str})</span>
                            </div>
                            <div style="color: #64748B; font-size: 0.7rem;">Baseline was: {impact['pre_neg_pct']:.1%}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600;">FLIGHT ATTRITION RISK</div>
                            <div style="color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">
                                {impact['post_risk']:.1%}
                                <span style="color: {risk_color}; font-size: 0.82rem; font-weight: 600;">({risk_delta_str})</span>
                            </div>
                            <div style="color: #64748B; font-size: 0.7rem;">Baseline was: {impact['pre_risk']:.1%}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600;">TEAM SUBMISSIONS</div>
                            <div style="color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">{impact['post_count']}</div>
                            <div style="color: #10B981; font-size: 0.7rem;">+1 New Review (Was {impact['pre_count']})</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600;">TOTAL COMPANY DATA</div>
                            <div style="color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">{impact.get('post_total', len(df_feedback))}</div>
                            <div style="color: #38BDF8; font-size: 0.7rem;">+1 Total Corpus (Was {impact.get('pre_total', len(df_feedback) - 1)})</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Before vs After Detailed Delta Table
            with st.expander("📊 **View Full Numerical Before vs. After Delta Table**", expanded=True):
                delta_rows = [
                    {
                        "Metric": "🏢 Team Average Sentiment Score",
                        "Before Submission": f"{impact['pre_sentiment']:+.4f}",
                        "After Submission": f"{impact['post_sentiment']:+.4f}",
                        "Shift (Δ)": f"{sent_delta:+.4f}",
                        "Status & Trend": "🔴 Dropped (More Negative)" if sent_delta < -0.005 else ("🟢 Improved (More Positive)" if sent_delta > 0.005 else "⚪ Unchanged")
                    },
                    {
                        "Metric": "📉 Negative Feedback Ratio (%)",
                        "Before Submission": f"{impact['pre_neg_pct']:.2%}",
                        "After Submission": f"{impact['post_neg_pct']:.2%}",
                        "Shift (Δ)": f"{neg_delta:+.2%}",
                        "Status & Trend": "🔴 Negative Feedback Increased" if neg_delta > 0.001 else ("🟢 Negative Feedback Decreased" if neg_delta < -0.001 else "⚪ Unchanged")
                    },
                    {
                        "Metric": "📈 Positive Feedback Ratio (%)",
                        "Before Submission": f"{impact.get('pre_pos_pct', 0.0):.2%}",
                        "After Submission": f"{impact.get('post_pos_pct', 0.0):.2%}",
                        "Shift (Δ)": f"{pos_delta:+.2%}",
                        "Status & Trend": "🟢 Positive Feedback Increased" if pos_delta > 0.001 else ("🔴 Positive Feedback Decreased" if pos_delta < -0.001 else "⚪ Unchanged")
                    },
                    {
                        "Metric": "⚠️ Retention Flight Attrition Risk",
                        "Before Submission": f"{impact['pre_risk']:.2%}",
                        "After Submission": f"{impact['post_risk']:.2%}",
                        "Shift (Δ)": f"{risk_delta:+.2%}",
                        "Status & Trend": "🚨 Team Attrition Risk Escalated" if risk_delta > 0.005 else ("🛡️ Team Attrition Risk Lowered" if risk_delta < -0.005 else "⚪ Stable")
                    },
                    {
                        "Metric": "👥 Target Team Total Submissions",
                        "Before Submission": f"{impact['pre_count']}",
                        "After Submission": f"{impact['post_count']}",
                        "Shift (Δ)": f"+{impact['post_count'] - impact['pre_count']}",
                        "Status & Trend": "✅ Live Review Committed into Team Data"
                    },
                    {
                        "Metric": "🌐 Company-Wide Total Feedbacks",
                        "Before Submission": f"{impact.get('pre_total', 720)}",
                        "After Submission": f"{impact.get('post_total', 721)}",
                        "Shift (Δ)": f"+{impact.get('post_total', 721) - impact.get('pre_total', 720)}",
                        "Status & Trend": "✅ Saved in Active Corpus"
                    }
                ]
                st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

            # Quick Navigation & Reset Bar
            nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns(5)
            with nav_c1:
                if st.button("📊 Executive Overview", use_container_width=True, help="See team sentiment shift on main dashboard"):
                    st.session_state["main_nav_radio"] = "📊 Executive Overview"
                    st.rerun()
            with nav_c2:
                if st.button("🎯 Team Risk Matrix", use_container_width=True, help="See team attrition risk update"):
                    st.session_state["main_nav_radio"] = "🎯 RAG Risk Matrix & Teams"
                    st.rerun()
            with nav_c3:
                if st.button("🔍 Feedback Explorer", use_container_width=True, help="See the scrubbed review in the dataset table"):
                    st.session_state["main_nav_radio"] = "🔍 Anonymised Feedback Explorer"
                    st.rerun()
            with nav_c4:
                if st.button("✖️ Dismiss Confirmation", use_container_width=True):
                    st.session_state["last_submission_impact"] = None
                    st.rerun()
            with nav_c5:
                if st.button("🔄 Reset to Baseline (720)", use_container_width=True):
                    st.session_state["custom_feedbacks"] = []
                    st.session_state["last_submission_impact"] = None
                    st.rerun()

        # Active Session Submissions Audit Trail Log
        if st.session_state.get("custom_feedbacks"):
            with st.expander(f"📋 **Active Session Submissions Audit Log ({len(st.session_state['custom_feedbacks'])} Live Reviews Added)**", expanded=False):
                st.caption("All reviews listed here are active in memory and dynamically driving the sentiment scores and retention risks across all dashboards.")
                df_custom_log = pd.DataFrame(st.session_state["custom_feedbacks"])[
                    ["feedback_id", "team_id", "department", "sentiment", "severity", "theme", "anonymised_text", "timestamp"]
                ]
                st.dataframe(df_custom_log, use_container_width=True, hide_index=True)

        # 1. Target Team Affiliation & AI Settings
        st.markdown("#### **1. Select Submitting Employee Affiliation & Safety Settings:**")
        col_t_dept, col_t_id = st.columns([1, 1])
        with col_t_dept:
            available_depts = sorted(df_feedback["department"].dropna().unique())
            selected_sub_dept = st.selectbox("Submitting Department", available_depts, index=0)
        with col_t_id:
            available_teams = sorted(df_feedback[df_feedback["department"] == selected_sub_dept]["team_id"].dropna().unique())
            selected_sub_team = st.selectbox("Target Team ID", available_teams, index=0)

        # Privacy Cloaking Style & Prediction Pipeline Engine Selectors
        col_cloaking, col_pred_engine = st.columns([1, 1])
        with col_cloaking:
            cloaking_style = st.radio(
                "🛡️ Privacy Anonymization Style",
                [
                    "👤 Natural Colleague Cloaking ('The colleague is happy / bad')",
                    "🏷️ Cryptographic Safeguard Tokens ('[NAME] is happy / bad')"
                ],
                index=0,
                horizontal=False
            )
            is_natural = "Natural Colleague" in cloaking_style

        with col_pred_engine:
            pred_mode = st.radio(
                "Prediction Engine",
                [
                    "🤖 Machine Learning (TF-IDF + Calibrated Classifier)",
                    "🧠 Generative AI LLM (Direct Predictive Reasoner)",
                    "⚡ Hybrid AI + ML Ensemble"
                ],
                index=0,
                horizontal=False
            )
            st.markdown(f"<div style='font-size: 0.8rem; color: #94A3B8; margin-top: 4px;'>Active LLM: <b>{prov_info['name']}</b></div>", unsafe_allow_html=True)

        cur_team_row = df_teams[df_teams["team_id"] == selected_sub_team]
        cur_t = cur_team_row.iloc[0].to_dict() if not cur_team_row.empty else {
            "avg_sentiment_score": 0.0,
            "pct_negative": 0.0,
            "pct_positive": 0.0,
            "continuous_risk_score": 0.0,
            "feedback_count": 0
        }
        team_custom_count = len([fb for fb in st.session_state.get("custom_feedbacks", []) if fb.get("team_id") == selected_sub_team])
        live_badge = f"<span style='color: #10B981; font-weight: 700;'> (+{team_custom_count} Live Added)</span>" if team_custom_count > 0 else ""
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 14px; margin-bottom: 14px; font-size: 0.82rem; color: #94A3B8;">
                🏢 <b>Target Team Live Status ({selected_sub_team}):</b> 
                Avg Sentiment: <span style="color: #38BDF8; font-weight: 700;">{cur_t['avg_sentiment_score']:+.2f}</span> • 
                Negative Feedback: <span style="color: #F87171; font-weight: 700;">{cur_t['pct_negative']:.1%}</span> • 
                Retention Flight Risk: <span style="color: #FBBF24; font-weight: 700;">{cur_t['continuous_risk_score']:.1%}</span> • 
                Submissions in Team: <span style="color: #F8FAFC; font-weight: 700;">{int(cur_t['feedback_count'])}</span>{live_badge}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Core Submission Engine Function
        def execute_submission(raw_text_to_submit: str):
            if not raw_text_to_submit or not raw_text_to_submit.strip():
                st.warning("⚠️ Please enter feedback text before submitting.")
                return

            # 1. Anonymize with selected style (GUARANTEE: No raw names stored)
            anon_res = anonymizer_engine.anonymize(raw_text_to_submit.strip(), natural=is_natural)

            # 2. Build RAG context
            rag_ctx = rag_retriever.build_rag_context(anon_res["anonymised_text"], top_policies=2, top_precedents=2)

            # 3. Classify with selected engine
            ml_res = classifier_engine.classify(raw_text_to_submit.strip(), anon_res["anonymised_text"])
            if "Generative AI LLM" in pred_mode or "Hybrid" in pred_mode:
                llm_res = llm_inst.predict_feedback(
                    anon_res["anonymised_text"],
                    rag_context_text=rag_ctx["rag_context_text"]
                )
            else:
                llm_res = None

            if "Generative AI LLM" in pred_mode and llm_res:
                final_clf = {
                    "theme": llm_res.get("theme", ml_res["theme"]),
                    "theme_confidence": llm_res.get("confidence", ml_res["theme_confidence"]),
                    "sentiment": llm_res.get("sentiment", ml_res["sentiment"]),
                    "sentiment_confidence": llm_res.get("confidence", ml_res["sentiment_confidence"]),
                    "severity": llm_res.get("severity", ml_res["severity"]),
                    "severity_confidence": llm_res.get("confidence", ml_res["severity_confidence"]),
                    "escalation_status": llm_res.get("escalation_status", ml_res["escalation_status"]),
                    "escalation_reason": llm_res.get("escalation_reason", ml_res["escalation_reason"]),
                    "rule_override_applied": ml_res["rule_override_applied"],
                    "prompt_injection_flag": ml_res["prompt_injection_flag"],
                    "prompt_injection_details": ml_res["prompt_injection_details"],
                    "valence_score": ml_res.get("valence_score", 0.0),
                    "sentiment_keywords": llm_res.get("key_drivers", ml_res.get("sentiment_keywords", []))
                }
            else:
                final_clf = ml_res

            # Pre-submission metrics
            cur_slice = df_teams[df_teams["team_id"] == selected_sub_team]
            pre_t = cur_slice.iloc[0].to_dict() if not cur_slice.empty else {}
            pre_tot = len(df_feedback)

            # New record - GUARANTEED zero names stored!
            new_rec = {
                "feedback_id": f"FB-CUSTOM-{len(df_feedback) + 1}",
                "department": selected_sub_dept,
                "team_id": selected_sub_team,
                "theme": final_clf["theme"],
                "sentiment": final_clf["sentiment"],
                "severity": final_clf["severity"],
                "privacy_risk": anon_res["privacy_risk"],
                "escalation_status": final_clf["escalation_status"],
                "feedback_text": anon_res["anonymised_text"],  # Scrubbed text only!
                "anonymised_text": anon_res["anonymised_text"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["custom_feedbacks"].insert(0, new_rec)

            # Recompute post-submission metrics
            df_new_fbs = pd.concat([pd.DataFrame([new_rec]), df_feedback], ignore_index=True)
            df_new_tms = compute_live_team_features(df_new_fbs, df_teams_base)
            post_slice = df_new_tms[df_new_tms["team_id"] == selected_sub_team]
            post_t = post_slice.iloc[0].to_dict() if not post_slice.empty else {}
            post_tot = len(df_new_fbs)

            st.session_state["last_submission_impact"] = {
                "feedback_id": new_rec["feedback_id"],
                "team_id": selected_sub_team,
                "department": selected_sub_dept,
                "sentiment": final_clf["sentiment"],
                "theme": final_clf["theme"],
                "severity": final_clf["severity"],
                "raw_text": raw_text_to_submit.strip(),
                "anonymised_text": anon_res["anonymised_text"],
                "timestamp": new_rec["timestamp"],
                "pre_sentiment": pre_t.get("avg_sentiment_score", 0.0),
                "post_sentiment": post_t.get("avg_sentiment_score", 0.0),
                "pre_neg_pct": pre_t.get("pct_negative", 0.0),
                "post_neg_pct": post_t.get("pct_negative", 0.0),
                "pre_pos_pct": pre_t.get("pct_positive", 0.0),
                "post_pos_pct": post_t.get("pct_positive", 0.0),
                "pre_risk": pre_t.get("continuous_risk_score", 0.0),
                "post_risk": post_t.get("continuous_risk_score", 0.0),
                "pre_count": int(pre_t.get("feedback_count", 0)),
                "post_count": int(post_t.get("feedback_count", 0)),
                "pre_total": pre_tot,
                "post_total": post_tot,
                "valence_score": final_clf.get("valence_score", 0.0)
            }
            st.session_state["feedback_just_submitted"] = True
            st.session_state["playground_active_text"] = anon_res["anonymised_text"]
            st.rerun()

        # Session state initialization for playground
        if "playground_active_text" not in st.session_state:
            st.session_state["playground_active_text"] = "I am very happy with my team, manager, and the great collaborative culture here. Everything is going well!"
        if "custom_feedback_text_field" not in st.session_state:
            st.session_state["custom_feedback_text_field"] = ""

        def set_custom_text(text: str):
            st.session_state["custom_feedback_text_field"] = text
            st.session_state["playground_active_text"] = text

        def clear_custom_text():
            st.session_state["custom_feedback_text_field"] = ""
            st.session_state["playground_active_text"] = ""

        st.markdown("#### **2. Choose Input Mode & Write Feedback:**")
        input_mode = st.radio(
            "Sandbox Input Mode",
            ["✍️ Custom Feedback (Type Any Text)", "⚡ One-Click Demo Presets"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "✍️ Custom Feedback (Type Any Text)":
            st.markdown("##### **Type or paste any custom employee feedback:**")
            custom_input_val = st.text_area(
                "Custom Feedback Input (type freely, e.g. 'Pay is too low', 'Manager made sexual remarks', 'Harassment by director'):",
                placeholder="Type or paste any custom feedback here... e.g. 'Ram is bad' or 'My manager subjected me to sexual harassment and unwanted touching'...",
                height=110,
                key="custom_feedback_text_field"
            )

            col_act1, col_act2, col_act3 = st.columns([2.5, 1.8, 1])
            with col_act1:
                if st.button(
                    f"📤 Anonymize & Submit to {selected_sub_team} Live", 
                    type="primary", 
                    use_container_width=True,
                    key="btn_custom_submit",
                    help=f"Submits this feedback into {selected_sub_team}'s live dataset, updates sentiment scores, and recalculates retention flight risk."
                ):
                    execute_submission(custom_input_val)
            with col_act2:
                if st.button("🔍 Preview Analysis Only", use_container_width=True, key="btn_custom_preview"):
                    if custom_input_val and custom_input_val.strip():
                        st.session_state["playground_active_text"] = custom_input_val.strip()
            with col_act3:
                st.button("🧹 Clear Box", on_click=clear_custom_text, use_container_width=True, key="btn_custom_clear")

            st.markdown("###### *Quick Idea Starters (Click any card to load custom phrase):*")
            chip_c1, chip_c2, chip_c3, chip_c4, chip_c5 = st.columns(5)
            with chip_c1:
                st.button(
                    "🚨 'Sexual Harassment'",
                    on_click=set_custom_text,
                    args=("My manager made inappropriate sexual remarks and subjected me to persistent sexual harassment.",),
                    use_container_width=True
                )
            with chip_c2:
                st.button(
                    "💸 'Pay is low & no bonuses'",
                    on_click=set_custom_text,
                    args=("Salary is too low for the cost of living and we received no bonuses this year.",),
                    use_container_width=True
                )
            with chip_c3:
                st.button(
                    "💔 'Manager does not care'",
                    on_click=set_custom_text,
                    args=("Our manager does not care about team burnout and micromanages all daily tasks.",),
                    use_container_width=True
                )
            with chip_c4:
                st.button(
                    "🌟 'Great support & culture'",
                    on_click=set_custom_text,
                    args=("Great support from my team lead and manager. I really enjoy the collaborative culture!",),
                    use_container_width=True
                )
            with chip_c5:
                st.button(
                    "⚙️ 'Outdated slow tools'",
                    on_click=set_custom_text,
                    args=("The internal development tools are completely outdated, slow, and frustrating to use.",),
                    use_container_width=True
                )

            # Update active text from custom field if user typed
            if custom_input_val and custom_input_val.strip():
                st.session_state["playground_active_text"] = custom_input_val.strip()
            elif not st.session_state.get("playground_active_text"):
                st.session_state["playground_active_text"] = ""

        else:
            st.markdown("##### **⚡ Quick Test Presets (Click any preset to test immediately):**")
            def set_preset_text(preset_text: str):
                st.session_state["playground_active_text"] = preset_text

            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                st.button(
                    "🟢 1. Happy & Praising",
                    on_click=set_preset_text,
                    args=("I am very happy with my team, manager, and our great collaborative culture. Everything is going great!",),
                    use_container_width=True
                )
            with p2:
                st.button(
                    "⚪ 2. Routine Neutral",
                    on_click=set_preset_text,
                    args=("Standard weekly sprint retro completed. Workload is manageable and routine tasks are on schedule.",),
                    use_container_width=True
                )
            with p3:
                st.button(
                    "🟡 3. Direct PII Exposure",
                    on_click=set_preset_text,
                    args=("Hi, I am Alex Johnson (EMP-4821). Routine weekly sync notes: Project Titan sprint goals were reviewed during Tuesday's standup. Call my desk phone at 415-555-0199 or email me at alex.johnson@company.internal for the schedule.",),
                    use_container_width=True
                )
            with p4:
                st.button(
                    "🟠 4. Severe Burnout",
                    on_click=set_preset_text,
                    args=("We are forced to work 75+ hour weeks including all weekends. Two teammates were hospitalized for acute stress. Our manager publicly berates anyone who speaks up. The team is preparing a mass resignation walkout.",),
                    use_container_width=True
                )
            with p5:
                st.button(
                    "🔴 5. Harassment Crisis",
                    on_click=set_preset_text,
                    args=("A senior director subjected me to sexual harassment and unwanted physical contact during the offsite dinner. When I objected, they threatened me with demotion and retaliation.",),
                    use_container_width=True
                )

            col_p_sub, col_p_prev = st.columns([2.5, 1.8])
            with col_p_sub:
                if st.button(
                    f"📤 Anonymize & Submit Selected Preset to {selected_sub_team} Live",
                    type="primary",
                    use_container_width=True,
                    key="btn_preset_submit",
                    help=f"Submits this preset review to {selected_sub_team} and updates sentiment scores."
                ):
                    execute_submission(st.session_state.get("playground_active_text", ""))
            with col_p_prev:
                st.button("🔍 Preview Selected Preset Below", use_container_width=True, key="btn_preset_prev")

        input_text = st.session_state.get("playground_active_text", "").strip()

        if not input_text:
            st.info("💡 **Ready for Input:** Type any custom employee feedback above or select a preset scenario to see real-time classification, PII scrubbing, and sentiment analysis.")
            st.stop()

        st.markdown("---")
        st.markdown("### **🎯 AI Analysis & Redaction Results:**")

        # Live Real-Time Anonymization
        anon_result = anonymizer_engine.anonymize(input_text, natural=is_natural)
        
        # RAG Retrieval: Governing HR Policies & Historical Case Precedents
        rag_context = rag_retriever.build_rag_context(anon_result["anonymised_text"], top_policies=2, top_precedents=2)

        # ML Classification
        ml_clf_result = classifier_engine.classify(input_text, anon_result["anonymised_text"])

        # LLM Classification if selected
        if "Generative AI LLM" in pred_mode or "Hybrid" in pred_mode:
            llm_pred_result = llm_inst.predict_feedback(
                anon_result["anonymised_text"],
                rag_context_text=rag_context["rag_context_text"]
            )
        else:
            llm_pred_result = None

        if "Generative AI LLM" in pred_mode and llm_pred_result:
            clf_result = {
                "theme": llm_pred_result.get("theme", ml_clf_result["theme"]),
                "theme_confidence": llm_pred_result.get("confidence", ml_clf_result["theme_confidence"]),
                "sentiment": llm_pred_result.get("sentiment", ml_clf_result["sentiment"]),
                "sentiment_confidence": llm_pred_result.get("confidence", ml_clf_result["sentiment_confidence"]),
                "severity": llm_pred_result.get("severity", ml_clf_result["severity"]),
                "severity_confidence": llm_pred_result.get("confidence", ml_clf_result["severity_confidence"]),
                "escalation_status": llm_pred_result.get("escalation_status", ml_clf_result["escalation_status"]),
                "escalation_reason": llm_pred_result.get("escalation_reason", ml_clf_result["escalation_reason"]),
                "rule_override_applied": ml_clf_result["rule_override_applied"],
                "prompt_injection_flag": ml_clf_result["prompt_injection_flag"],
                "prompt_injection_details": ml_clf_result["prompt_injection_details"],
                "valence_score": ml_clf_result.get("valence_score", 0.0),
                "sentiment_keywords": llm_pred_result.get("key_drivers", ml_clf_result.get("sentiment_keywords", []))
            }
        else:
            clf_result = ml_clf_result

        # Extract Sentiment Visuals
        sent_label = clf_result["sentiment"]
        sent_conf = clf_result["sentiment_confidence"]
        valence_score = clf_result.get("valence_score", 0.0)
        sentiment_kws = clf_result.get("sentiment_keywords", [])

        if sent_label == "Positive":
            sent_bg = "linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(6, 95, 70, 0.4) 100%)"
            sent_border = "#10B981"
            sent_icon = "🟢"
            sent_title_color = "#34D399"
        elif sent_label == "Negative":
            sent_bg = "linear-gradient(135deg, rgba(127, 29, 29, 0.7) 0%, rgba(153, 27, 27, 0.4) 100%)"
            sent_border = "#EF4444"
            sent_icon = "🔴"
            sent_title_color = "#F87171"
        else:
            sent_bg = "linear-gradient(135deg, rgba(120, 53, 15, 0.7) 0%, rgba(146, 64, 14, 0.4) 100%)"
            sent_border = "#F59E0B"
            sent_icon = "⚪"
            sent_title_color = "#FBBF24"

        kw_summary = f"<b>Detected Drivers:</b> {', '.join(sentiment_kws)}" if sentiment_kws else "Balanced neutral tone with no strong emotional polarity"

        # Big Live Sentiment Banner
        st.markdown(
            f"""
            <div style="background: {sent_bg}; border: 2px solid {sent_border}; border-radius: 12px; padding: 1.25rem 1.5rem; margin-top: 0.5rem; margin-bottom: 1.25rem; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #CBD5E1; font-weight: 600;">Live Sentiment Analysis</span>
                        <h2 style="color: {sent_title_color}; margin: 2px 0 0 0; font-size: 1.8rem; font-weight: 800;">
                            {sent_icon} {sent_label.upper()} ({sent_conf:.1%} Confidence)
                        </h2>
                        <p style="color: #E2E8F0; margin: 6px 0 0 0; font-size: 0.95rem;">
                            {kw_summary}
                        </p>
                    </div>
                    <div style="text-align: right; background: rgba(0, 0, 0, 0.3); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                        <div style="color: #94A3B8; font-size: 0.75rem; font-weight: 600;">COMMENT VALENCE</div>
                        <div style="color: #FFFFFF; font-size: 1.5rem; font-weight: 700;">{valence_score:+.2f}</div>
                        <div style="color: #A5B4FC; font-size: 0.68rem;">(This individual text score)</div>
                    </div>
                </div>
            </div>
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 6px 14px; margin-top: -0.5rem; margin-bottom: 1.25rem; font-size: 0.8rem; color: #CBD5E1;">
                💡 <b>Valence Score Scope:</b> The <b>Valence Score ({valence_score:+.2f})</b> evaluates <i>this single feedback comment</i>. Submitting it dynamically updates <b>{selected_sub_team}'s Team Average Sentiment ({cur_t['avg_sentiment_score']:+.2f})</b> and recomputes continuous attrition flight risk across all dashboards.
            </div>
            """,
            unsafe_allow_html=True
        )

        # High-Impact Emergency Alert if Critical Severity
        if clf_result["severity"] == "Critical" or clf_result["rule_override_applied"]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(127, 29, 29, 0.95) 0%, rgba(69, 10, 10, 0.98) 100%); 
                            border: 2.5px solid #EF4444; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; 
                            box-shadow: 0 0 30px rgba(239, 68, 68, 0.45);">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <span style="font-size: 2.5rem;">🚨</span>
                        <div>
                            <h3 style="color: #FEE2E2; margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: 0.02em;">
                                HIGH-SEVERITY CRISIS CONFIRMED: LEVEL 4 CRITICAL COMPLIANCE ALERT
                            </h3>
                            <p style="color: #FCA5A5; margin: 6px 0 0 0; font-size: 1rem; line-height: 1.4;">
                                <b>Trigger:</b> {clf_result['escalation_reason']}<br>
                                <b>Mandatory Protocol:</b> Instant Sequester to Human Review Queue • Legal/Compliance Alert Triggered • Whistleblower Protections Activated.
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif clf_result["severity"] == "High":
            st.markdown(
                f"""
                <div style="background: rgba(120, 53, 15, 0.5); border: 2px solid #F59E0B; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.25rem;">
                    <h4 style="color: #FCD34D; margin: 0;">⚠️ ELEVATED SEVERITY: Action Required to Prevent Team Attrition</h4>
                    <p style="color: #FDE68A; margin: 4px 0 0 0;">{clf_result['escalation_reason']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("📋 Classification & Privacy Assessment")

        # Result Badges
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Predicted Theme", clf_result["theme"], f"{clf_result['theme_confidence']:.1%} Conf")
        with b2:
            st.metric("Sentiment", f"{sent_icon} {clf_result['sentiment']}", f"{clf_result['sentiment_confidence']:.1%} Conf")
        with b3:
            sev_color = "🔴 " if clf_result['severity'] == "Critical" else ("🟠 " if clf_result['severity'] == "High" else "🟢 ")
            st.metric("Severity Level", f"{sev_color}{clf_result['severity']}", f"{clf_result['severity_confidence']:.1%} Conf")
        with b4:
            priv_color = "🔴 " if anon_result['privacy_risk'] == "High" else ("🟠 " if anon_result['privacy_risk'] == "Medium" else "🟢 ")
            st.metric("Privacy Risk", f"{priv_color}{anon_result['privacy_risk']}", f"{len(anon_result['entities_found'])} PII scrubbed")

        # Live Projected Team Attrition Impact Preview
        temp_proj_rec = {
            "feedback_id": "PREVIEW",
            "department": selected_sub_dept,
            "team_id": selected_sub_team,
            "theme": clf_result["theme"],
            "sentiment": clf_result["sentiment"],
            "severity": clf_result["severity"],
            "privacy_risk": anon_result["privacy_risk"],
            "escalation_status": clf_result["escalation_status"],
            "feedback_text": anon_result["anonymised_text"],
            "anonymised_text": anon_result["anonymised_text"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df_proj_fbs = pd.concat([pd.DataFrame([temp_proj_rec]), df_feedback], ignore_index=True)
        df_proj_tms = compute_live_team_features(df_proj_fbs, df_teams_base)
        proj_slice = df_proj_tms[df_proj_tms["team_id"] == selected_sub_team]
        proj_t = proj_slice.iloc[0].to_dict() if not proj_slice.empty else {}

        proj_risk = proj_t.get("continuous_risk_score", cur_t["continuous_risk_score"])
        proj_risk_delta = proj_risk - cur_t["continuous_risk_score"]
        proj_sent = proj_t.get("avg_sentiment_score", cur_t["avg_sentiment_score"])
        proj_sent_delta = proj_sent - cur_t["avg_sentiment_score"]

        p_risk_color = "#EF4444" if proj_risk_delta > 0.001 else ("#10B981" if proj_risk_delta < -0.001 else "#94A3B8")
        p_risk_badge = f"{proj_risk_delta:+.2%}" if abs(proj_risk_delta) >= 0.0001 else "±0.0%"
        p_trend_msg = "🚨 Projected to Increase Flight Risk" if proj_risk_delta > 0.001 else ("🛡️ Projected to Reduce Flight Risk" if proj_risk_delta < -0.001 else "⚪ Projected Impact: Neutral")

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1.5px solid rgba(255, 255, 255, 0.12); border-radius: 10px; padding: 10px 16px; margin-top: 10px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 700;">
                        📈 PREDICTED TARGET TEAM ATTRITION IMPACT ({selected_sub_team})
                    </span>
                    <div style="color: #F8FAFC; font-size: 0.95rem; font-weight: 600; margin-top: 2px;">
                        Current Risk: <b>{cur_t['continuous_risk_score']:.1%}</b> ➔ Projected Risk: <b>{proj_risk:.1%}</b> 
                        <span style="color: {p_risk_color}; font-weight: 800; margin-left: 6px;">({p_risk_badge})</span>
                    </div>
                </div>
                <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                    <span style="color: {p_risk_color}; font-size: 0.85rem; font-weight: 700;">{p_trend_msg}</span>
                    <span style="background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #CBD5E1;">
                        Team Sentiment: {proj_sent:+.2f} ({proj_sent_delta:+.2f})
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Redaction Comparison
        st.markdown("#### **🛡️ PII Redaction Live Comparison**")
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.markdown("**Original Raw Text:**")
            st.code(input_text, language="text")
        with diff_col2:
            badge_lbl = "👤 Natural Colleague Cloaking" if is_natural else "🏷️ Cryptographic Safeguard Tokens"
            st.markdown(f"**Anonymised Text ({badge_lbl}):**")
            st.code(anon_result["anonymised_text"], language="text")
            if is_natural:
                st.caption(f"🔒 **Strict Cryptographic Token Format:** `{anon_result['token_redacted_text']}`")
            else:
                st.caption(f"👤 **Natural Colleague Cloaking Format:** `{anon_result['natural_text']}`")

        # Entity Inventory
        if anon_result["entities_found"]:
            st.markdown("#### **🔍 Detected & Redacted PII Inventory**")
            df_entities = pd.DataFrame(anon_result["entities_found"])
            st.dataframe(df_entities, hide_index=True, use_container_width=True)
        else:
            st.success("✅ No direct personal identifiers detected in submitted text.")

        # Residual Quasi-Identifiers
        if anon_result["residual_hazards"]:
            st.markdown(
                f"""
                <div class="privacy-callout" style="border-left-color: #F59E0B;">
                    <h4 style="color: #F59E0B; margin: 0;">⚠️ Residual Re-Identification Hazards Detected</h4>
                    <ul style="color: #E2E8F0; margin: 8px 0 0 0;">
                        {''.join(f'<li>{h}</li>' for h in anon_result["residual_hazards"])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Prompt Injection Flag
        if clf_result["prompt_injection_flag"]:
            st.warning(
                f"🛡️ **Prompt Injection Safeguard Triggered**: Input attempted instruction override. "
                f"Details: {', '.join(clf_result['prompt_injection_details'])}"
            )

        # Dedicated Live Feedback Submission Card
        st.markdown("---")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%); 
                        border: 1.5px solid #6366F1; border-radius: 12px; padding: 1.25rem; margin-top: 1rem; margin-bottom: 1.25rem;
                        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <h3 style="color: #FFFFFF; margin: 0; font-size: 1.25rem; font-weight: 800;">
                            📤 Submit Anonymized Feedback to Team Pulse
                        </h3>
                        <p style="color: #CBD5E1; margin: 4px 0 0 0; font-size: 0.88rem;">
                            Submitting for <b>{selected_sub_team}</b> ({selected_sub_dept}). The scrubbed, anonymized text will be committed to the active dataset, dynamically updating the team's sentiment scores, negative ratio, and retention flight risk across all dashboards.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_sub_btn, col_sub_info = st.columns([1.5, 2])
        with col_sub_btn:
            submit_to_pulse = st.button(
                f"🚀 Anonymize & Submit to {selected_sub_team} Pulse",
                type="primary",
                use_container_width=True,
                key="btn_submit_to_pulse"
            )
        with col_sub_info:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10B981; border-radius: 8px; padding: 8px 12px; font-size: 0.78rem; color: #34D399;">
                    🔒 <b>Zero-PII Storage Guarantee:</b> Raw names are replaced with: <i>'{anon_result['anonymised_text']}'</i> before saving.
                </div>
                """,
                unsafe_allow_html=True
            )

        if submit_to_pulse:
            execute_submission(input_text)

        # RAG Knowledge Base Retrieval Card
        with st.expander("📚 **RAG Knowledge Base & Historical Precedent Retrieval**", expanded=True):
            st.caption("Retrieval-Augmented Generation (RAG) matches feedback against governing policies and historical precedent resolutions.")
            rag_p_col, rag_c_col = st.columns(2)
            with rag_p_col:
                st.markdown("##### **Governing Corporate HR Policies Matched:**")
                if rag_context["policies"]:
                    for p in rag_context["policies"]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #6366F1; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <b style="color: #A5B4FC; font-size: 0.85rem;">{p['policy_id']} — {p['title']}</b>
                                    <span style="background: rgba(99,102,241,0.25); color: #C7D2FE; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px;">{p['similarity_score']:.1%} Match</span>
                                </div>
                                <div style="color: #94A3B8; font-size: 0.75rem; margin: 2px 0;">{p['section']}</div>
                                <div style="color: #E2E8F0; font-size: 0.8rem; line-height: 1.3;">"{p['text'][:160]}..."</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No direct corporate policy triggers matched.")

            with rag_c_col:
                st.markdown("##### **Historical Feedback Precedents & Past Actions:**")
                if rag_context["precedents"]:
                    for prec in rag_context["precedents"]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #10B981; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <b style="color: #6EE7B7; font-size: 0.85rem;">Precedent {prec['feedback_id']} ({prec['theme']})</b>
                                    <span style="background: rgba(16,185,129,0.25); color: #A7F3D0; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px;">{prec['similarity_score']:.1%} Similarity</span>
                                </div>
                                <div style="color: #CBD5E1; font-size: 0.8rem; margin: 4px 0;">"{prec['anonymised_text'][:120]}..."</div>
                                <div style="color: #34D399; font-size: 0.75rem;"><b>Past Resolution:</b> {prec['resolution']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No close historical precedents retrieved.")

        # Generative AI Qualitative Deep-Dive & Action Plan
        st.markdown("---")
        st.markdown("#### **🧠 Generative AI Deep-Dive & Leadership Action Playbook**")
        st.caption("AI psychological root-cause analysis, retention implications, and manager 1-on-1 talking points.")

        col_llm_run, col_llm_prov = st.columns([1, 2])
        with col_llm_run:
            run_llm_deep_dive = st.button("✨ Generate AI Qualitative Analysis", type="primary", use_container_width=True)
        with col_llm_prov:
            st.markdown(
                f"<div style='margin-top: 6px; font-size: 0.85rem; color: #94A3B8;'>Backend: <b>{prov_info['name']}</b> • 🔒 Strictly uses scrubbed anonymised text</div>",
                unsafe_allow_html=True
            )

        if "llm_deep_dive_cache" not in st.session_state:
            st.session_state["llm_deep_dive_cache"] = {}

        cache_key = anon_result["anonymised_text"]

        if run_llm_deep_dive:
            with st.spinner("🤖 Consulting AI Organizational Psychologist with RAG grounding..."):
                deep_res = llm_inst.generate_rag_grounded_deep_dive(
                    feedback_text=anon_result["anonymised_text"],
                    theme=clf_result["theme"],
                    sentiment=clf_result["sentiment"],
                    severity=clf_result["severity"],
                    rag_context=rag_context
                )
                st.session_state["llm_deep_dive_cache"][cache_key] = deep_res

        if cache_key in st.session_state["llm_deep_dive_cache"]:
            res_deep = st.session_state["llm_deep_dive_cache"][cache_key]
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.85); border: 1.5px solid rgba(139, 92, 246, 0.45); border-radius: 14px; padding: 1.5rem; margin-top: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem;">
                        <span style="font-weight: 700; color: #A78BFA; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">🧠 AI Organizational Psychologist Assessment</span>
                        <span style="font-size: 0.8rem; background: rgba(139,92,246,0.2); color: #DDD6FE; padding: 3px 8px; border-radius: 6px;">Provider: {res_deep['provider']}</span>
                    </div>
                    <div style="color: #F1F5F9; font-size: 0.95rem; line-height: 1.6;">
                        {res_deep['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ==============================================================================
    # PAGE 5: TEAM ATTRITION PREDICTION LAB
    # ==============================================================================
    elif menu == "📈 Team Attrition Prediction Lab":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    📈 Team-Level Attrition Risk Predictor
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Machine learning risk attribution strictly aggregated at the team level (Zero individual prediction).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="privacy-callout">
                <b>🛡️ Ethical ML Guarantee:</b> This model strictly evaluates aggregated team dynamics (e.g., negative sentiment ratio, burnout density, leadership friction). It is mathematically prohibited from ingesting individual employee identities or predicting individual resignation.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Feature Importance Chart
        st.subheader("📊 Key Predictive Drivers (Feature Importances)")
        df_importances = attrition_engine.get_feature_importances()
        
        fig_imp = px.bar(
            df_importances.head(10),
            x="rf_importance",
            y="feature",
            orientation="h",
            color="rf_importance",
            color_continuous_scale="Purples",
            title="Top 10 Team Features Driving Attrition Risk (Random Forest)"
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            yaxis=dict(autorange="reversed", gridcolor="#334155"),
            xaxis=dict(title="Importance Weight", gridcolor="#334155"),
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        # Interactive Team Risk Simulator
        st.subheader("🎛️ Interactive Team Dynamics Risk Simulator")
        st.caption("Simulate how changes in team sentiment, burnout, or leadership issues impact predicted attrition risk.")

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_neg = st.slider("Negative Feedback Ratio (%)", 0, 100, 35) / 100.0
            sim_burnout = st.slider("Burnout Complaint Ratio (%)", 0, 100, 25) / 100.0
        with sim_c2:
            sim_mgmt = st.slider("Management Issue Ratio (%)", 0, 100, 20) / 100.0
            sim_comp = st.slider("Compensation Issue Ratio (%)", 0, 100, 15) / 100.0
        with sim_c3:
            sim_crit = st.number_input("Critical Incident Count", min_value=0, max_value=10, value=0)
            sim_part = st.slider("Survey Participation Rate (%)", 10, 100, 75) / 100.0

        # Build simulated feature dictionary matching the 21 features
        sim_features = {
            "team_size": 20,
            "feedback_count": 18,
            "participation_rate": sim_part,
            "silent_disengagement_index": round(max(0.0, 1.0 - sim_part), 4),
            "avg_sentiment_score": round((1.0 - sim_neg) - sim_neg, 4),
            "sentiment_volatility": round(min(1.0, sim_neg * 0.9 + 0.1), 4),
            "pct_negative": sim_neg,
            "pct_positive": round(max(0.0, 1.0 - sim_neg - 0.2), 4),
            "pct_neutral": 0.2,
            "avg_severity_score": round(1.5 + (sim_neg * 2.0), 4),
            "critical_count": sim_crit,
            "high_severity_count": int(sim_neg * 5),
            "pct_burnout": sim_burnout,
            "burnout_velocity": round(sim_burnout * (1.5 + (sim_neg * 2.0)) / 2.0, 4),
            "pct_management_issues": sim_mgmt,
            "manager_support_index": round(0.40 - (sim_mgmt * 1.5), 4),
            "pct_comp_dissatisfaction": sim_comp,
            "comp_friction_score": round(sim_comp * (1.0 + sim_neg), 4),
            "pct_culture_issues": 0.10,
            "escalated_count": sim_crit,
            "pct_escalated": round(sim_crit / 18.0, 4)
        }

        pred_res = attrition_engine.predict_team(sim_features, model_type=model_code)
        prob_score = pred_res["high_risk_probability"]

        # Gauge visual for probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_score * 100,
            title={'text': f"Simulated High Attrition Risk ({selected_model})"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6366F1"},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.4)"},
                    {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.4)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.4)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': red_thresh * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            height=280,
            margin=dict(t=30, b=10, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # AI LLM Team Attrition Reasoning & RAG Comparison Card
        st.markdown("---")
        st.markdown("#### **🤖 AI LLM Attrition Risk Reasoning & RAG Context Comparison**")
        st.caption("Evaluates quantitative team metrics alongside qualitative feedback nuance using the Generative AI LLM.")

        col_llm_team_btn, col_llm_team_info = st.columns([1, 2])
        with col_llm_team_btn:
            run_llm_team_eval = st.button("✨ Run AI LLM Attrition Prediction", type="primary", use_container_width=True)
        with col_llm_team_info:
            st.markdown(f"<div style='margin-top: 6px; font-size: 0.85rem; color: #94A3B8;'>LLM Engine: <b>{prov_info['name']}</b> • Evaluates simulated team dynamics</div>", unsafe_allow_html=True)

        if "sim_llm_pred" not in st.session_state:
            st.session_state["sim_llm_pred"] = None

        if run_llm_team_eval:
            with st.spinner("🤖 LLM analyzing team risk vectors and capacity strain..."):
                llm_team_res = llm_inst.predict_team_attrition(
                    team_id="SIM-Cohort-Alpha",
                    department="Simulated Unit",
                    team_features=sim_features,
                    retrieved_feedback_snippets=[
                        f"Team burnout ratio is at {sim_burnout:.1%}",
                        f"Management friction reported by {sim_mgmt:.1%} of members",
                        f"{sim_crit} critical incidents on record"
                    ]
                )
                st.session_state["sim_llm_pred"] = llm_team_res

        if st.session_state.get("sim_llm_pred"):
            llm_t = st.session_state["sim_llm_pred"]
            ml_prob = prob_score
            llm_prob = llm_t["llm_attrition_probability"]
            hybrid_prob = 0.5 * ml_prob + 0.5 * llm_prob

            # Consensus badge
            both_agree = (ml_prob >= 0.5 and llm_prob >= 0.5) or (ml_prob < 0.5 and llm_prob < 0.5)
            consensus_text = "🟢 Strong AI & ML Consensus" if both_agree else "🟡 Model Disagreement / Edge Case"
            consensus_color = "#10B981" if both_agree else "#F59E0B"

            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.85); border: 1.5px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                        <span style="font-weight: 700; color: #818CF8; font-size: 0.95rem; text-transform: uppercase;">AI & ML Dual-Engine Attrition Comparison</span>
                        <span style="font-size: 0.8rem; background: rgba(99,102,241,0.2); color: {consensus_color}; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{consensus_text}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 1rem;">
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="color: #94A3B8; font-size: 0.75rem; font-weight: 600;">🤖 MACHINE LEARNING (ML)</div>
                            <div style="color: #FFFFFF; font-size: 1.4rem; font-weight: 700;">{ml_prob:.1%}</div>
                            <div style="color: #CBD5E1; font-size: 0.75rem;">Model: {selected_model}</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(99,102,241,0.3);">
                            <div style="color: #A5B4FC; font-size: 0.75rem; font-weight: 600;">🧠 GENERATIVE AI (LLM)</div>
                            <div style="color: #818CF8; font-size: 1.4rem; font-weight: 700;">{llm_prob:.1%}</div>
                            <div style="color: #CBD5E1; font-size: 0.75rem;">Risk Status: {llm_t['llm_risk_status']}</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(16,185,129,0.3);">
                            <div style="color: #6EE7B7; font-size: 0.75rem; font-weight: 600;">⚡ HYBRID CONSENSUS</div>
                            <div style="color: #10B981; font-size: 1.4rem; font-weight: 700;">{hybrid_prob:.1%}</div>
                            <div style="color: #CBD5E1; font-size: 0.75rem;">Ensemble Blended Weight</div>
                        </div>
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0.5rem;">
                        <b>AI Primary Risk Vectors:</b> {', '.join(llm_t['primary_risk_vectors'])}<br>
                        <b>AI Diagnostic:</b> {llm_t['ai_diagnostic_summary']}
                    </div>
                    <div style="background: rgba(99, 102, 241, 0.15); border-left: 3px solid #6366F1; padding: 10px; border-radius: 6px; color: #C7D2FE; font-size: 0.85rem;">
                        <b>Prescribed Strategic Intervention:</b> {llm_t['prescribed_intervention']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ==============================================================================
    # PAGE 6: MODEL EVALUATION & BENCHMARKS
    # ==============================================================================
    elif menu == "📉 Model Evaluation & Benchmarks":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    📉 Model Evaluation & Rigorous Benchmarks
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Empirical evaluation on held-out independent test teams: Baseline vs Logistic Regression vs Random Forest.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        eval_results = evaluate_attrition_models()

        # Summary Comparison Table
        st.subheader("📊 Held-Out Test Set Performance Benchmark")
        st.caption(
            f"Evaluated on {eval_results['test_size']} independent held-out teams "
            f"({eval_results['test_high_risk_count']} High Risk, {eval_results['test_low_risk_count']} Low Risk). "
            "Strictly split by independent team groups to eliminate train/test data leakage."
        )
        st.table(eval_results["summary_table"])

        # High-Risk Callout Highlights
        rf_eval = eval_results["models"]["Random Forest"]
        logreg_eval = eval_results["models"]["Logistic Regression"]

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">HIGH-RISK PRECISION</div>
                <div style="color: #10B981; font-size: 1.8rem; font-weight: 700;">{rf_eval['precision_high_risk']:.1%}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Random Forest Model</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">HIGH-RISK RECALL</div>
                <div style="color: #6366F1; font-size: 1.8rem; font-weight: 700;">{rf_eval['recall_high_risk']:.1%}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Detects At-Risk Teams</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">HELD-OUT ROC-AUC</div>
                <div style="color: #F59E0B; font-size: 1.8rem; font-weight: 700;">{rf_eval['roc_auc']:.3f}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">High Discriminative Power</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion Matrix Visuals across all 3 advanced models
        gb_eval = eval_results["models"]["Gradient Boosting"]
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        with col_cm1:
            st.subheader("Logistic Regression")
            cm_lr = logreg_eval["confusion_matrix"]
            cm_matrix_lr = [[cm_lr["true_negative"], cm_lr["false_positive"]],
                            [cm_lr["false_negative"], cm_lr["true_positive"]]]
            fig_cm_lr = px.imshow(
                cm_matrix_lr,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Low Risk (0)", "High Risk (1)"],
                y=["Low Risk (0)", "High Risk (1)"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm_lr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), height=280)
            st.plotly_chart(fig_cm_lr, use_container_width=True)

        with col_cm2:
            st.subheader("Random Forest")
            cm_rf = rf_eval["confusion_matrix"]
            cm_matrix_rf = [[cm_rf["true_negative"], cm_rf["false_positive"]],
                            [cm_rf["false_negative"], cm_rf["true_positive"]]]
            fig_cm_rf = px.imshow(
                cm_matrix_rf,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Low Risk (0)", "High Risk (1)"],
                y=["Low Risk (0)", "High Risk (1)"],
                text_auto=True,
                color_continuous_scale="Purples"
            )
            fig_cm_rf.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), height=280)
            st.plotly_chart(fig_cm_rf, use_container_width=True)

        with col_cm3:
            st.subheader("Gradient Boosting")
            cm_gb = gb_eval["confusion_matrix"]
            cm_matrix_gb = [[cm_gb["true_negative"], cm_gb["false_positive"]],
                            [cm_gb["false_negative"], cm_gb["true_positive"]]]
            fig_cm_gb = px.imshow(
                cm_matrix_gb,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Low Risk (0)", "High Risk (1)"],
                y=["Low Risk (0)", "High Risk (1)"],
                text_auto=True,
                color_continuous_scale="Teal"
            )
            fig_cm_gb.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), height=280)
            st.plotly_chart(fig_cm_gb, use_container_width=True)

        # Synthetic Data Limitations Markdown Section
        st.markdown("---")
        st.markdown(get_synthetic_data_limitations())


    # ==============================================================================
    # PAGE 7: ESCALATION & HUMAN REVIEW QUEUE
    # ==============================================================================
    elif menu == "🚨 Escalation & Human Review Queue":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🚨 Human-in-the-Loop Escalation Queue
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Mandatory human review for serious workplace complaints, safety threats, and ethics violations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        escalated_items = df_feedback[
            (df_feedback["escalation_status"] == "Escalated - Human Review Required") |
            (df_feedback["severity"] == "Critical")
        ].copy()

        st.markdown(
            f"""
            <div class="privacy-callout" style="border-left-color: #EF4444;">
                <b>Mandatory Policy Trigger:</b> All {len(escalated_items)} records below were flagged by automated ethics/safety filters or critical severity detectors. These cases are strictly sequestered for human HR review and cannot be closed automatically by machine learning algorithms.
            </div>
            """,
            unsafe_allow_html=True
        )

        for idx, row in escalated_items.iterrows():
            fb_id = row["feedback_id"]
            current_status = st.session_state["escalation_status_override"].get(fb_id, row["escalation_status"])
            notes = st.session_state["escalation_notes"].get(fb_id, "")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #F87171;">Case {fb_id} — {row['team_id']} ({row['department']})</h4>
                            <span class="badge-red">{current_status}</span>
                        </div>
                        <p style="color: #CBD5E1; margin: 8px 0; font-size: 0.95rem;"><b>Reason:</b> {row['escalation_reason']}</p>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; color: #F1F5F9; font-size: 0.9rem;">
                            <b>Anonymised Feedback:</b> {row['anonymised_text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # AI Case Dossier Expander / Generator
                dossier_key = f"dossier_{fb_id}"
                col_dos1, col_dos2 = st.columns([1, 2])
                with col_dos1:
                    generate_dossier = st.button(f"🤖 Generate AI Incident Dossier", key=f"btn_dossier_{fb_id}")
                with col_dos2:
                    st.caption(f"Generates compliance triage & non-retaliation roadmap with {prov_info['name']}")

                if generate_dossier:
                    with st.spinner("Compiling investigative dossier..."):
                        briefing = llm_inst.generate_investigative_briefing(
                            feedback_id=fb_id,
                            department=row["department"],
                            team_id=row["team_id"],
                            anonymised_text=row["anonymised_text"],
                            escalation_reason=row.get("escalation_reason", "Policy Escalation")
                        )
                        st.session_state[dossier_key] = briefing

                if dossier_key in st.session_state:
                    d_item = st.session_state[dossier_key]
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid rgba(239, 68, 68, 0.4); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; color: #F1F5F9; font-size: 0.92rem; line-height: 1.5;">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.4rem; margin-bottom: 0.6rem;">
                                <span style="font-weight: 700; color: #FCA5A5; font-size: 0.85rem; text-transform: uppercase;">AI Compliance Investigative Assessment</span>
                                <span style="font-size: 0.75rem; background: rgba(239,68,68,0.2); color: #FECACA; padding: 2px 6px; border-radius: 4px;">{d_item['provider']}</span>
                            </div>
                            {d_item['text']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                col_act1, col_act2 = st.columns([1, 2])
                with col_act1:
                    new_status = st.selectbox(
                        f"Update Status for {fb_id}",
                        ["Escalated - Human Review Required", "Under Investigation", "Action Plan Initiated", "Resolved & Closed"],
                        index=["Escalated - Human Review Required", "Under Investigation", "Action Plan Initiated", "Resolved & Closed"].index(current_status),
                        key=f"status_sel_{fb_id}"
                    )
                with col_act2:
                    new_notes = st.text_input(
                        f"Confidential HR Reviewer Notes ({fb_id})",
                        value=notes,
                        placeholder="Enter investigative actions or mitigation plan...",
                        key=f"notes_in_{fb_id}"
                    )

                if st.button(f"💾 Save Review Decision for {fb_id}", key=f"btn_{fb_id}"):
                    st.session_state["escalation_status_override"][fb_id] = new_status
                    st.session_state["escalation_notes"][fb_id] = new_notes
                    st.success(f"Audit log updated for Case {fb_id}.")


    # ==============================================================================
    # PAGE 8: PRIVACY, GOVERNANCE & METHODOLOGY
    # ==============================================================================
    elif menu == "🛡️ Privacy, Ethics & Methodology":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800;">
                    🛡️ Privacy Architecture, AI Ethics & Governance
                </h1>
                <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">
                    Comprehensive documentation of anonymity guarantees, non-retaliation ethics, and security guardrails.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        t1, t2, t3 = st.tabs(["🔒 Privacy Safeguards", "⚖️ AI Ethics & Non-Retaliation", "⚙️ Technical Architecture"])

        with t1:
            st.markdown(f"""
            ### 1. The k-Anonymity Standard (k ≥ {MIN_K_ANONYMITY})
            - **Subgroup Suppression**: Any demographic, departmental, or thematic slice containing fewer than 5 respondents is automatically suppressed from visual charts, data tables, and export functions.
            - **Quasi-Identifier Scrubbing**: Singular role mentions (e.g. "the only staff engineer in Austin") and unique tenure markers are detected and flagged for redaction.
            - **Direct PII Elimination**: Automated regex and heuristic NER scrub names, email addresses, phone numbers, employee badge IDs, and internal project codenames prior to any human or AI ingestion.
            """)

        with t2:
            st.markdown("""
            ### 2. Ethical Machine Learning Guidelines
            - **Strict Prohibition on Individual Prediction**:
            The platform operates strictly at the *aggregate team level*. Predicting individual flight risk or sentiment score is mathematically disallowed to prevent managerial surveillance and punitive actions.
            - **Non-Retaliation Safeguards**:
            Feedback is designed to empower organizational listening. No automated disciplinary action or compensation adjustments can be triggered by feedback data.
            - **Mandatory Human-in-the-Loop**:
            AI models serve solely as advisory prioritization filters. Severe concerns (harassment, ethical fraud, safety violations) immediately bypass algorithmic scoring and alert trained HR professionals.
            """)

        with t3:
            st.markdown("""
            ### 3. Architecture & Security Specifications
            - **Multi-Provider AI LLM Engine**: Seamlessly integrates Groq (Llama 3.3 70B), OpenAI (GPT-4o Mini), Google Gemini (1.5 Flash), Local Ollama, and an intelligent Local AI Reasoner simulation fallback for zero-key offline environments.
            - **Zero-Raw-PII Privacy Gateway**: All generative AI prompts strictly ingest scrubbed text passed through the client-side `AnonymizerEngine`. Direct personal identifiers (names, emails, phones, IDs, codenames) never reach external LLM APIs.
            - **Prompt-Injection Defense**: Pre-execution sanitizers detect adversarial instruction overrides (`ignore previous instructions`, `reveal system prompt`, `system tag manipulation`) and sanitize HTML/script tags.
            - **Reproducible Data Pipeline**: Fully seeded synthetic data generation engine guarantees verifiable metrics without handling confidential real-world PII.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 10px 0;">
            EchoGuard AI v1.0.0 • Ethical Anonymous Feedback Intelligence • Built with Streamlit & scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )
