from pathlib import Path

import streamlit as st

from pages.analytics_dashboard import render_dashboard


def show_model_comparison_legend():
    """Render a small legend explaining comparison result colors."""
    st.markdown(
        """
    <div style="display:flex;gap:15px;margin:10px 0;font-size:0.9rem;">
        <span style="background:#ff444440;padding:2px 10px;border-radius:4px;">🔴 SPAM</span>
        <span style="background:#00d4aa40;padding:2px 10px;border-radius:4px;">🟢 HAM</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Load unified global styles once
def load_global_styles():
    """Inject the global CSS stylesheet (assets/styles.css) once per session."""
    key = "__global_css_loaded__"
    if not st.session_state.get(key, False):
        # page_functions.py sits in the repo root, so assets/ is a sibling folder
        css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
        try:
            css = css_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            st.session_state[key] = True
        except Exception as e:
            st.warning(f"Could not load global styles from {css_path}: {e}")

    # Also inject theme CSS variables
    try:
        from assets.theme_manager import theme_css_variables

        st.markdown(theme_css_variables(), unsafe_allow_html=True)
    except ImportError:
        pass


# navigate_to is intentionally NOT stored as a module-level variable.
# Callers must pass it explicitly to show_feedback_page(navigate_to=...).


def ui_error_boundary(func):
    """Decorator to isolate component rendering errors and present recovery UI."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"⚠️ A rendering error occurred in this component: {e!s}")
            if st.button("🔄 Reload Page / Recover UI", key=f"recover_{func.__name__}"):
                st.rerun()

    return wrapper


def show_feedback_page(navigate_to):
    """Feedback page for user comments, suggestions, and bug reports"""
    # Import the feedback handler
    try:
        from models.feedback_handler import FeedbackHandler

        feedback_handler = FeedbackHandler()
    except ImportError:
        st.warning("Feedback handler not found. Feedback will not be saved.")
        feedback_handler = None

    # Feedback page header
    st.markdown(
        """
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1a1a1a, #2d2d2d); border-radius: 15px; margin-bottom: 30px; border: 1px solid #404040;">
        <h1 style="color: #00d4aa; font-size: 3rem; margin: 0; text-shadow: 0 0 20px rgba(0, 212, 170, 0.3);">
            💬 Feedback
        </h1>
        <p style="color: #d1d1d1; margin: 10px 0 0; font-size: 1.2rem;">
            Help us improve Spamlyser by sharing your thoughts!
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for feedback form and submission status
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "feedback_rating" not in st.session_state:
        st.session_state.feedback_rating = 3
    if "feedback_context" not in st.session_state:
        st.session_state.feedback_context = None

    # Check if feedback was just submitted
    if st.session_state.feedback_submitted:
        st.success(
            "🎉 Thank you for your feedback! Your input helps make Spamlyser better for everyone."
        )

        # Add a button to submit another feedback
        if st.button("Submit Another Feedback"):
            st.session_state.feedback_submitted = False
            st.rerun()
    else:
        # Main feedback form
        with st.container():
            st.markdown("""
            ## 📝 Share Your Feedback

            Your insights are valuable to us! Use this form to:
            - Report bugs or issues
            - Request new features
            - Suggest improvements
            - Share your experience
            """)

            # Create a form to collect feedback
            with st.form("feedback_form"):
                # Feedback type selection
                feedback_type = st.selectbox(
                    "Type of Feedback",
                    options=[
                        "Bug Report",
                        "Feature Request",
                        "Suggestion",
                        "Question",
                        "Compliment",
                        "Other",
                    ],
                    index=2,
                    help="Select the category that best describes your feedback",
                )

                # Context display if available
                if st.session_state.feedback_context:
                    st.info(
                        f"Providing feedback about: **{st.session_state.feedback_context}**"
                    )

                # Experience rating
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("### Rate Your Experience")
                with col2:
                    rating = st.slider(
                        "How would you rate your experience with Spamlyser?",
                        min_value=1,
                        max_value=5,
                        value=st.session_state.feedback_rating,
                        help="1 = Poor, 5 = Excellent",
                        label_visibility="collapsed",
                    )

                # Rating stars visualization
                st.markdown(
                    f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    {"⭐" * rating}{"☆" * (5 - rating)}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Detailed feedback
                feedback_message = st.text_area(
                    "Detailed Feedback",
                    height=150,
                    max_chars=1000,
                    help="Please provide details about your feedback. What worked well? What could be improved?",
                    placeholder="Share your thoughts, suggestions, or report issues here...",
                )

                # Optional email for follow-up
                st.markdown("### Contact Information (Optional)")
                email = st.text_input(
                    "Email Address",
                    help="Provide your email if you'd like us to follow up on your feedback",
                    placeholder="your.email@example.com (optional)",
                )

                # Privacy note
                st.markdown(
                    """
                <div style="font-size: 0.8rem; color: #888888; margin-bottom: 15px;">
                    <i>Your email will only be used to respond to your feedback if necessary and will not be shared with third parties.</i>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Submit button
                submit_button = st.form_submit_button(
                    "Submit Feedback", use_container_width=True, type="primary"
                )

                # Handle form submission
                if submit_button:
                    if not feedback_message:
                        st.error("Please provide some feedback before submitting.")
                    else:
                        # Prepare feedback data
                        feedback_data = {
                            "feedback_type": feedback_type,
                            "rating": rating,
                            "message": feedback_message,
                            "email": email if email else None,
                            "context": st.session_state.feedback_context
                            if st.session_state.feedback_context
                            else "General",
                        }

                        # Reset context after using it
                        st.session_state.feedback_context = None

                        # Save feedback if handler is available
                        if feedback_handler:
                            success = feedback_handler.save_feedback(feedback_data)
                            if success:
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_rating = rating
                                st.rerun()
                            else:
                                st.error(
                                    "There was an error saving your feedback. Please try again later."
                                )
                        else:
                            # Mock success if handler is not available (for demo)
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_rating = rating
                            st.rerun()

        # Additional information
        with st.expander("Why We Value Your Feedback"):
            st.markdown("""
            ### 🚀 Improving Together

            At Spamlyser, we believe in continuous improvement, and your feedback is essential to this process. Here's how your input helps:

            - **Bug Reports**: Help us identify and fix issues quickly
            - **Feature Requests**: Guide our development roadmap based on user needs
            - **Suggestions**: Provide insights on how we can enhance user experience
            - **Questions**: Help us identify areas where documentation could be improved

            ### 🔄 Feedback Loop

            We review all feedback regularly and use it to prioritize improvements and new features. If you've provided your email, we may reach out for clarification or to let you know when your suggestion has been implemented.
            """)

    # Navigation buttons
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            navigate_to("home")
    with col2:
        if st.button("🔍 Try SMS Analysis", use_container_width=True):
            navigate_to("analyzer")
    with col3:
        if st.button("❓ Get Help", use_container_width=True):
            navigate_to("help")
