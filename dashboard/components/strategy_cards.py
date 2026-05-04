import streamlit as st


def show_strategy_cards(strategies):
    """Display card-style information for each strategy."""
    cols = st.columns(len(strategies))

    for i, s in enumerate(strategies):
        with cols[i]:
            st.markdown(f"### {s['name']}")
            st.markdown(f"**Action:** {s['action']}")
            st.markdown(f"**Confidence:** {s['confidence']}/5")
            st.markdown(f"**Risk:** {s['expected_risk']}")
            st.markdown(f"**Expected Return:** {s['expected_return']}")

            # Risk color bar
            risk_color = {"Low": "green", "Medium": "yellow", "High": "red"}.get(s["expected_risk"], "gray")
            st.markdown(f"**Risk Level:** {risk_color}")

            st.progress(s["confidence"] / 5)
