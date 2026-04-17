import streamlit as st


def show_strategy_cards(strategies):
    """展示各策略的卡片式信息"""
    cols = st.columns(len(strategies))

    for i, s in enumerate(strategies):
        with cols[i]:
            st.markdown(f"### {s['name']}")
            st.markdown(f"**Action:** {s['action']}")
            st.markdown(f"**Confidence:** {s['confidence']}/5")
            st.markdown(f"**Risk:** {s['expected_risk']}")
            st.markdown(f"**Expected Return:** {s['expected_return']}")

            # 风险色条
            risk_color = {"Low": "🟩", "Medium": "🟨", "High": "🟥"}.get(s["expected_risk"], "⬜️")
            st.markdown(f"{risk_color} **Risk Level**")

            st.progress(s["confidence"] / 5)
