import plotly.graph_objects as go
import streamlit as st

def plot_portfolio_chart(df):
    """Plot account total value curve"""
    fig = go.Figure()

    for col in df.columns[1:]:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[col],
            mode='lines+markers',
            name=col
        ))

    fig.update_layout(
        title="Total Account Value ($)",
        xaxis_title="Date",
        yaxis_title="Account Value (USD)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, width='stretch')
