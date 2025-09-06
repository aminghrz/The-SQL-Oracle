import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from streamlit.components.v1 import html

# Disable proxy for localhost
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

BASE_URL = "http://localhost:9995"

st.title("💬 SQL Oracle Chat Interface")

# st.markdown("""
# <style>
# /* Make the specific text_input RTL */
# div[data-testid="stTextInput"] input[placeholder="Ask your question!"],
# div[data-testid="stTextInput"] input[aria-label="Ask a question:"] {
#   direction: rtl;
#   text-align: right;
# }
# </style>
# """, unsafe_allow_html=True)

html("""
<script>
const sel = 'div[data-testid="stTextInput"] input[placeholder="Ask your question!"]';
const trySet = () => {
  const el = window.parent.document.querySelector(sel);
  if (el) {
    el.setAttribute('dir', 'auto');   // auto-detect base direction
    el.style.textAlign = 'start';     // align with the detected base direction
  } else { setTimeout(trySet, 5); }
};
trySet();
</script>
""", height=0)

def check_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def query_agent(prompt):
    """Send a query to the agent"""
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"prompt": prompt},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def display_visualization(viz_data):
    """Display Plotly visualization from JSON data"""
    if viz_data:
        try:
            fig_json = json.loads(viz_data['figure'])
            fig = go.Figure(fig_json)
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            return fig
        except Exception as e:
            st.error(f"Error displaying visualization: {e}")
            return None
    return None

def run():
    st.markdown("Ask questions about your database in natural language")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.spinner("Checking API connection..."):
        health = check_health()

    if not health:
        st.error("❌ Cannot connect to the SQL Oracle API. Please ensure it's running on port 9995.")
        st.stop()
    else:
        st.success("✅ Connected to SQL Oracle API")

    st.markdown("---")

    for i, (query, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"""
                    <div dir="rtl" style="text-align:right; font-family:'Vazirmatn', Tahoma, sans-serif">
                    🧑 You: {query}
                    </div>
                    """,
                    unsafe_allow_html=True)
            # st.markdown(f"**🧑 You:** {query}")

            if response['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{response.get('confidence', 0):.1%}")
                with col2:
                    st.metric("Rows", response.get('row_count', 0))
                with col3:
                    st.metric("Time", f"{response.get('execution_time', 0):.2f}s")
                with col4:
                    st.metric("Tables", len(response.get('tables_used', [])))

                if response.get('explanation'):
                    st.markdown("**🤖 Explanation:**")
                    st.markdown(f"""
                    <div dir="rtl" style="text-align:right; font-family:'Vazirmatn', Tahoma, sans-serif">
                    {response['explanation']}
                    </div>
                    """,
                    unsafe_allow_html=True)

                if response.get('summary'):
                    st.info(f"📋 **Summary:** {response['summary']}")

                if response.get('insights'):
                    st.markdown("**💡 Insights:**")
                    for insight in response['insights']:
                        st.markdown(f"• {insight}")

                with st.expander("🔍 View SQL Query"):
                    st.code(response.get('sql_query', 'No SQL query available'), language='sql')

                if response.get('data') and response['row_count'] > 0:
                    st.markdown("**📊 Results:**")
                    df = pd.DataFrame(response['data'])

                    if len(df) > 100:
                        st.dataframe(df.head(100))
                        st.caption(f"Showing first 100 of {len(df)} rows")
                    else:
                        st.dataframe(df)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                if response.get('visualization'):
                    st.markdown("**📈 Visualization:**")
                    fig = display_visualization(response['visualization'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                if response.get('quality_assessment'):
                    st.success(f"✅ {response['quality_assessment']}")

                if response.get('validation_details', {}).get('issues'):
                    st.warning("⚠️ **Validation Issues:**")
                    for issue in response['validation_details']['issues']:
                        st.markdown(f"• {issue}")

                if response.get('next_steps'):
                    with st.expander("🔄 Suggested next steps"):
                        for step in response['next_steps']:
                            st.markdown(f"• {step}")
            else:
                st.error(f"❌ **Error:** {response.get('error', 'Unknown error')}")

            st.markdown("---")

    with st.form(key='query_form', clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_input(
                "Ask a question:",
                placeholder="Ask your question!",
                label_visibility="collapsed",
            )
        with col2:
            submit_button = st.form_submit_button("🚀 Send", use_container_width=True)

    if submit_button and user_query:
        with st.spinner("🔮 Processing your query..."):
            response = query_agent(user_query)
        st.session_state.chat_history.append((user_query, response))
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Ensure the page renders via Navigation API
run()