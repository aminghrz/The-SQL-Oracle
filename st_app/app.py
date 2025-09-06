import streamlit as st

st.set_page_config(
    page_title="SQL Oracle Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* 1) Just import your font globally – no RTL here */
  @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
  html, body, [class*="st-"] {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
  }

  /* 2) Make the chat‐input itself RTL */
  [data-testid="stChatInput"] {
    direction: rtl;
  }
  [data-testid="stChatInput"] input,
  [data-testid="stChatInput"] input::placeholder {
    direction: rtl;
    text-align: right;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
  }

  /* 3) Force each chat‐message bubble back into a normal L→R flex row */
  [data-testid="stChatMessage"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center;
  }

  /* 4) Give the icon (first child) a little right margin */
  [data-testid="stChatMessage"] > div:first-child {
    margin-right: 8px;
  }

  /* 5) Make only the text-container (last child) RTL and right-aligned */
  [data-testid="stChatMessage"] > div:last-child {
    flex: 1;                /* fill remaining width */
    direction: rtl !important;
    text-align: right !important;
  }

  /* 6) If you’re using markdown inside the bubble… */
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    direction: rtl !important;
    text-align: right !important;
  }
</style>
""", unsafe_allow_html=True)
# Sidebar for navigation
st.sidebar.title("🔮 SQL Oracle")

# Navigation
nav = st.navigation([
    st.Page("pages/chat_interface.py", title="💬 Chat Interface", icon="💬"),
    st.Page("pages/graph_visualization.py", title="🕸️ Graph Visualization", icon="🕸️"),
])

nav.run()
# # Import and run the selected page
# if page == "💬 Chat Interface":
#     from pages import chat_interface
#     chat_interface.run()
# elif page == "🕸️ Graph Visualization":
#     from pages import graph_visualization
#     graph_visualization.run()

# # Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application provides a natural language interface to query databases "
    "and visualize the relationships between tables."
)