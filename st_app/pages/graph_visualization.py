import streamlit as st
import requests
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import os
from typing import Dict, Any
import json

# Disable proxy for localhost
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

BASE_URL = "http://localhost:9995"

def get_graph_data():
    """Fetch graph data from API"""
    try:
        response = requests.get(f"{BASE_URL}/graph", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting graph: {e}")
        return None

def get_table_relationships(table_name: str):
    """Get relationships for a specific table"""
    try:
        response = requests.get(f"{BASE_URL}/graph/table/{table_name}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

# def build_canvas_legend(node_type_colors, edge_type_colors):
#     legend_nodes, legend_edges = [], []

#     # Top-left corner of the canvas (tune as needed)
#     x0, y0 = -600, -400
#     spacing = 36

#     # Node type legend (colored dots with labels)
#     for i, (typ, color) in enumerate(node_type_colors.items()):
#         legend_nodes.append(Node(
#             id=f"legend:node:{typ}",
#             label=typ.title(),
#             color=color,
#             size=18,
#             shape="dot",
#             font={"color": "#ffffff"},
#             x=x0,
#             y=y0 + i * spacing,
#             fixed=True,
#             physics=False,
#         ))

#     # Edge type legend (a small edge with a label on the right node)
#     x1 = x0 + 220
#     for i, (etype, color) in enumerate(edge_type_colors.items()):
#         a_id = f"legend:edge:{i}:a"
#         b_id = f"legend:edge:{i}:b"
#         y = y0 + i * spacing

#         # Small anchor dots
#         legend_nodes.extend([
#             Node(id=a_id, label="", color="#95a5a6", size=6, shape="dot",
#                  x=x1, y=y, fixed=True, physics=False),
#             Node(id=b_id, label=etype, color="#2c3e50", size=6, shape="dot",
#                  font={"color": "#ffffff"}, x=x1 + 75, y=y, fixed=True, physics=False),
#         ])

#         legend_edges.append(Edge(
#             source=a_id,
#             target=b_id,
#             color=color,
#             width=2,
#             length=70
#         ))

#     return legend_nodes, legend_edges

def create_agraph_visualization(graph_data: Dict[str, Any], selected_table: str = None):
    """Create an interactive graph using streamlit-agraph"""
    if not graph_data or not graph_data.get('success'):
        st.error("No graph data available")
        return

    graph = graph_data['graph']
    nodes_data = graph['nodes']
    edges_data = graph['edges']

    node_type_colors = {
        'fact': '#ff6b6b',
        'dimension': '#4ecdc4',
        'junction': '#45b7d1',
        'transaction': '#f7b731',
        'reference': '#5f27cd',
        'other': '#95a5a6',
        'unknown': '#dfe6e9',
    }

    edge_type_colors = {
        'explicit_fk': '#27ae60',
        'table_similarity': '#3498db',
        'column_similarity': '#e67e22',
        'fk_candidate_from_summary': '#f39c12',
        'fk_from_table_summary': '#e91e63',
        'content_inclusion': '#9b59b6',
        'via_junction': '#e74c3c',
        'fact_dimension_similarity': '#795548',
        'fk_like_inclusion': '#16a085',
    }

    nodes = []
    for node in nodes_data:
        connections = sum(1 for e in edges_data if e['source'] == node['id'] or e['target'] == node['id'])
        size = 25 + min(connections * 5, 50)

        if selected_table and node['id'] == selected_table:
            size *= 1.5
            color = '#e74c3c'
        else:
            color = node_type_colors.get(node['type'], '#95a5a6')

        label = f"{node['label']}\n({node['type']})"
        if node['row_count']:
            label += f"\n{node['row_count']:,} rows"

        nodes.append(Node(
            id=node['id'],
            label=label,
            size=size,
            color=color,
            title=(
                f"{node['full_name']} "
                f"Type: {node['type']} "
                f"Rows: {node['row_count']:,} "
                f"Purpose: {node['purpose'] or 'N/A'} "
                f"Confidence: {node['metadata']['confidence']:.1%}"
            ),
            font={"color": "#ffffff"}

        ))

    edges = []
    for edge in edges_data:
        if not st.session_state.get('show_weak_edges', False) and edge['weight'] < 0.6:
            continue

        if selected_table and (edge['source'] == selected_table or edge['target'] == selected_table):
            width = 5
            color = '#e74c3c'
        else:
            width = 1 + edge['weight'] * 4
            color = edge_type_colors.get(edge['type'], '#95a5a6')

        edges.append(Edge(
            source=edge['source'],
            target=edge['target'],
            label=f"{edge['weight']:.2f}" if st.session_state.get('show_edge_labels', False) else "",
            color=color,
            width=width,
            title=f"<b>{edge['type']}</b><br>Weight: {edge['weight']:.2f}<br>{edge['explanation']}",
            length=300
        ))

    config = Config(
        width=1200,
        height=800,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )

    # legend_nodes, legend_edges = build_canvas_legend(node_type_colors, edge_type_colors)
    # nodes.extend(legend_nodes)
    # edges.extend(legend_edges)

    return nodes, edges, config



def run():
    st.title("🕸️ Database Relationship Graph")
    st.markdown("Interactive visualization of table relationships")

    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None

    with st.sidebar:
        st.markdown("### Graph Controls")

        if st.button("🔄 Refresh Graph"):
            st.session_state.graph_data = None

        st.session_state.show_weak_edges = st.checkbox(
            "Show weak relationships",
            value=False,
            help="Display edges with weight < 0.6",
        )

        st.session_state.show_edge_labels = st.checkbox(
            "Show edge weights",
            value=False,
            help="Display weight values on edges",
        )

        # st.markdown("### Legend")
        # st.markdown("#### Node Types")
        # node_types = {
        #     '🔴 Fact': 'Transaction/event tables',
        #     '🔵 Dimension': 'Reference/lookup tables',
        #     '🟢 Junction': 'Many-to-many relationship tables',
        #     '🟡 Transaction': 'Business transaction tables',
        #     '🟣 Reference': 'Static reference data',
        #     '⚪ Other': 'Other table types',
        # }
        # for icon_type, desc in node_types.items():
        #     st.markdown(f"{icon_type}: {desc}")

        # st.markdown("#### Edge Types")
        # edge_types = {
        #     'explicit_fk': 'Foreign Key (database)',
        #     'fk_like_inclusion': 'FK-like (data inclusion)',
        #     'column_similarity': 'Similar columns',
        #     'table_similarity': 'Similar tables',
        #     'via_junction': 'Connected via junction',
        #     'content_inclusion': 'Content overlap',
        # }
        # for edge_type, desc in edge_types.items():
        #     st.markdown(f"• **{edge_type}**: {desc}")

    if st.session_state.graph_data is None:
        with st.spinner("Loading graph data..."):
            st.session_state.graph_data = get_graph_data()

    if st.session_state.graph_data and st.session_state.graph_data.get('success'):
        graph_data = st.session_state.graph_data
        stats = graph_data['graph']['statistics']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tables", stats['nodes'])
        with col2:
            st.metric("Relationships", stats['edges'])
        with col3:
            st.metric("Graph Density", f"{stats['density']:.1%}")
        with col4:
            st.metric("Avg Weight", f"{stats.get('avg_weight', 0):.2f}")

        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col1:
            table_names = [node['id'] for node in graph_data['graph']['nodes']]
            selected_table = st.selectbox(
                "Select a table to highlight its relationships:",
                ["None"] + sorted(table_names),
                index=0,
            )

        with col2:
            if selected_table and selected_table != "None":
                if st.button("📊 View Details"):
                    st.session_state.show_table_details = selected_table

        if hasattr(st.session_state, 'show_table_details') and st.session_state.show_table_details:
            table_name = st.session_state.show_table_details
            with st.expander(f"📊 Details for {table_name}", expanded=True):
                rel_data = get_table_relationships(table_name)
                if rel_data and rel_data.get('success'):
                    node_info = rel_data['node_info']
                    st.markdown(f"**Type:** {node_info['type']}")
                    st.markdown(f"**Purpose:** {node_info.get('purpose', 'N/A')}")
                    st.markdown(f"**Row Count:** {node_info.get('row_count', 0):,}")
                    if node_info.get('key_columns'):
                        st.markdown(f"**Key Columns:** {', '.join(node_info['key_columns'])}")

                    summary = rel_data['summary']
                    st.markdown(f"**Total Connections:** {summary['total_connections']}")
                    st.markdown(f"**Strong Connections:** {summary['strong_connections']}")

                    if rel_data['outgoing_edges']:
                        st.markdown("#### Outgoing Relationships")
                        for edge in rel_data['outgoing_edges'][:5]:
                            st.markdown(f"→ **{edge['to_table']}** (weight: {edge['weight']:.2f})")
                            st.caption(f" {edge['explanation']}")

                    if rel_data['incoming_edges']:
                        st.markdown("#### Incoming Relationships")
                        for edge in rel_data['incoming_edges'][:5]:
                            st.markdown(f"← **{edge['from_table']}** (weight: {edge['weight']:.2f})")
                            st.caption(f" {edge['explanation']}")

                    if st.button("Close Details"):
                        del st.session_state.show_table_details

        st.markdown("---")

        result = create_agraph_visualization(
            graph_data,
            selected_table if selected_table != "None" else None,
        )
        if result:
            nodes, edges, config = result
            return_value = agraph(nodes=nodes, edges=edges, config=config)
            if return_value:
                st.info(f"Selected: {return_value}")

        st.markdown("---")
        st.markdown("### Graph Analysis")

        all_tables = set(node['id'] for node in graph_data['graph']['nodes'])
        connected_tables = set()
        for edge in graph_data['graph']['edges']:
            connected_tables.add(edge['source'])
            connected_tables.add(edge['target'])

        isolated_tables = all_tables - connected_tables

        if isolated_tables:
            st.warning(f"⚠️ **Isolated Tables:** {len(isolated_tables)} tables have no relationships")
            with st.expander("View isolated tables"):
                for table in sorted(isolated_tables):
                    st.markdown(f"• {table}")

        if 'common_paths' in graph_data['graph'] and graph_data['graph']['common_paths']:
            st.markdown("### Common Join Paths")
            with st.expander("View common join paths"):
                for path in graph_data['graph']['common_paths'][:10]:
                    path_str = " → ".join([step['from'] for step in path['path']])
                    if path['path']:
                        path_str += f" → {path['path'][-1]['to']}"
                    st.markdown(f"**{path['from']} to {path['to']}**")
                    st.caption(f"{path_str} (weight: {path['total_weight']:.2f})")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export Graph Data"):
                graph_json = json.dumps(graph_data['graph'], indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=graph_json,
                    file_name="graph_data.json",
                    mime="application/json",
                )

        with col2:
            if st.button("📊 Generate Network Statistics"):
                G = nx.DiGraph()
                for node in graph_data['graph']['nodes']:
                    G.add_node(node['id'], **node)
                for edge in graph_data['graph']['edges']:
                    G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

                with st.expander("Network Statistics", expanded=True):
                    st.markdown("#### Centrality Measures")

                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)

                    st.markdown("**Most Connected Tables (Degree Centrality):**")
                    for node, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"• {node}: {score:.3f}")

                    st.markdown("**Key Bridge Tables (Betweenness Centrality):**")
                    for node, score in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"• {node}: {score:.3f}")

                    if G.is_directed():
                        num_weakly_connected = nx.number_weakly_connected_components(G)
                        num_strongly_connected = nx.number_strongly_connected_components(G)
                        st.markdown(f"**Weakly Connected Components:** {num_weakly_connected}")
                        st.markdown(f"**Strongly Connected Components:** {num_strongly_connected}")

    else:
        st.error("Failed to load graph data. Please check if the API is running.")

# Ensure the page renders via Navigation API
run()