# app.py
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import time, io, math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
import networkx as nx

# ================= THEME / CSS =================
st.set_page_config(page_title="Futuristic Math Lab", layout="wide", page_icon="🎛️")
NEON_CSS = """
<style>
:root{
  --bg:#0b0f19; --glass:#111827cc; --muted:#94a3b8; --text:#e5e7eb;
  --neon:#7c83ff; --aqua:#00ffd0;
}
html,body,[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% 0%, #0b1224 0%, var(--bg) 50%);
}
.block-container{ padding-top: 2.6rem; }
.header-wrap{ position:relative; overflow:visible; margin-bottom:.25rem; }
.neon-title{
  display:inline-block; padding:.25rem .6rem; border-radius:14px;
  font-size: 2.0rem; font-weight:800; color:var(--text);
  background: linear-gradient(180deg, rgba(17,24,39,.65), rgba(17,24,39,.35));
  border: 1px solid rgba(124,131,255,.28);
  text-shadow: 0 0 6px rgba(124,131,255,.6), 0 0 18px rgba(0,255,208,.18);
  box-shadow: 0 6px 26px rgba(0,0,0,.35), inset 0 0 0 1px rgba(124,131,255,.12);
}
.glass{
  background: var(--glass); border: 1px solid #1f2937; border-radius: 14px;
  box-shadow: 0 0 0 1px rgba(124,131,255,.12) inset, 0 10px 30px rgba(0,0,0,.35);
  padding: 1rem 1.2rem; backdrop-filter: blur(8px);
}
.kpi{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; }
.kpi .card{ background: linear-gradient(180deg, rgba(124,131,255,.08), rgba(124,131,255,.02));
  border: 1px solid rgba(124,131,255,.22); border-radius: 12px; padding:.9rem; }
.kpi .v{ font-size:1.05rem; font-weight:700; color:#fff }
.kpi .l{ font-size:.8rem; color:var(--muted) }
.stTabs [data-baseweb="tab-list"]{ gap:.25rem }
.stTabs [data-baseweb="tab"]{
  background: rgba(17,24,39,.6); border:1px solid #243043; border-radius:10px;
}
.stTabs [aria-selected="true"]{
  border-color: rgba(124,131,255,.6); box-shadow: 0 0 0 1px rgba(124,131,255,.25) inset;
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)
st.markdown('<div class="header-wrap"><div class="neon-title">🎛️ QuantX Playground  |SDEs • Algorithms • Numerical Methods|s</div></div>', unsafe_allow_html=True)


# ================= UTIL =================
def download_csv_from_paths(X, t, name):
    df = pd.DataFrame(X.T, columns=[f"path_{i}" for i in range(X.shape[0])])
    df.insert(0, "t", t)
    buf = io.StringIO(); df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", data=buf.getvalue(),
        file_name=f"{name}.csv", mime="text/csv")

# =================== MAIN TABS ===================
tab_sde, tab_algo, tab_num = st.tabs([
    "Stochastic Differential Equations",
    "Algorithms Lab",
    "Numerical Methods"
])

# ===========================================================
#                     S D E   L A B  (as before)
# ===========================================================
with tab_sde:
    if "seed" not in st.session_state: st.session_state.seed = 0
    def reseed(): st.session_state.seed = int(np.random.randint(1_000_000_000))

    colA, colB = st.columns([2,1])
    with colA:
        st.markdown('<div class="glass">Explore classic and advanced SDEs with live paths, histograms, and moments.</div>', unsafe_allow_html=True)
    with colB:
        if st.button("🎯 Resample / Rerun"): reseed(); st.rerun()
        st.caption("New randomness each click.")

    st.sidebar.header("SDE Controls")
    model = st.sidebar.selectbox(
        "Model",
        ["Brownian Motion (BM)","Geometric Brownian Motion (GBM)","Ornstein–Uhlenbeck (OU)",
         "CIR (square-root mean reversion)","Merton Jump-Diffusion (GBM + jumps)","Heston Stochastic Volatility"]
    )
    T = st.sidebar.slider("Time horizon T", 0.1, 10.0, 2.0, 0.1)
    N = st.sidebar.slider("Time steps (N)", 50, 5000, 1000, 50)
    paths = st.sidebar.slider("Number of paths", 1, 300, 80, 1)

    st.sidebar.subheader("Randomness")
    freeze = st.sidebar.checkbox("Freeze seed", value=False)
    custom_seed = st.sidebar.number_input("Seed", value=0, step=1)
    seed_to_use = (int(custom_seed) if (freeze and custom_seed!=0) else (int(st.session_state.seed) if st.session_state.seed else 0))
    if seed_to_use: np.random.seed(seed_to_use)

    dt = T/ N; t = np.linspace(0,T,N+1)
    mean_t = var_t = None

    if model=="Brownian Motion (BM)":
        mu = st.sidebar.slider("μ (drift)", -2.0, 2.0, 0.0, 0.1)
        sigma = st.sidebar.slider("σ (vol)", 0.0, 3.0, 1.0, 0.1)
        x0 = st.sidebar.number_input("X₀", value=0.0)
        dW = np.sqrt(dt)*np.random.randn(paths, N)
        X = np.zeros((paths, N+1)); X[:,0]=x0
        for i in range(N): X[:,i+1] = X[:,i] + mu*dt + sigma*dW[:,i]
        mean_t, var_t = x0+mu*t, (sigma**2)*t
        stats_text = "BM: E[X]=X₀+μt, Var=σ²t"

    elif model=="Geometric Brownian Motion (GBM)":
        mu = st.sidebar.slider("μ", -1.0, 1.0, 0.1, 0.01)
        sigma = st.sidebar.slider("σ", 0.0, 2.0, 0.3, 0.01)
        S0 = st.sidebar.number_input("S₀", value=1.0)
        dW = np.sqrt(dt)*np.random.randn(paths, N)
        X = np.zeros((paths, N+1)); X[:,0]=S0
        for i in range(N): X[:,i+1] = X[:,i]*np.exp((mu-0.5*sigma**2)*dt + sigma*dW[:,i])
        mean_t = S0*np.exp(mu*t); var_t = (S0**2)*np.exp(2*mu*t)*(np.exp(sigma**2*t)-1)
        stats_text = "GBM: lognormal; E[S]=S₀e^{μt}, Var=S₀²e^{2μt}(e^{σ²t}-1)"

    elif model=="Ornstein–Uhlenbeck (OU)":
        theta = st.sidebar.slider("θ (reversion)", 0.0, 5.0, 1.0, 0.1)
        mu_bar = st.sidebar.slider("μ̄ (long-run)", -5.0, 5.0, 0.0, 0.1)
        sigma = st.sidebar.slider("σ", 0.0, 3.0, 0.6, 0.1)
        x0 = st.sidebar.number_input("X₀", value=0.0)
        dW = np.sqrt(dt)*np.random.randn(paths, N)
        X = np.zeros((paths, N+1)); X[:,0]=x0
        for i in range(N): X[:,i+1] = X[:,i] + theta*(mu_bar-X[:,i])*dt + sigma*dW[:,i]
        e = np.exp(-theta*t); mean_t = mu_bar + (x0-mu_bar)*e
        var_t = (sigma**2)/(2*theta)*(1-np.exp(-2*theta*t)) if theta>0 else (sigma**2)*t
        stats_text = "OU: mean-reverting Gaussian; Var→σ²/(2θ)"

    elif model=="CIR (square-root mean reversion)":
        kappa = st.sidebar.slider("κ", 0.0, 10.0, 1.5, 0.1)
        theta_bar = st.sidebar.slider("θ (long-run level)", 0.0, 10.0, 1.0, 0.1)
        sigma = st.sidebar.slider("σ", 0.0, 5.0, 0.5, 0.1)
        x0 = st.sidebar.number_input("X₀ ≥ 0", value=0.5, min_value=0.0)
        dW = np.sqrt(dt)*np.random.randn(paths, N)
        X = np.zeros((paths, N+1)); X[:,0]=x0
        for i in range(N):
            x = np.maximum(X[:,i], 0.0)
            X[:,i+1] = x + kappa*(theta_bar-x)*dt + sigma*np.sqrt(x)*dW[:,i]
            X[:,i+1] = np.maximum(X[:,i+1], 0.0)
        stats_text = "CIR: √X noise; full-truncation Euler (keeps X≥0)"

    elif model=="Merton Jump-Diffusion (GBM + jumps)":
        mu = st.sidebar.slider("μ", -1.0, 1.0, 0.08, 0.01)
        sigma = st.sidebar.slider("σ (diffusion)", 0.0, 1.5, 0.25, 0.01)
        lam = st.sidebar.slider("λ (jump intensity)", 0.0, 10.0, 1.0, 0.1)
        mJ = st.sidebar.slider("mJ (log-mean)", -1.0, 1.0, -0.1, 0.01)
        sJ = st.sidebar.slider("sJ (log-sd)", 0.0, 1.5, 0.4, 0.01)
        S0 = st.sidebar.number_input("S₀", value=1.0)
        kappaJ = np.exp(mJ+0.5*sJ*sJ)-1.0
        X = np.zeros((paths, N+1)); X[:,0]=S0
        for i in range(N):
            dW = np.sqrt(dt)*np.random.randn(paths)
            K = np.random.poisson(lam*dt, size=paths)
            jump_log = np.where(K>0, mJ*K + sJ*np.sqrt(K)*np.random.randn(paths), 0.0)
            X[:,i+1] = X[:,i]*np.exp((mu-0.5*sigma**2 - lam*kappaJ)*dt + sigma*dW + jump_log)
        stats_text = "Merton: GBM + Poisson jumps with lognormal sizes"

    else:  # Heston
        mu = st.sidebar.slider("μ (drift)", -0.5, 0.5, 0.05, 0.01)
        kappa = st.sidebar.slider("κ (speed)", 0.0, 10.0, 1.5, 0.1)
        theta_bar = st.sidebar.slider("θ (long-run var)", 0.0, 1.0, 0.04, 0.005)
        xi = st.sidebar.slider("ξ (vol of var)", 0.0, 5.0, 0.5, 0.01)
        rho = st.sidebar.slider("ρ (corr)", -0.99, 0.99, -0.5, 0.01)
        S0 = st.sidebar.number_input("S₀", value=1.0)
        v0 = st.sidebar.number_input("v₀ ≥ 0", value=0.04, min_value=0.0, step=0.01)
        X = np.zeros((paths, N+1)); V = np.zeros((paths, N+1))
        X[:,0], V[:,0] = S0, v0
        for i in range(N):
            Z1, Z2 = np.random.randn(paths), np.random.randn(paths)
            dW1 = np.sqrt(dt)*Z1
            dW2 = np.sqrt(dt)*(rho*Z1 + np.sqrt(1-rho*rho)*Z2)
            v = np.maximum(V[:,i],0.0)
            V[:,i+1] = v + kappa*(theta_bar-v)*dt + xi*np.sqrt(v)*dW2
            V[:,i+1] = np.maximum(V[:,i+1],0.0)
            X[:,i+1] = X[:,i]*np.exp((mu-0.5*v)*dt + np.sqrt(v)*dW1)
        stats_text = "Heston: stochastic variance; full-truncation Euler"

    t1, t2, t3 = st.tabs(["Paths","Histogram @ T","Stats"])
    with t1:
        fig = go.Figure()
        for k in range(paths):
            fig.add_trace(go.Scatter(x=t, y=X[k], mode="lines", line=dict(width=1), showlegend=False))
        fig.update_layout(title="Sample Paths", xaxis_title="t", yaxis_title="X(t) / S(t)", height=460, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig2 = go.Figure(data=[go.Histogram(x=X[:,-1], nbinsx=60)])
        fig2.update_layout(title=f"Distribution at t = {T:.2f}", xaxis_title="X(T) / S(T)", yaxis_title="Count", height=460, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        st.markdown('<div class="glass kpi">', unsafe_allow_html=True)
        mean_final = float(X[:,-1].mean()); var_final = float(X[:,-1].var())
        st.markdown(f'<div class="card"><div class="l">E[X(T)] (sample)</div><div class="v">{mean_final:.6f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="l">Var[X(T)] (sample)</div><div class="v">{var_final:.6f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="l">Paths × Steps</div><div class="v">{paths} × {N}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if mean_t is not None and var_t is not None:
            with st.expander("Show theoretical E[X(t)] and Var[X(t)]"):
                figm = go.Figure(); figm.add_trace(go.Scatter(x=t,y=mean_t,mode="lines"))
                figm.update_layout(height=320, title="E[X(t)]", template="plotly_dark")
                st.plotly_chart(figm, use_container_width=True)
                figv = go.Figure(); figv.add_trace(go.Scatter(x=t,y=var_t,mode="lines"))
                figv.update_layout(height=320, title="Var[X(t)]", template="plotly_dark")
                st.plotly_chart(figv, use_container_width=True)
        st.caption(stats_text)
    st.subheader("Export simulated data")
    download_csv_from_paths(X, t, f"sde_{model.replace(' ','_')}")

# ===========================================================
#                A L G O R I T H M S   L A B  (extended)
# ===========================================================
with tab_algo:
    st.markdown('<div class="glass">Interactive CS algorithm visualizers with step controls and neon charts.</div>', unsafe_allow_html=True)
    algo_tab1, algo_tab2, algo_tab3 = st.tabs(["Sorting Visualizer", "Graph Algorithms", "A* Grid Path"])


    # ---------- Sorting ----------
    with algo_tab1:
        size = st.slider("Array size", 8, 160, 40, 1, key="sz")
        speed = st.slider("Speed (sec per step)", 0.0, 0.2, 0.02, 0.01, key="spd")
        algo = st.selectbox("Algorithm", ["Bubble","Insertion","Selection","Merge","Quick","Heap","Shell","Radix (LSD)"])
        np.random.seed(42); arr = np.random.permutation(size) + 1

        placeholder = st.empty()
        def draw(a, i=None, j=None, piv=None):
            colors = ["#7c83ff"]*len(a)
            if i is not None and 0<=i<len(a): colors[i] = "#00ffd0"
            if j is not None and 0<=j<len(a): colors[j] = "#ff6b6b"
            if piv is not None and 0<=piv<len(a): colors[piv] = "#f5b942"
            fig = go.Figure(data=[go.Bar(x=list(range(len(a))), y=a, marker=dict(color=colors))])
            fig.update_layout(height=360, template="plotly_dark", xaxis_title="Index", yaxis_title="Value",
                              margin=dict(l=20,r=20,b=40,t=30))
            placeholder.plotly_chart(fig, use_container_width=True)

        def bubble(a):
            a = a.copy(); n=len(a)
            for i in range(n):
                for j in range(0, n-i-1):
                    draw(a, j, j+1); time.sleep(speed)
                    if a[j] > a[j+1]:
                        a[j], a[j+1] = a[j+1], a[j]
                        draw(a, j, j+1); time.sleep(speed)
            draw(a); return a
        def insertion(a):
            a=a.copy()
            for i in range(1,len(a)):
                key=a[i]; j=i-1
                while j>=0 and a[j]>key:
                    a[j+1]=a[j]; j-=1; draw(a, j, j+1); time.sleep(speed)
                a[j+1]=key; draw(a, j+1); time.sleep(speed)
            draw(a); return a
        def selection(a):
            a=a.copy()
            for i in range(len(a)):
                m=i
                for j in range(i+1,len(a)):
                    draw(a, m, j); time.sleep(speed/2)
                    if a[j]<a[m]: m=j
                a[i],a[m]=a[m],a[i]; draw(a, i, m); time.sleep(speed)
            draw(a); return a
        def merge_sort(a):
            a=a.copy()
            def merge(l, m, r):
                L=a[l:m+1]; R=a[m+1:r+1]; i=j=0; k=l
                while i<len(L) and j<len(R):
                    a[k]=L[i] if L[i]<=R[j] else R[j]; k+=1; i+= (L[i]<=R[j]); j+= (L[i]>R[j])
                    draw(a, k-1); time.sleep(speed)
                while i<len(L): a[k]=L[i]; i+=1; k+=1; draw(a, k-1); time.sleep(speed)
                while j<len(R): a[k]=R[j]; j+=1; k+=1; draw(a, k-1); time.sleep(speed)
            def ms(l,r):
                if l<r:
                    m=(l+r)//2; ms(l,m); ms(m+1,r); merge(l,m,r)
            ms(0,len(a)-1); draw(a); return a
        def quick_sort(a):
            a=a.copy()
            def qs(l,r):
                if l>=r: return
                pivot=a[r]; i=l
                for j in range(l,r):
                    draw(a, j, r, piv=r); time.sleep(speed/2)
                    if a[j] <= pivot:
                        a[i],a[j]=a[j],a[i]; i+=1; draw(a, i-1, j, piv=r); time.sleep(speed/2)
                a[i],a[r]=a[r],a[i]; draw(a, i, r); time.sleep(speed)
                qs(l,i-1); qs(i+1,r)
            qs(0,len(a)-1); draw(a); return a
        def heap_sort(a):
            a=a.copy()
            def heapify(n,i):
                largest=i; l=2*i+1; r=2*i+2
                if l<n and a[l]>a[largest]: largest=l
                if r<n and a[r]>a[largest]: largest=r
                if largest!=i:
                    a[i],a[largest]=a[largest],a[i]; draw(a, i, largest); time.sleep(speed)
                    heapify(n,largest)
            n=len(a)
            for i in range(n//2-1,-1,-1): heapify(n,i)
            for i in range(n-1,0,-1):
                a[i],a[0]=a[0],a[i]; draw(a, 0, i); time.sleep(speed)
                heapify(i,0)
            draw(a); return a
        def shell_sort(a):
            a=a.copy(); gap=len(a)//2
            while gap>0:
                for i in range(gap,len(a)):
                    temp=a[i]; j=i
                    while j>=gap and a[j-gap]>temp:
                        a[j]=a[j-gap]; j-=gap; draw(a, j, j+gap); time.sleep(speed/2)
                    a[j]=temp; draw(a, j); time.sleep(speed/2)
                gap//=2
            draw(a); return a
        def radix_lsd(a):
            a=a.copy(); max_val=max(a); exp=1
            while max_val//exp>0:
                buckets=[[] for _ in range(10)]
                for v in a: buckets[(v//exp)%10].append(v)
                idx=0
                for b in buckets:
                    for v in b:
                        a[idx]=v; idx+=1; draw(a, idx-1); time.sleep(speed/4)
                exp*=10
            draw(a); return a

        st.write("")
        if st.button("▶️ Run / Animate"):
            draw(arr)
            {"Bubble":bubble,"Insertion":insertion,"Selection":selection,"Merge":merge_sort,
             "Quick":quick_sort,"Heap":heap_sort,"Shell":shell_sort,"Radix (LSD)":radix_lsd}[algo](arr)

    # ---------- Graph Algorithms ----------
    with algo_tab2:
        cols = st.columns(5)
        with cols[0]: n_nodes = st.number_input("Nodes", 6, 80, 18, 1)
        with cols[1]: p_edge = st.slider("Edge probability", 0.02, 1.0, 0.18, 0.01)
        with cols[2]: w_min  = st.number_input("Min weight", 1.0, 50.0, 1.0, 1.0)
        with cols[3]: w_max  = st.number_input("Max weight", 1.0, 50.0, 12.0, 1.0)
        with cols[4]: algo_g = st.selectbox("Algorithm", ["Dijkstra (weighted)","A* (Euclidean)","BFS (unweighted)","DFS (unweighted)","Prim MST"])

        np.random.seed(7)
        G = nx.gnp_random_graph(int(n_nodes), float(p_edge), directed=False)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps)-1):
                u = list(comps[i])[0]; v = list(comps[i+1])[0]; G.add_edge(u,v)
        for (u,v) in G.edges(): G.edges[u,v]['weight'] = float(np.random.uniform(w_min, w_max))
        nodes = list(G.nodes())
        s = st.selectbox("Start", nodes, index=0)
        tnode = st.selectbox("Target", nodes, index=min(1, len(nodes)-1))
        pos = nx.spring_layout(G, seed=3, dim=2)

        path, length = [], np.inf; edges_highlight=[]
        try:
            if algo_g.startswith("Dijkstra"):
                path = nx.shortest_path(G, s, tnode, weight='weight')
                length = nx.shortest_path_length(G, s, tnode, weight='weight')
                edges_highlight = list(zip(path[:-1], path[1:]))
            elif algo_g.startswith("A*"):
                def h(a,b):
                    ax,ay = pos[a]; bx,by = pos[b]; return math.hypot(ax-bx, ay-by)
                path = nx.astar_path(G, s, tnode, heuristic=lambda x,y: h(x,y), weight='weight')
                length = nx.path_weight(G, path, weight='weight')
                edges_highlight = list(zip(path[:-1], path[1:]))
            elif algo_g.startswith("BFS"):
                path = nx.shortest_path(G, s, tnode); length = len(path)-1
                edges_highlight = list(zip(path[:-1], path[1:]))
            elif algo_g.startswith("DFS"):
                T = nx.dfs_tree(G, s)
                if tnode in T:
                    path = nx.shortest_path(T, s, tnode); length = len(path)-1
                    edges_highlight = list(zip(path[:-1], path[1:]))
            else:  # Prim
                T = nx.minimum_spanning_tree(G, algorithm="prim", weight='weight')
                edges_highlight = list(T.edges()); length = T.size(weight='weight')
        except nx.NetworkXNoPath:
            path, length = [], np.inf

        x_edges, y_edges = [], []
        for (u,v) in G.edges():
            x_edges += [pos[u][0], pos[v][0], None]
            y_edges += [pos[u][1], pos[v][1], None]
        edge_trace = go.Scatter(x=x_edges, y=y_edges, mode='lines',
                                line=dict(width=1, color="#4b5563"), hoverinfo='none')
        hx, hy = [], []
        for (u,v) in edges_highlight:
            hx += [pos[u][0], pos[v][0], None]; hy += [pos[u][1], pos[v][1], None]
        path_trace = go.Scatter(x=hx, y=hy, mode='lines',
                                line=dict(width=4, color="#00ffd0"), hoverinfo='none')
        node_x, node_y, colors = [], [], []
        for n in G.nodes():
            node_x.append(pos[n][0]); node_y.append(pos[n][1])
            colors.append("#00ffd0" if n in path else "#7c83ff")
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=[str(n) for n in G.nodes()],
            textposition="top center",
            marker=dict(size=14, color=colors, line=dict(width=1, color="#111827"))
        )
        title = ("Prim MST (total weight = " + f"{length:.2f})") if algo_g=="Prim MST" \
                else (("No path" if not path and algo_g!="Prim MST" else f"{algo_g}: length = {length:.2f}"))
        figG = go.Figure(data=[edge_trace, path_trace, node_trace])
        figG.update_layout(height=520, template="plotly_dark", showlegend=False,
                           margin=dict(l=10,r=10,t=30,b=10), title=title)
        st.plotly_chart(figG, use_container_width=True)
        st.caption("Tip: change algorithm or node choices to recompute. Hover edges to see weights.")

    with algo_tab3:
        st.write("Draw walls on the grid, then run A*. You can drag to paint/erase.")
        cols = st.columns(5)
        with cols[0]:
            GRID = st.number_input("Grid size (N×N)", 10, 80, 30, 1)
        with cols[1]:
            BRUSH = st.slider("Brush size", 1, 30, 10, 1)
        with cols[2]:
            allow_diag = st.checkbox("Allow diagonals", value=True)
        with cols[3]:
            heuristic_name = st.selectbox("Heuristic", ["Manhattan", "Euclidean", "Chebyshev"])
        with cols[4]:
            erase_mode = st.checkbox("Eraser", value=False)

        # Canvas to draw walls (black = wall)
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=BRUSH,
            stroke_color="#000000" if not erase_mode else "#0b0f19",
            background_color="#0b0f19",
            height=500, width=500, drawing_mode="freedraw", key="astar_canvas",
            display_toolbar=True, update_streamlit=True
        )

        # Convert drawing to occupancy grid
        import numpy as np
        occ = np.zeros((GRID, GRID), dtype=np.uint8)
        if canvas.image_data is not None:
            # Downsample the canvas image to GRID×GRID; treat dark pixels as walls
            img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
            img_small = img.resize((GRID, GRID), resample=Image.BILINEAR)
            arr = np.array(img_small)  # 0..255 (0 is black)
            occ = (arr < 50).astype(np.uint8)  # 1 = wall, 0 = free

        c2 = st.columns(4)
        with c2[0]:
            sx = st.number_input("Start x", 0, GRID-1, 0)
        with c2[1]:
            sy = st.number_input("Start y", 0, GRID-1, 0)
        with c2[2]:
            gx = st.number_input("Goal x", 0, GRID-1, GRID-1)
        with c2[3]:
            gy = st.number_input("Goal y", 0, GRID-1, GRID-1)

        # --- A* implementation on grid ---
        import heapq, math
        def neighbors(x,y):
            steps4 = [(1,0),(-1,0),(0,1),(0,-1)]
            steps8 = steps4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
            for dx,dy in (steps8 if allow_diag else steps4):
                nx, ny = x+dx, y+dy
                if 0<=nx<GRID and 0<=ny<GRID and occ[ny,nx]==0:
                    yield nx, ny, (math.hypot(dx,dy) if allow_diag else 1.0)

        def h(ax,ay,bx,by):
            if heuristic_name=="Manhattan":  return abs(ax-bx)+abs(ay-by)
            if heuristic_name=="Chebyshev":  return max(abs(ax-bx), abs(ay-by))
            return math.hypot(ax-bx, ay-by)

        def astar(start, goal):
            sx,sy = start; gx,gy = goal
            if occ[sy,sx]==1 or occ[gy,gx]==1: return [], np.inf
            pq = []; heapq.heappush(pq,(0,(sx,sy)))
            g = {(sx,sy):0.0}; parent = {(sx,sy):None}
            while pq:
                _, (x,y) = heapq.heappop(pq)
                if (x,y)==(gx,gy): break
                for nx,ny,c in neighbors(x,y):
                    ng = g[(x,y)] + c
                    if ng < g.get((nx,ny), float("inf")):
                        g[(nx,ny)] = ng; parent[(nx,ny)] = (x,y)
                        f = ng + h(nx,ny,gx,gy)
                        heapq.heappush(pq,(f,(nx,ny)))
            if (gx,gy) not in parent: return [], np.inf
            # reconstruct
            path = [(gx,gy)]
            cur = (gx,gy)
            while parent[cur] is not None:
                cur = parent[cur]; path.append(cur)
            path.reverse()
            return path, g[(gx,gy)]

        if st.button("⭐ Run A*"):
            path, cost = astar((sx,sy),(gx,gy))

            # Plot as heatmap + path overlay
            import plotly.graph_objs as go
            grid_show = occ.copy().astype(float)
            grid_show[sy,sx] = -0.5  # start
            grid_show[gy,gx] = -1.0  # goal
            fig = go.Figure(data=go.Heatmap(
                z=grid_show, colorscale=[ [0.0,"#00ffd0"], [0.05,"#00ffd0"],  # goal
                                        [0.1,"#7c83ff"],  [0.49,"#7c83ff"],  # start
                                        [0.5,"#0b0f19"], [0.9,"#0b0f19"],   # free
                                        [1.0,"#222"] ],                      # walls
                showscale=False))
            if path:
                xs = [p[0]+0.5 for p in path]; ys = [p[1]+0.5 for p in path]
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",
                                        line=dict(width=4,color="#00ffd0"),
                                        marker=dict(size=6,color="#00ffd0"),
                                        name=f"path cost {cost:.2f}"))
                title = f"A* path length ≈ {cost:.2f}"
            else:
                title = "No path"
            fig.update_layout(template="plotly_dark", height=520,
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            title=title)
            st.plotly_chart(fig, use_container_width=True)
            if path:
                st.caption("Tip: Toggle Eraser to open corridors; move start/goal; rerun.")

# ===========================================================
#                 N U M E R I C A L   M E T H O D S
# ===========================================================
with tab_num:
    st.markdown('<div class="glass">Monte-Carlo estimators, Runge–Kutta ODE solvers, and Fourier transforms — all interactive.</div>', unsafe_allow_html=True)
    nm1, nm2, nm3, nm4 = st.tabs(["🎲 Monte Carlo", "🌀 Runge–Kutta ODE", "🎼 Fourier Lab", "🖼️ 2-D FFT Image Filter"])

    # --------- Monte Carlo ----------
    with nm1:
        mc_mode = st.selectbox("Estimator", ["π via unit square", "European Call (GBM, MC)"])
        sims = st.slider("Simulations / Samples", 1000, 2_000_000, 100_000, step=1000)
        if mc_mode.startswith("π"):
            # π ≈ 4 * (# points inside quarter circle) / N
            rng = np.random.default_rng(123)
            x = rng.random(sims); y = rng.random(sims)
            inside = (x*x + y*y) <= 1.0
            pi_hat = 4.0 * inside.mean()
            se = 4.0 * np.sqrt(inside.mean()*(1-inside.mean())/sims)
            ci = (pi_hat - 1.96*se, pi_hat + 1.96*se)

            fig = go.Figure()
            samp = min(5000, sims)
            fig.add_trace(go.Scatter(x=x[:samp], y=y[:samp],
                                     mode="markers", marker=dict(size=3, color=np.where((x[:samp]**2+y[:samp]**2)<=1,"#00ffd0","#7c83ff")),
                                     showlegend=False))
            theta = np.linspace(0, np.pi/2, 300)
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                     line=dict(width=2, color="#f5b942"), showlegend=False))
            fig.update_layout(template="plotly_dark", height=420, title="Monte Carlo π (quarter circle)")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("π estimate", f"{pi_hat:.6f}")
            st.caption(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]   •   Std. error ≈ {se:.6f}")

        else:
            # European Call under GBM using antithetic variates
            S0 = st.number_input("S₀", value=100.0)
            K  = st.number_input("Strike K", value=100.0)
            r  = st.number_input("Risk-free r", value=0.02, step=0.01, format="%.4f")
            mu = st.number_input("Drift μ", value=0.08, step=0.01, format="%.4f")
            sigma = st.number_input("Vol σ", value=0.25, step=0.01, format="%.4f")
            T  = st.number_input("Maturity T (yrs)", value=1.0, step=0.25)
            n  = sims // 2
            Z  = np.random.randn(n)
            ST1 = S0*np.exp((mu-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
            ST2 = S0*np.exp((mu-0.5*sigma**2)*T + sigma*np.sqrt(T)*(-Z))
            payoff = np.maximum(ST1-K,0)*0.5 + np.maximum(ST2-K,0)*0.5
            disc = np.exp(-r*T)
            price = disc * payoff.mean()
            se = disc * payoff.std(ddof=1)/np.sqrt(n)
            ci = (price - 1.96*se, price + 1.96*se)

            fig = go.Figure(data=[go.Histogram(x=np.concatenate([ST1,ST2]), nbinsx=60)])
            fig.update_layout(template="plotly_dark", height=420, title="Terminal price S(T) distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Call price (MC, antithetic)", f"{price:.4f}")
            st.caption(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]   •   Std. error ≈ {se:.4f}")

    # --------- Runge–Kutta ----------
    with nm2:
        st.write("Choose an ODE and solver; compare trajectories.")
        ode = st.selectbox("ODE", ["y' = λy (exponential)","Logistic y' = r y (1 − y/K)","Harmonic Oscillator y'' + ω² y = 0"])
        solver = st.selectbox("Solver", ["Euler", "RK2 (midpoint)", "RK4"])
        T = st.slider("Time horizon", 1.0, 40.0, 10.0, 0.5)
        N = st.slider("Steps (N)", 10, 5000, 500, 10)
        t = np.linspace(0, T, N+1); dt = T/N

        def euler_step(f, y, t, dt):  return y + dt * f(t, y)
        def rk2_step(f, y, t, dt):
            k1 = f(t, y); k2 = f(t+0.5*dt, y+0.5*dt*k1); return y + dt*k2
        def rk4_step(f, y, t, dt):
            k1 = f(t, y)
            k2 = f(t+0.5*dt, y+0.5*dt*k1)
            k3 = f(t+0.5*dt, y+0.5*dt*k2)
            k4 = f(t+dt, y+dt*k3)
            return y + (dt/6.0)*(k1+2*k2+2*k3+k4)

        if ode.startswith("y' = λy"):
            lam = st.number_input("λ", value=-0.5, step=0.1, format="%.3f")
            y0 = st.number_input("y(0)", value=1.0)
            def f(t, y): return lam*y
            step = {"Euler":euler_step,"RK2 (midpoint)":rk2_step,"RK4":rk4_step}[solver]
            y = np.zeros_like(t); y[0]=y0
            for i in range(N): y[i+1] = step(f, y[i], t[i], dt)
            y_exact = y0*np.exp(lam*t)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t,y=y,mode="lines",name=solver))
            fig.add_trace(go.Scatter(x=t,y=y_exact,mode="lines",name="Exact",line=dict(dash="dash")))
            fig.update_layout(template="plotly_dark", height=420, title="Exponential ODE")
            st.plotly_chart(fig, use_container_width=True)
            err = np.linalg.norm(y - y_exact, ord=np.inf)
            st.caption(f"Max absolute error vs. exact: {err:.3e}")

        elif ode.startswith("Logistic"):
            r = st.number_input("r", value=1.0, step=0.1)
            K = st.number_input("K", value=10.0, step=1.0)
            y0 = st.number_input("y(0)", value=1.0, step=0.1)
            def f(t, y): return r*y*(1 - y/K)
            step = {"Euler":euler_step,"RK2 (midpoint)":rk2_step,"RK4":rk4_step}[solver]
            y = np.zeros_like(t); y[0]=y0
            for i in range(N): y[i+1] = step(f, y[i], t[i], dt)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t,y=y,mode="lines",name=solver))
            fig.update_layout(template="plotly_dark", height=420, title="Logistic growth (no simple exact shown)")
            st.plotly_chart(fig, use_container_width=True)

        else:  # Harmonic Oscillator -> system: y1' = y2 ; y2' = -ω^2 y1
            w = st.number_input("ω", value=1.5, step=0.1)
            y10 = st.number_input("y(0)", value=1.0)
            y20 = st.number_input("y'(0)", value=0.0)
            def step_sys(y, t, dt, method):
                y1,y2 = y
                def f(t, Y): return np.array([Y[1], -w*w*Y[0]])
                if method=="Euler":   return y + dt * f(t, y)
                if method=="RK2 (midpoint)":
                    k1=f(t,y); k2=f(t+0.5*dt, y+0.5*dt*k1); return y + dt*k2
                k1=f(t,y); k2=f(t+0.5*dt, y+0.5*dt*k1); k3=f(t+0.5*dt, y+0.5*dt*k2); k4=f(t+dt, y+dt*k3)
                return y + (dt/6)*(k1+2*k2+2*k3+k4)
            method = solver
            Y = np.zeros((N+1,2)); Y[0]=[y10,y20]
            for i in range(N): Y[i+1]=step_sys(Y[i], t[i], dt, method)
            y_exact = y10*np.cos(w*t) + (y20/w)*np.sin(w*t)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t,y=Y[:,0],mode="lines",name=f"{solver} (y)"))
            fig.add_trace(go.Scatter(x=t,y=y_exact,mode="lines",name="Exact y",line=dict(dash="dash")))
            fig.update_layout(template="plotly_dark", height=420, title="Harmonic oscillator")
            st.plotly_chart(fig, use_container_width=True)

    # --------- Fourier Lab ----------
    with nm3:
        st.write("Build a signal, see its spectrum, and reconstruct. Try adding noise.")
        fs = st.slider("Sampling rate (Hz)", 64, 4096, 1024, 32)
        dur = st.slider("Duration (s)", 0.25, 5.0, 2.0, 0.25)
        t = np.arange(0, dur, 1/fs)
        # up to 3 tones
        with st.expander("Tones"):
            f1 = st.slider("f1 (Hz)", 1, 400, 50);  a1 = st.slider("a1", 0.0, 2.0, 1.0, 0.05)
            f2 = st.slider("f2 (Hz)", 1, 400, 120); a2 = st.slider("a2", 0.0, 2.0, 0.6, 0.05)
            f3 = st.slider("f3 (Hz)", 1, 400, 220); a3 = st.slider("a3", 0.0, 2.0, 0.3, 0.05)
        noise = st.slider("Add noise (σ)", 0.0, 1.0, 0.15, 0.01)
        sig = a1*np.sin(2*np.pi*f1*t) + a2*np.sin(2*np.pi*f2*t) + a3*np.sin(2*np.pi*f3*t) + noise*np.random.randn(len(t))

        # FFT
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(t), d=1/fs)
        mag = np.abs(X) / len(t)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=sig, mode="lines"))
            fig.update_layout(template="plotly_dark", height=340, title="Time domain")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=freqs, y=mag))
            fig2.update_layout(template="plotly_dark", height=340, title="Magnitude spectrum |X(f)|", xaxis_title="Hz")
            st.plotly_chart(fig2, use_container_width=True)

        # Simple band-pass reconstruction
        with st.expander("Reconstruct with band-pass"):
            f_lo = st.slider("Keep ≥ f_lo (Hz)", 0, int(fs//2), 20)
            f_hi = st.slider("Keep ≤ f_hi (Hz)", 0, int(fs//2), 300)
            keep = (freqs>=f_lo) & (freqs<=f_hi)
            X_filt = X * keep
            sig_rec = np.fft.irfft(X_filt, n=len(t))
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=t, y=sig_rec, mode="lines"))
            fig3.update_layout(template="plotly_dark", height=320, title="Reconstructed signal (IFFT)")
            st.plotly_chart(fig3, use_container_width=True)

        # Spectrogram (short-time FFT)
        with st.expander("Spectrogram"):
            win_ms = st.slider("Window (ms)", 5, 200, 40, 5)
            hop_ms = st.slider("Hop (ms)", 2, 100, 10, 2)
            win = int(fs*win_ms/1000); hop = int(fs*hop_ms/1000)
            if win < 8: win = 8
            frames = 1 + (len(sig)-win)//hop if len(sig)>=win else 0
            spec = []
            for i in range(frames):
                seg = sig[i*hop : i*hop + win] * np.hanning(win)
                S = np.fft.rfft(seg)
                spec.append(20*np.log10(np.maximum(np.abs(S), 1e-12)))
            if spec:
                spec = np.array(spec).T  # (freq, time)
                figS = px.imshow(spec, origin="lower", aspect="auto",
                                 color_continuous_scale="Turbo",
                                 labels=dict(x="Frame", y="Freq bin", color="dB"))
                figS.update_layout(template="plotly_dark", height=360, title="Spectrogram (dB)")
                st.plotly_chart(figS, use_container_width=True)
            else:
                st.info("Signal shorter than one window; increase duration or reduce window.")

# =============== NEW: 2-D FFT Image Filter ===============
    with nm4:
        st.write("Upload an image, mask frequencies, and reconstruct.")
        up = st.file_uploader("Image (png/jpg)", type=["png","jpg","jpeg"])
        if up:
            img = Image.open(up).convert("L")  # grayscale
        else:
            # fallback demo image (generated stripes)
            w,h = 512, 512
            x = np.linspace(0, 20*np.pi, w)
            stripes = (np.sin(x)[None,:]*127 + 128).astype(np.uint8)
            img = Image.fromarray(np.repeat(stripes, h, axis=0))

        img = ImageOps.exif_transpose(img)
        max_side = 512
        scale = max(img.size)/max_side
        if scale>1: img = img.resize((int(img.width/scale), int(img.height/scale)), Image.BICUBIC)

        arr = np.array(img, dtype=float)
        F = np.fft.fftshift(np.fft.fft2(arr))
        mag = np.log1p(np.abs(F))

        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(); fig.add_trace(go.Image(z=arr))
            fig.update_layout(template="plotly_dark", title="Original", height=360,
                            xaxis_showticklabels=False, yaxis_showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(data=go.Heatmap(z=mag, colorscale="Turbo"))
            fig2.update_layout(template="plotly_dark", title="Log magnitude spectrum", height=360,
                            xaxis_showticklabels=False, yaxis_showticklabels=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Mask")
        shape = st.selectbox("Shape", ["Low-pass (circle)", "High-pass (circle)", "Band-pass (annulus)", "Notch (two stripes)"])
        H, W = arr.shape
        cy, cx = H//2, W//2
        R1 = st.slider("R1 (inner)", 1, min(H,W)//2, min(H,W)//6)
        R2 = st.slider("R2 (outer)", R1, min(H,W)//2, min(H,W)//3)

        Y,X = np.ogrid[:H,:W]
        dist = np.sqrt((Y-cy)**2 + (X-cx)**2)
        mask = np.zeros_like(arr, dtype=float)

        if shape.startswith("Low-pass"):
            mask = (dist <= R1).astype(float)
        elif shape.startswith("High-pass"):
            mask = (dist >= R1).astype(float)
        elif shape.startswith("Band-pass"):
            mask = ((dist >= R1) & (dist <= R2)).astype(float)
        else:  # Notch (kill horizontal/vertical stripes)
            bw = st.slider("Notch width", 1, 50, 6)
            mask = np.ones_like(arr, dtype=float)
            mask[:, cx-bw:cx+bw] = 0.0
            mask[cy-bw:cy+bw, :] = 0.0

        Ff = F * mask
        rec = np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))
        rec = np.clip(rec, 0, 255)

        c3,c4 = st.columns(2)
        with c3:
            fig3 = go.Figure(data=go.Heatmap(z=mask, colorscale="Greys"))
            fig3.update_layout(template="plotly_dark", title="Mask", height=360,
                            xaxis_showticklabels=False, yaxis_showticklabels=False)
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            fig4 = go.Figure(); fig4.add_trace(go.Image(z=rec))
            fig4.update_layout(template="plotly_dark", title="Reconstructed", height=360,
                            xaxis_showticklabels=False, yaxis_showticklabels=False)
            st.plotly_chart(fig4, use_container_width=True)

        # Download
        from PIL import Image as PILImage
        out = PILImage.fromarray(rec.astype(np.uint8))
        buf = io.BytesIO(); out.save(buf, format="PNG")
        st.download_button("⬇️ Download filtered image", data=buf.getvalue(),
                        file_name="fft_filtered.png", mime="image/png")
