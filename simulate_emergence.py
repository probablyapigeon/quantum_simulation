import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def compute_pure_python_curvature(G):
    edges_to_sample = list(G.edges())
    if len(edges_to_sample) > 500:
        np.random.seed(42)
        idx = np.random.choice(len(edges_to_sample), 500, replace=False)
        edges_to_sample = [edges_to_sample[i] for i in idx]
        
    curvatures = []
    for u, v in edges_to_sample:
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        shared_neighbors = len(list(nx.common_neighbors(G, u, v)))
        edge_curvature = 4 - deg_u - deg_v + (3 * shared_neighbors)
        curvatures.append(edge_curvature)
    return np.mean(curvatures) if curvatures else 0

def run_simulation(matrix_size=128, steps=150):
    print(f"=== Starting Gradient-Safe Dual-Defect Simulation (Matrix Size: {matrix_size}x{matrix_size}) ===")
    
    X1 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    X2 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    
    optimizer = torch.optim.Adam([X1, X2], lr=0.005)
    
    defect_mask_1 = torch.zeros(matrix_size, matrix_size)
    defect_mask_2 = torch.zeros(matrix_size, matrix_size)
    
    # Seeding distinct coordinate sub-blocks as mass defect targets
    defect_mask_1[15:35, 15:35] = 2.5
    defect_mask_2[90:110, 90:110] = -2.5
    
    defect_a_nodes = set(range(15, 35))
    defect_b_nodes = set(range(90, 110))
    
    for step in range(steps):
        optimizer.zero_grad()
        
        X1_effective = X1 + defect_mask_1
        X2_effective = X2 + defect_mask_2
        
        commutator = torch.mm(X1_effective, X2_effective) - torch.mm(X2_effective, X1_effective)
        action = torch.trace(torch.mm(commutator, commutator))
        
        action.backward()
        optimizer.step()
        
        if step % 30 == 0:
            print(f"Step {step:03d} | Total Space Action S: {action.item():.2f}")
            
    with torch.no_grad():
        X1_final = X1 + defect_mask_1
        X2_final = X2 + defect_mask_2
        correlation = torch.abs(torch.mm(X1_final, X2_final)).numpy()
        
    threshold = np.percentile(correlation, 93)
    adj_matrix = (correlation > threshold).astype(int)
    
    G = nx.from_numpy_array(adj_matrix)
    print(f"\nGenerated Manifold | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    
    # --- VISUALIZATION PIPELINE ---
    print("Mapping the warped spacetime manifold...")
    plt.figure(figsize=(16, 16), dpi=300)
    plt.facecolor = '#05050A'
    ax = plt.gca()
    ax.set_facecolor('#05050A')
    
    pos = nx.fruchterman_reingold_layout(G, k=2.5/np.sqrt(len(G.nodes)), iterations=300, seed=42)
    
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        if node in defect_a_nodes or node in defect_b_nodes:
            node_sizes.append(250)
            node_colors.append('#00FFFF')
        else:
            deg = G.degree(node)
            node_sizes.append(deg * 12)
            node_colors.append(deg)
            
    nx.draw_networkx_edges(G, pos, alpha=0.10, edge_color='#5A5A7A', width=0.8)
    
    bg_nodes = [n for n in G.nodes() if n not in defect_a_nodes and n not in defect_b_nodes]
    bg_colors = [node_colors[n] for n in bg_nodes]
    bg_sizes = [node_sizes[n] for n in bg_nodes]
    
    if bg_nodes:
        nodes_gfx = nx.draw_networkx_nodes(
            G, pos, nodelist=bg_nodes, node_size=bg_sizes, node_color=bg_colors,
            cmap=plt.cm.plasma, alpha=0.85, linewidths=0.3, edgecolors='#05050A'
        )
        
    fg_nodes = [n for n in G.nodes() if n in defect_a_nodes or n in defect_b_nodes]
    if fg_nodes:
        fg_sizes = [node_sizes[n] for n in fg_nodes]
        # Fixed parameter from 'edge_color' to 'edgecolors' here
        nx.draw_networkx_nodes(
            G, pos, nodelist=fg_nodes, node_size=fg_sizes, node_color='#00FFFF',
            alpha=1.0, linewidths=0.8, edgecolors='#FFFFFF'
        )
        
    mean_curv = compute_pure_python_curvature(G)
    
    plt.title(
        "Dual Particle Infused Emergent Spacetime\n"
        f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Warped Curvature: {mean_curv:.2f}", 
        color='#FFFFFF', fontsize=16, fontweight='bold', pad=25
    )
    
    plt.axis('off')
    output_filename = "dual_particle_universe.png"
    plt.savefig(output_filename, facecolor=ax.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccess! Map with dual matter defects saved to: .\\{output_filename}")

if __name__ == "__main__":
    run_simulation()
