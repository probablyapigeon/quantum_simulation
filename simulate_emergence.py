import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run_animated_simulation(matrix_size=64, total_frames=60):
    print(f"=== Starting Animated Simulation Engine (Substrate: {matrix_size}x{matrix_size}) ===")
    print("Preparing quantum fields for time-evolution recording...")
    
    # Tier 1: Setup baseline coordinate matrices
    X1 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    X2 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    optimizer = torch.optim.Adam([X1, X2], lr=0.01)
    
    # Build static gradient-safe defect masks (Seeding mass points)
    defect_mask_1 = torch.zeros(matrix_size, matrix_size)
    defect_mask_2 = torch.zeros(matrix_size, matrix_size)
    defect_mask_1[10:20, 10:20] = 3.0
    defect_mask_2[45:55, 45:55] = -3.0
    
    defect_nodes = set(range(10, 20)).union(set(range(45, 55)))
    
    # Initialize the plot canvas
    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#05050A')
    ax = plt.gca()
    
    # Fix positioning using a static template graph to ensure smooth node movement tracking
    # If nodes jump randomly every frame, the video looks chaotic. This fixes the spatial tracking framework.
    print("Pre-calculating spatial layout pathways...")
    X1_init = X1.detach() + defect_mask_1
    X2_init = X2.detach() + defect_mask_2
    corr_init = torch.abs(torch.mm(X1_init, X2_init)).numpy()
    G_template = nx.from_numpy_array((corr_init > np.percentile(corr_init, 90)).astype(int))
    fixed_positions = nx.fruchterman_reingold_layout(G_template, k=2.0/np.sqrt(64), iterations=100, seed=42)

    def update_frame(frame_idx):
        ax.clear()
        ax.set_facecolor('#05050A')
        plt.axis('off')
        
        # Run 3 mini optimization loops per video frame to speed up physics visualization
        for _ in range(3):
            optimizer.zero_grad()
            X1_eff = X1 + defect_mask_1
            X2_eff = X2 + defect_mask_2
            commutator = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
            action = torch.trace(torch.mm(commutator, commutator))
            action.backward()
            optimizer.step()
            
        # Extract the current state of emergent geometry links
        with torch.no_grad():
            X1_f = X1 + defect_mask_1
            X2_f = X2 + defect_mask_2
            correlation = torch.abs(torch.mm(X1_f, X2_f)).numpy()
            
        threshold = np.percentile(correlation, 92)
        adj_matrix = (correlation > threshold).astype(int)
        G = nx.from_numpy_array(adj_matrix)
        
        # Parse visual aesthetics based on mass defects vs background spacetime vacuum
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            if node in defect_nodes:
                node_sizes.append(150)
                node_colors.append('#00FFFF') # Cyan mass defect loop
            else:
                deg = G.degree(node) if node in G else 0
                node_sizes.append(max(deg * 5, 10))
                node_colors.append('#9B51E0') # Purple background canvas
                
        # Render edges and nodes cleanly onto the active movie frame
        nx.draw_networkx_edges(G, fixed_positions, alpha=0.08, edge_color='#5A5A7A', width=0.6, ax=ax)
        
        # Split drawing to guarantee cyan matter knots sit natively on top of space
        bg_n = [n for n in G.nodes() if n not in defect_nodes]
        fg_n = [n for n in G.nodes() if n in defect_nodes]
        
        if bg_n:
            nx.draw_networkx_nodes(G, fixed_positions, nodelist=bg_n, node_size=[node_sizes[i] for i in bg_n], 
                                   node_color='#2D1B4E', alpha=0.7, edgecolors='#05050A', ax=ax)
        if fg_n:
            nx.draw_networkx_nodes(G, fixed_positions, nodelist=fg_n, node_size=[node_sizes[i] for i in fg_n], 
                                   node_color='#00FFFF', alpha=1.0, edgecolors='#FFFFFF', ax=ax)
            
        ax.set_title(
            f"Dynamic Emergent Spacetime Evolution\n"
            f"Simulation Frame: {frame_idx:03d} / {total_frames} | Action S: {action.item():.2f}",
            color='#FFFFFF', fontsize=12, fontweight='bold', pad=10
        )

    print("\nEncoding video timelines (this pass maps frame-by-frame structural evolution)...")
    ani = animation.FuncAnimation(fig, update_frame, frames=total_frames, interval=100)
    
    # Save using standard HTML5-friendly canvas packaging (bypasses raw system compiler errors)
    output_filename = "spacetime_collapse.gif"
    ani.save(output_filename, writer='pillow', fps=10)
    plt.close()
    
    print(f"\nSuccess! Time-lapse animation saved cleanly to: .\\{output_filename}")

if __name__ == "__main__":
    run_animated_simulation()
