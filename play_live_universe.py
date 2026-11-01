import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time

def run_zero_dependency_simulation(matrix_size=64, audio_path=r"C:\Users\kiers\Downloads\He Can Try (Demo_Mix).mp3"):
    print(f"=== Initializing Pure-Python Live Spacetime Player ===")
    
    # Initialize basic coordinate fields
    X1 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    X2 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    optimizer = torch.optim.Adam([X1, X2], lr=0.01)
    
    total_frames = 250 
    
    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor('#05050A')
    ax = plt.gca()
    plt.axis('off')
    
    print("Mapping static spatial topology template...")
    corr_init = torch.abs(torch.mm(X1.detach(), X2.detach())).numpy()
    G_template = nx.from_numpy_array((corr_init > np.percentile(corr_init, 90)).astype(int))
    fixed_positions = nx.fruchterman_reingold_layout(G_template, k=2.0/np.sqrt(matrix_size), iterations=100, seed=42)

    # Pre-generate an algorithmic rhythmic wave matching a 120 BPM track structure
    t = np.linspace(0, 20, total_frames)
    bass_timeline = np.abs(np.sin(2 * np.pi * 2.0 * t)) * np.exp(-2 * (t % 0.5))
    treble_timeline = np.abs(np.cos(2 * np.pi * 0.5 * t)) * 0.5 + np.random.normal(0, 0.1, total_frames)

    def update_live_frame(frame_idx):
        if frame_idx >= total_frames: return
        ax.clear()
        ax.set_facecolor('#05050A')
        plt.axis('off')
        
        # Pull rhythmic modifiers for this frame
        bass_volt = min(max(float(bass_timeline[frame_idx] * 4.5), 0.0), 5.0)
        highs_volt = min(max(float(treble_timeline[frame_idx] * 2.5), 0.0), 3.0)
        mids_volt = min(max(float((bass_volt + highs_volt) * 0.4), 0.0), 4.0)
        
        # Inject dynamic frequency stress fields into the matrix configurations
        audio_defect_mask = torch.zeros(matrix_size, matrix_size)
        audio_defect_mask[22:42, 22:42] = bass_volt * 3.0
        audio_defect_mask[0:10, :] = highs_volt * 1.5
        audio_defect_mask[:, 54:64] = mids_volt * 2.0
        
        # Run matrix optimization pass
        for _ in range(2):
            optimizer.zero_grad()
            X1_eff = X1 + audio_defect_mask
            X2_eff = X2 + audio_defect_mask
            commutator = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
            action = torch.trace(torch.mm(commutator, commutator))
            action.backward()
            optimizer.step()
            
        with torch.no_grad():
            X1_f = X1 + audio_defect_mask
            X2_f = X2 + audio_defect_mask
            correlation = torch.abs(torch.mm(X1_f, X2_f)).numpy()
            
        dynamic_percentile = 94.0 - (mids_volt * 0.5)
        threshold = np.percentile(correlation, max(min(dynamic_percentile, 95.0), 88.0))
        adj_matrix = (correlation > threshold).astype(int)
        G = nx.from_numpy_array(adj_matrix)
        
        node_sizes = [max(G.degree(n) * 6, 8) if n in G else 8 for n in G.nodes()]
        node_colors = [G.degree(n) + (highs_volt * 4.0) if n in G else 0 for n in G.nodes()]
        
        nx.draw_networkx_edges(G, fixed_positions, alpha=0.10, edge_color='#00FFCC', width=0.7, ax=ax)
        if len(G.nodes) > 0:
            nx.draw_networkx_nodes(G, fixed_positions, node_size=node_sizes, node_color=node_colors,
                                   cmap=plt.cm.plasma, alpha=0.85, linewidths=0.3, edgecolors='#05050A', ax=ax)
        
        ax.set_title(
            f"Live Spacetime Playback Stream\n"
            f"Core Stress: {bass_volt:.2f}V | Boundary Stress: {highs_volt:.2f}V",
            color='#FFFFFF', fontsize=12, fontweight='bold', pad=10
        )

    # --- THE ASYNCHRONOUS WINDOWS MEDIA HANDOFF ---
    if os.path.exists(audio_path):
        print(f"\nHooking into Windows media framework to launch your music track...")
        # Uses the native OS shell execution to launch the audio file completely in the background
        os.system(f'start "" "{audio_path}"')
        time.sleep(0.5) # Give the hardware player a brief moment to initialize channels
    else:
        print(f"\n[!] Notice: Audio file not found at local path. Running visualizer loop independently.")

    print("Launching fluid player layout matrix. Close graphic frame to end simulation.")
    ani = animation.FuncAnimation(fig, update_live_frame, frames=total_frames, interval=45, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_zero_dependency_simulation()
