import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import miniaudio
import os
import csv
import math

def compute_pure_python_curvature(G):
    edges_to_sample = list(G.edges())
    if len(edges_to_sample) > 200:
        np.random.seed(42)
        idx = np.random.choice(len(edges_to_sample), 200, replace=False)
        edges_to_sample = [edges_to_sample[i] for i in idx]
    curvatures = []
    for u, v in edges_to_sample:
        deg_u, deg_v = G.degree(u), G.degree(v)
        shared = len(list(nx.common_neighbors(G, u, v)))
        curvatures.append(4 - deg_u - deg_v + (3 * shared))
    return np.mean(curvatures) if curvatures else 0

def run_fourier_analysis_3d(matrix_size=64, audio_path=r"C:\Users\kiers\Downloads\He Can Try (Demo_Mix).mp3"):
    print(f"=== Initializing Orbital Phase-Space Trajectory Engine ===")
    
    if not os.path.exists(audio_path):
        print(f"[!] Error: Target file not found at: {audio_path}")
        return

    print("Decoding native MP3 parameters...")
    sound = miniaudio.decode_file(audio_path)
    sample_rate = sound.sample_rate
    data = np.array(sound.samples, dtype=np.float32)
    if sound.nchannels > 1:
        data = data.reshape(-1, sound.nchannels)[:, 0]
    audio_signal = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
    
    frames_per_second = 12
    total_frames = min(int((len(audio_signal) / sample_rate) * frames_per_second), 250)
    samples_per_frame = int(sample_rate / frames_per_second)
    
    # Initialize position matrices with a clean, circular orbital offset
    theta_init = np.linspace(0, 2 * np.pi, matrix_size)
    X1 = torch.tensor(np.diag(np.cos(theta_init) * 0.4), dtype=torch.float32)
    X2 = torch.tensor(np.diag(np.sin(theta_init) * 0.4), dtype=torch.float32)
    
    # Initialize momentum matrices to act as perpendicular velocity vectors
    P1 = torch.tensor(np.diag(-np.sin(theta_init) * 0.15), dtype=torch.float32)
    P2 = torch.tensor(np.diag(np.cos(theta_init) * 0.15), dtype=torch.float32)
    
    dt = 0.04          
    damping = 0.96 # Reduced damping slightly to let orbital momentum slide further
    
    metrics_log_path = "matrix_energy_metrics.csv"
    csv_file = open(metrics_log_path, mode='w', newline='')
    log_writer = csv.writer(csv_file)
    log_writer.writerow(["Frame", "Potential_Energy_PE", "Kinetic_Energy_KE", "Total_Energy_E", "Mean_Curv_Scale", "Dominant_Frequency_Hz"])

    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#05050A')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#05050A')

    def process_fourier_step_3d(frame_idx):
        nonlocal X1, X2, P1, P2
        if frame_idx >= total_frames: return
        ax.clear()
        ax.set_facecolor('#05050A')
        ax.grid(False)
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        start, end = frame_idx * samples_per_frame, (frame_idx + 1) * samples_per_frame
        audio_chunk = audio_signal[start:end]
        if len(audio_chunk) < samples_per_frame: 
            audio_chunk = np.pad(audio_chunk, (0, samples_per_frame - len(audio_chunk)))
            
        windowed_chunk = audio_chunk * np.hanning(len(audio_chunk))
        fft_values = np.abs(np.fft.rfft(windowed_chunk))
        fft_frequencies = np.fft.rfftfreq(len(windowed_chunk), d=1.0/sample_rate)
        
        bass_mask = (fft_frequencies >= 20) & (fft_frequencies <= 150)
        mid_mask = (fft_frequencies > 150) & (fft_frequencies <= 2500)
        high_mask = (fft_frequencies > 2500) & (fft_frequencies <= 15000)
        
        max_fft = np.max(fft_values) if len(fft_values) > 0 and np.max(fft_values) > 0 else 1.0
        bass_volt = min((np.mean(fft_values[bass_mask]) / max_fft) * 5.0, 5.0) if np.any(bass_mask) else 0
        mid_volt = min((np.mean(fft_values[mid_mask]) / max_fft) * 4.0, 4.0) if np.any(mid_mask) else 0
        high_volt = min((np.mean(fft_values[high_mask]) / max_fft) * 3.0, 3.0) if np.any(high_mask) else 0
        
        dominant_freq = fft_frequencies[np.argmax(fft_values)] if len(fft_values) > 0 else 0
        
        defect_potentials = torch.zeros(matrix_size, matrix_size)
        defect_potentials[22:42, 22:42] = float(bass_volt * 1.5)
        defect_potentials[0:12, :] = float(high_volt * 0.8)
        defect_potentials[:, 50:62] = float(mid_volt * 1.2)
        
        # --- PHASE SPACE LEAPFROG update ENGINE ---
        X1_eff, X2_eff = X1 + defect_potentials, X2 + defect_potentials
        commutator = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
        
        F1 = -2.0 * (torch.mm(commutator, X2_eff) - torch.mm(X2_eff, commutator))
        F2 = 2.0 * (torch.mm(commutator, X1_eff) - torch.mm(X1_eff, commutator))
        
        F1 = torch.clamp(F1, min=-5.0, max=10.0)
        F2 = torch.clamp(F2, min=-5.0, max=10.0)
        
        P1 = P1 * damping + F1 * (dt / 2.0)
        P2 = P2 * damping + F2 * (dt / 2.0)
        X1 += P1 * dt
        X2 += P2 * dt
        
        X1_eff, X2_eff = X1 + defect_potentials, X2 + defect_potentials
        commutator_new = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
        F1_new = -2.0 * (torch.mm(commutator_new, X2_eff) - torch.mm(X2_eff, commutator_new))
        F2_new = 2.0 * (torch.mm(commutator_new, X1_eff) - torch.mm(X1_eff, commutator_new))
        
        F1_new = torch.clamp(F1_new, min=-5.0, max=10.0)
        F2_new = torch.clamp(F2_new, min=-5.0, max=10.0)
        
        P1 = P1 * damping + F1_new * (dt / 2.0)
        P2 = P2 * damping + F2_new * (dt / 2.0)
        
        pe_val = -torch.trace(torch.mm(commutator_new, commutator_new)).item()
        ke_val = (torch.trace(torch.mm(P1, P1)) + torch.trace(torch.mm(P2, P2))).item()
        total_energy = pe_val + ke_val
        
        correlation = torch.abs(torch.mm(X1_eff, X2_eff)).numpy()
        threshold = np.percentile(correlation, max(93.0 - float(mid_volt * 0.4), 88.0))
        G = nx.from_numpy_array((correlation > threshold).astype(int))
        
        curv_scale = compute_pure_python_curvature(G)
        log_writer.writerow([frame_idx, f"{pe_val:.4f}", f"{ke_val:.4f}", f"{total_energy:.4f}", f"{curv_scale:.4f}", f"{dominant_freq:.1f}"])
        
        nodes_list = list(G.nodes())
        
        # --- UNLOCKING TRUE SWIRL TRACKING ---
        # Instead of a static template layout, we extract the live diagonal elements of the active matrices.
        # This means the X and Y coordinates on your screen are flowing straight out of the physical equations!
        xs = np.diag(X1.detach().numpy())
        ys = np.diag(X2.detach().numpy())
        
        # The Z axis acts as a depth offset driven directly by the localized momentum energy loops
        zs = np.diag((P1 + P2).detach().numpy()) * 1.5 - (np.array([G.degree(n) if n in G else 0 for n in nodes_list]) * 0.005 * bass_volt)
        
        current_pos_3d = {node: (xs[i], ys[i], zs[i]) for i, node in enumerate(nodes_list)}
        
        # Draw dynamic links
        for u, v in G.edges():
            if u in current_pos_3d and v in current_pos_3d:
                ax.plot(
                    [current_pos_3d[u][0], current_pos_3d[v][0]],
                    [current_pos_3d[u][1], current_pos_3d[v][1]],
                    [current_pos_3d[u][2], current_pos_3d[v][2]],
                    color='#00FFCC', alpha=0.07, linewidth=0.5
                )
                
        node_sizes = [max(G.degree(n) * 7, 10) if n in G else 10 for n in nodes_list]
        node_colors = [G.degree(n) + float(high_volt * 6.0) if n in G else 0 for n in nodes_list]
        
        ax.scatter(xs, ys, zs, s=node_sizes, c=node_colors, cmap=plt.cm.plasma, alpha=0.85, edgecolors='#05050A', linewidths=0.2)
        
        # Rotate the viewer camera slowly over time to see the system swirling from different perspectives
        ax.view_init(elev=25, azim=frame_idx * 0.8)
        
        ax.set_title(
            f"Active Phase-Space Trajectory Engine [FFT]\n"
            f"PE (Action): {pe_val:.2f} | KE (Momentum): {ke_val:.2f} | Peak: {dominant_freq:.0f}Hz",
            color='#FFFFFF', fontsize=12, fontweight='bold', y=0.95
        )

    print("\nProcessing audio frequencies and compiling active orbital paths...")
    ani = animation.FuncAnimation(fig, process_fourier_step_3d, frames=total_frames, interval=80, repeat=False)
    output_filename = "true_fft_universe_3d.gif"
    ani.save(output_filename, writer='pillow', fps=12)
    plt.close()
    csv_file.close()
    print(f"\nSuccess! Fluid 3D phase-space trajectory compilation complete.")
    print(f" -> Live Animation Saved: .\\{output_filename}")

if __name__ == "__main__":
    run_fourier_analysis_3d()
