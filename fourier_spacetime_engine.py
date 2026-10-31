import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import miniaudio
import os
import csv

def compute_pure_python_curvature(G):
    edges_to_sample = list(G.edges())
    if len(edges_to_sample) > 300:
        np.random.seed(42)
        idx = np.random.choice(len(edges_to_sample), 300, replace=False)
        edges_to_sample = [edges_to_sample[i] for i in idx]
    curvatures = []
    for u, v in edges_to_sample:
        deg_u, deg_v = G.degree(u), G.degree(v)
        shared = len(list(nx.common_neighbors(G, u, v)))
        curvatures.append(4 - deg_u - deg_v + (3 * shared))
    return np.mean(curvatures) if curvatures else 0

def run_fourier_analysis(matrix_size=64, target_bpm=120, audio_path=r"C:\Users\kiers\Downloads\He Can Try (Demo_Mix).mp3"):
    print(f"=== Initializing Production MP3 FFT Spacetime Engine ===")
    
    if not os.path.exists(audio_path):
        print(f"[!] Error: Target file not found at: {audio_path}")
        return

    # 1. TRUE MP3 DECODING (Corrected API Signature)
    print("Decoding native MP3 parameters into floating-point audio data arrays...")
    sound = miniaudio.decode_file(audio_path)
    sample_rate = sound.sample_rate
    
    # Convert decoded audio samples directly into a float32 numpy array
    data = np.array(sound.samples, dtype=np.float32)
    
    # Downmix stereo to mono channel tracking if multiple channels exist
    if sound.nchannels > 1:
        data = data.reshape(-1, sound.nchannels)
        data = data[:, 0]
        
    audio_signal = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
    
    frames_per_second = 15
    total_frames = int((len(audio_signal) / sample_rate) * frames_per_second)
    samples_per_frame = int(sample_rate / frames_per_second)
    
    print(f"Track Analytics | Length: {len(audio_signal)/sample_rate:.1f}s | Channels: {sound.nchannels} | Total Animation Frames: {total_frames}")

    X1 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    X2 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    optimizer = torch.optim.Adam([X1, X2], lr=0.01)
    
    metrics_log_path = "matrix_energy_metrics.csv"
    csv_file = open(metrics_log_path, mode='w', newline='')
    log_writer = csv.writer(csv_file)
    log_writer.writerow(["Frame", "Matrix_Action_S", "Mean_Curv_Scale", "Graph_Edges", "Dominant_Frequency_Hz"])

    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#05050A')
    ax = plt.gca()
    
    corr_init = torch.abs(torch.mm(X1.detach(), X2.detach())).numpy()
    G_template = nx.from_numpy_array((corr_init > np.percentile(corr_init, 90)).astype(int))
    fixed_positions = nx.fruchterman_reingold_layout(G_template, k=2.0/np.sqrt(matrix_size), iterations=100, seed=42)

    def process_fourier_step(frame_idx):
        if frame_idx >= total_frames: return
        ax.clear()
        ax.set_facecolor('#05050A')
        plt.axis('off')
        
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
        
        bass_energy = np.mean(fft_values[bass_mask]) if np.any(bass_mask) else 0
        mid_energy = np.mean(fft_values[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(fft_values[high_mask]) if np.any(high_mask) else 0
        
        max_val = np.max(fft_values) if len(fft_values) > 0 and np.max(fft_values) > 0 else 1.0
        b_volt = min((bass_energy / max_val) * 5.0, 5.0)
        m_volt = min((mid_energy / max_val) * 4.0, 4.0)
        h_volt = min((high_energy / max_val) * 3.0, 3.0)
        
        dominant_freq = fft_frequencies[np.argmax(fft_values)] if len(fft_values) > 0 else 0
        
        defect_potentials = torch.zeros(matrix_size, matrix_size)
        defect_potentials[22:42, 22:42] = float(b_volt * 4.0)
        defect_potentials[0:12, :] = float(h_volt * 2.0)
        defect_potentials[:, 50:62] = float(m_volt * 2.5)
        
        optimizer.zero_grad()
        X1_eff, X2_eff = X1 + defect_potentials, X2 + defect_potentials
        commutator = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
        action_S = torch.trace(torch.mm(commutator, commutator))
        action_S.backward()
        optimizer.step()
        
        with torch.no_grad():
            correlation = torch.abs(torch.mm(X1 + defect_potentials, X2 + defect_potentials)).numpy()
        
        threshold = np.percentile(correlation, max(94.0 - float(m_volt * 0.4), 89.0))
        G = nx.from_numpy_array((correlation > threshold).astype(int))
        
        curv_scale = compute_pure_python_curvature(G)
        log_writer.writerow([frame_idx, f"{action_S.item():.2f}", f"{curv_scale:.4f}", len(G.edges), f"{dominant_freq:.1f}"])
        
        node_sizes = [max(G.degree(n) * 6, 8) if n in G else 8 for n in G.nodes()]
        node_colors = [G.degree(n) + float(h_volt * 5.0) if n in G else 0 for n in G.nodes()]
        
        nx.draw_networkx_edges(G, fixed_positions, alpha=0.09, edge_color='#00FFCC', width=0.7, ax=ax)
        if len(G.nodes) > 0:
            nx.draw_networkx_nodes(G, fixed_positions, node_size=node_sizes, node_color=node_colors,
                                   cmap=plt.cm.plasma, alpha=0.85, linewidths=0.2, edgecolors='#05050A', ax=ax)
            
        ax.set_title(
            f"True MP3 FFT Analytic Cosmology Flow\n"
            f"Active Frequency: {dominant_freq:.1f} Hz | System Metric S: {action_S.item():.1f}",
            color='#FFFFFF', fontsize=12, fontweight='bold', pad=10
        )

    print("\nDecomposing MP3 audio spectrum and compiling layout animations...")
    ani = animation.FuncAnimation(fig, process_fourier_step, frames=total_frames, interval=66, repeat=False)
    ani.save("true_fft_universe.gif", writer='pillow', fps=15)
    plt.close()
    csv_file.close()
    print(f"\nSuccess! Audio analytics run complete.")
    print(f" -> High-Fidelity Animation: .\\true_fft_universe.gif")
    print(f" -> Telemetry Metrics Log: .\\{metrics_log_path}")

if __name__ == "__main__":
    run_fourier_analysis()
