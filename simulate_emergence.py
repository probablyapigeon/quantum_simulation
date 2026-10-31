import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave

def generate_synthetic_audio(duration=6, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    kick = np.sin(2 * np.pi * 50 * t) * np.exp(-10 * (t % 0.5))
    synth = np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 0.5 * t)
    audio_signal = kick + 0.4 * synth
    return audio_signal, sample_rate

def load_native_wav(path):
    """Parses uncompressed 16-bit mono/stereo WAV files using native wave module."""
    with wave.open(path, 'rb') as w:
        sample_rate = w.getframerate()
        n_channels = w.getnchannels()
        n_frames = w.getnframes()
        frames = w.readframes(n_frames)
        # Convert raw binary bytes into a numpy array
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        # Reshape and downmix if stereo channel setup exists
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
            data = data[:, 0]
        # Normalize amplitude limits
        audio_signal = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
    return audio_signal, sample_rate

def run_audio_reactive_simulation(matrix_size=64, total_frames=80, audio_path=None):
    print(f"=== Starting Audio-Reactive Spacetime Engine (Substrate: {matrix_size}x{matrix_size}) ===")
    
    X1 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    X2 = torch.randn(matrix_size, matrix_size, requires_grad=True)
    optimizer = torch.optim.Adam([X1, X2], lr=0.01)
    
    if audio_path is None:
        print("No input WAV path specified. Synthesizing internal algorithmic audio track...")
        audio_signal, sample_rate = generate_synthetic_audio()
    else:
        print(f"Loading custom music file via native wave module: {audio_path}")
        audio_signal, sample_rate = load_native_wav(audio_path)
        
    samples_per_frame = int(len(audio_signal) / total_frames)
    
    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#05050A')
    ax = plt.gca()
    
    print("Pre-calculating static metric space pathways...")
    corr_init = torch.abs(torch.mm(X1.detach(), X2.detach())).numpy()
    G_template = nx.from_numpy_array((corr_init > np.percentile(corr_init, 90)).astype(int))
    fixed_positions = nx.fruchterman_reingold_layout(G_template, k=2.0/np.sqrt(matrix_size), iterations=100, seed=42)

    def update_audio_frame(frame_idx):
        ax.clear()
        ax.set_facecolor('#05050A')
        plt.axis('off')
        
        start_sample = frame_idx * samples_per_frame
        end_sample = start_sample + samples_per_frame
        audio_chunk = audio_signal[start_sample:end_sample]
        
        if len(audio_chunk) == 0: audio_chunk = np.zeros(samples_per_frame)
        
        fft_data = np.abs(np.fft.rfft(audio_chunk))
        num_bins = len(fft_data)
        
        bass = np.mean(fft_data[:int(num_bins*0.15)]) if num_bins > 0 else 0
        mids = np.mean(fft_data[int(num_bins*0.15):int(num_bins*0.6)]) if num_bins > 0 else 0
        highs = np.mean(fft_data[int(num_bins*0.6):]) if num_bins > 0 else 0
        
        # Guard scale factors against zero audio track bounds
        max_val = np.max(fft_data) if num_bins > 0 and np.max(fft_data) > 0 else 1.0
        bass_volt = min(max((bass / max_val) * 4.0, 0.0), 5.0) if num_bins > 0 else 0
        mids_volt = min(max((mids / max_val) * 3.0, 0.0), 4.0) if num_bins > 0 else 0
        highs_volt = min(max((highs / max_val) * 3.0, 0.0), 3.0) if num_bins > 0 else 0
        
        audio_defect_mask = torch.zeros(matrix_size, matrix_size)
        audio_defect_mask[22:42, 22:42] = float(bass_volt * 3.0)
        audio_defect_mask[0:10, :] = float(highs_volt * 1.5)
        audio_defect_mask[:, 54:64] = float(mids_volt * 2.0)
        
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
            
        dynamic_percentile = 94.0 - float(mids_volt * 0.5)
        threshold = np.percentile(correlation, max(min(dynamic_percentile, 95.0), 88.0))
        adj_matrix = (correlation > threshold).astype(int)
        G = nx.from_numpy_array(adj_matrix)
        
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            deg = G.degree(node) if node in G else 0
            node_sizes.append(max(deg * 6, 8))
            node_colors.append(deg + float(highs_volt * 4.0))
            
        nx.draw_networkx_edges(G, fixed_positions, alpha=0.10, edge_color='#00FFCC', width=0.7, ax=ax)
        
        if len(G.nodes) > 0:
            nx.draw_networkx_nodes(
                G, fixed_positions, node_size=node_sizes, node_color=node_colors,
                cmap=plt.cm.plasma, alpha=0.85, linewidths=0.3, edgecolors='#05050A', ax=ax
            )
        
        ax.set_title(
            f"Audio-Reactive Spacetime Pipeline\n"
            f"Bass: {bass_volt:.2f}V | Treble: {highs_volt:.2f}V | Action S: {action.item():.2f}",
            color='#FFFFFF', fontsize=12, fontweight='bold', pad=10
        )

    print("\nProcessing audio matrix updates & encoding timeline video...")
    ani = animation.FuncAnimation(fig, update_audio_frame, frames=total_frames, interval=80)
    
    output_filename = "audio_spacetime_reactive.gif"
    ani.save(output_filename, writer='pillow', fps=12)
    plt.close()
    
    print(f"\nSuccess! Native audio-reactive cosmology animation saved to: .\\{output_filename}")

if __name__ == "__main__":
    run_audio_reactive_simulation()
