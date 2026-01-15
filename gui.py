import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from rabinfingerprint import RabinChunker
from gearhashing import GearChunker
from fastCDC import fastCDC

class RabinChunkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CDC Chunker Comparison - Rabin vs Gear")
        self.root.geometry("1400x900")
        
        self.rabin_chunker = None
        self.gear_chunker = None
        self.fastcdc_chunker = None
        self.rabin_chunks = []
        self.gear_chunks = []
        self.fastcdc_chunks = []
        self.rabin_stats = {}
        self.gear_stats = {}
        self.fastcdc_stats = {}
        self.rabin_sizes = []
        self.gear_sizes = []
        self.fastcdc_sizes = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ===== Parameters Section =====
        params_frame = ttk.LabelFrame(main_frame, text="Chunker Parameters", padding="10")
        params_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Min Size
        ttk.Label(params_frame, text="Min Size (bytes):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_size_var = tk.IntVar(value=2048)
        ttk.Entry(params_frame, textvariable=self.min_size_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        # Avg Size
        ttk.Label(params_frame, text="Avg Size (bytes):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.avg_size_var = tk.IntVar(value=8192)
        ttk.Entry(params_frame, textvariable=self.avg_size_var, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        # Max Size
        ttk.Label(params_frame, text="Max Size (bytes):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        self.max_size_var = tk.IntVar(value=16384)
        ttk.Entry(params_frame, textvariable=self.max_size_var, width=15).grid(row=0, column=5, padx=5, pady=5)
        
        # Window Mode (for Rabin only)
        ttk.Label(params_frame, text="Window Mode:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.window_mode_var = tk.StringVar(value="sqrt")
        window_combo = ttk.Combobox(params_frame, textvariable=self.window_mode_var, 
                                     values=["sqrt", "log"], width=12, state="readonly")
        window_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Norm Level (for FastCDC only)
        ttk.Label(params_frame, text="Norm Level:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.norm_level_var = tk.IntVar(value=3)
        ttk.Entry(params_frame, textvariable=self.norm_level_var, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        # Algorithm Selection
        ttk.Label(params_frame, text="Algorithm:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.algorithm_var = tk.StringVar(value="all")
        algo_combo = ttk.Combobox(params_frame, textvariable=self.algorithm_var, 
                                   values=["rabin", "gear", "fastcdc", "all"], width=12, state="readonly")
        algo_combo.grid(row=2, column=3, padx=5, pady=5)
        
        # Buttons
        ttk.Button(params_frame, text="Load File", command=self.load_file).grid(row=2, column=4, padx=5, pady=5)
        ttk.Button(params_frame, text="Chunk Data", command=self.chunk_data).grid(row=2, column=5, padx=5, pady=5)
        ttk.Button(params_frame, text="Clear", command=self.clear_results).grid(row=2, column=6, padx=5, pady=5)
        
        # ===== Left Panel: Text Input/Output =====
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Input Text
        ttk.Label(left_frame, text="Input Data:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(left_frame, height=10, width=50, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Statistics Output
        ttk.Label(left_frame, text="Statistics:").pack(anchor=tk.W)
        self.stats_text = scrolledtext.ScrolledText(left_frame, height=15, width=50, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== Right Panel: Visualizations =====
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Histogram
        self.hist_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.hist_frame, text="Size Distribution")
        
        # Tab 2: Chunk List
        self.chunks_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chunks_frame, text="Chunk Preview")
        
        self.chunks_text = scrolledtext.ScrolledText(self.chunks_frame, wrap=tk.WORD)
        self.chunks_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== Bottom: Chunk Details =====
        details_frame = ttk.LabelFrame(main_frame, text="Chunk Details", padding="10")
        details_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        details_frame.columnconfigure(1, weight=1)
        
        ttk.Label(details_frame, text="Chunk Index:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.chunk_index_var = tk.IntVar(value=0)
        self.chunk_spinbox = ttk.Spinbox(details_frame, from_=0, to=0, 
                                         textvariable=self.chunk_index_var, 
                                         command=self.show_chunk_detail, width=10)
        self.chunk_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Button(details_frame, text="Show Chunk", command=self.show_chunk_detail).grid(row=0, column=2, padx=5)
        
        self.chunk_detail_text = scrolledtext.ScrolledText(details_frame, height=6, wrap=tk.WORD)
        self.chunk_detail_text.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def load_file(self):
        filename = filedialog.askopenfilename(title="Select file to chunk")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = f.read()
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(1.0, data)
                messagebox.showinfo("Success", f"Loaded {len(data)} bytes from file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def chunk_data(self):
        try:
            # Get parameters
            min_size = self.min_size_var.get()
            avg_size = self.avg_size_var.get()
            max_size = self.max_size_var.get()
            window_mode = self.window_mode_var.get()
            algorithm = self.algorithm_var.get()
            
            # Validate
            if min_size >= avg_size or avg_size >= max_size:
                messagebox.showerror("Error", "Must satisfy: min_size < avg_size < max_size")
                return
            
            # Get input data
            data = self.input_text.get(1.0, tk.END).strip()
            if not data:
                messagebox.showerror("Error", "No input data provided")
                return
            
            # Run selected algorithm(s)
            if algorithm in ["rabin", "all"]:
                self.rabin_chunker = RabinChunker(min_size=min_size, avg_size=avg_size, 
                                           max_size=max_size, window_mode=window_mode)
                self.rabin_chunks = self.rabin_chunker.chunk_data(data)
                self.rabin_stats, self.rabin_sizes = self.rabin_chunker.analyze_chunks(self.rabin_chunks)
            
            if algorithm in ["gear", "all"]:
                self.gear_chunker = GearChunker(min_size=min_size, avg_size=avg_size, 
                                          max_size=max_size)
                self.gear_chunks = self.gear_chunker.chunk_data(data)
                self.gear_stats, self.gear_sizes = self.gear_chunker.analyze_chunks(self.gear_chunks)
            
            if algorithm in ["fastcdc", "all"]:
                norm_level = self.norm_level_var.get()
                self.fastcdc_chunker = fastCDC(min_size=min_size, avg_size=avg_size, 
                                          max_size=max_size, norm_level=norm_level)
                self.fastcdc_chunks = self.fastcdc_chunker.chunk_data(data)
                self.fastcdc_stats, self.fastcdc_sizes = self.fastcdc_chunker.analyze_chunks(self.fastcdc_chunks)
            
            # Update UI
            self.display_statistics()
            self.display_histogram()
            self.display_chunks()
            self.update_chunk_spinbox()
            
            msg = []
            if algorithm in ["rabin", "all"]:
                msg.append(f"Rabin: {len(self.rabin_chunks)} chunks")
            if algorithm in ["gear", "all"]:
                msg.append(f"Gear: {len(self.gear_chunks)} chunks")
            if algorithm in ["fastcdc", "all"]:
                msg.append(f"FastCDC: {len(self.fastcdc_chunks)} chunks")
            messagebox.showinfo("Success", " | ".join(msg))
            
        except Exception as e:
            messagebox.showerror("Error", f"Chunking failed: {str(e)}")
    
    def display_statistics(self):
        self.stats_text.delete(1.0, tk.END)
        
        output = ""
        
        # Display Rabin statistics if available
        if self.rabin_stats:
            output += f"""{'='*50}
RABIN FINGERPRINTING STATISTICS
{'='*50}

📊 Configuration:
   Window Size:      {self.rabin_stats['window_size']} bytes ({self.rabin_stats['window_size']*8} bits)
   Target Avg Size:  {self.rabin_stats['target_avg']:,} bytes
   Min/Max Bounds:   {self.rabin_chunker.min_size:,} / {self.rabin_chunker.max_size:,} bytes
   Mask:             0x{self.rabin_chunker.mask:X}

📈 Results:
   Total Data Size:  {self.rabin_stats['total_size']:,} bytes
   Number of Chunks: {self.rabin_stats['num_chunks']}
   Smallest Chunk:   {self.rabin_stats['min_chunk']:,} bytes
   Largest Chunk:    {self.rabin_stats['max_chunk']:,} bytes
   Average Size:     {self.rabin_stats['avg_chunk']:.2f} bytes
   Median Size:      {self.rabin_stats['median_chunk']:.2f} bytes
   Std Deviation:    {self.rabin_stats['std_dev']:.2f}
"""
            
            accuracy = (self.rabin_stats['avg_chunk'] / self.rabin_stats['target_avg'] * 100) if self.rabin_stats['target_avg'] > 0 else 0
            output += f"   Target Accuracy:  {accuracy:.1f}%\n"
            
            # Deduplication statistics
            output += f"\n💾 Deduplication:\n"
            output += f"   Total Chunks:     {self.rabin_stats['num_chunks']}\n"
            output += f"   Unique Chunks:    {self.rabin_stats['unique_chunks']}\n"
            output += f"   Duplicate Chunks: {self.rabin_stats['num_chunks'] - self.rabin_stats['unique_chunks']}\n"
            output += f"   Total Size:       {self.rabin_stats['total_size']:,} bytes\n"
            output += f"   Unique Size:      {self.rabin_stats['unique_size']:,} bytes\n"
            output += f"   Dedup Ratio:      {self.rabin_stats['dedup_rate']:.2f}x\n"
            savings = (1 - self.rabin_stats['unique_size'] / self.rabin_stats['total_size']) * 100 if self.rabin_stats['total_size'] > 0 else 0
            output += f"   Space Savings:    {savings:.1f}%\n"
            
            # Size distribution
            bins = [self.rabin_chunker.min_size, self.rabin_stats['target_avg'], self.rabin_chunker.max_size]
            hist, _ = np.histogram(self.rabin_sizes, bins=bins)
            
            output += f"\n📉 Size Distribution:\n"
            output += f"   < {self.rabin_stats['target_avg']:,} bytes:  {hist[0]:3d} chunks ({hist[0]/self.rabin_stats['num_chunks']*100:.1f}%)\n"
            output += f"   ≥ {self.rabin_stats['target_avg']:,} bytes:  {hist[1]:3d} chunks ({hist[1]/self.rabin_stats['num_chunks']*100:.1f}%)\n"
            output += f"\n{'='*50}\n\n"
        
        # Display Gear statistics if available
        if self.gear_stats:
            output += f"""{'='*50}
GEAR HASHING STATISTICS
{'='*50}

📊 Configuration:
   Window Size:      {self.gear_stats['window_size']} bytes
   Target Avg Size:  {self.gear_stats['target_avg']:,} bytes
   Min/Max Bounds:   {self.gear_chunker.min_size:,} / {self.gear_chunker.max_size:,} bytes
   Mask:             0x{self.gear_chunker.mask:X}

📈 Results:
   Total Data Size:  {self.gear_stats['total_size']:,} bytes
   Number of Chunks: {self.gear_stats['num_chunks']}
   Smallest Chunk:   {self.gear_stats['min_chunk']:,} bytes
   Largest Chunk:    {self.gear_stats['max_chunk']:,} bytes
   Average Size:     {self.gear_stats['avg_chunk']:.2f} bytes
   Median Size:      {self.gear_stats['median_chunk']:.2f} bytes
   Std Deviation:    {self.gear_stats['std_dev']:.2f}
"""
            
            accuracy = (self.gear_stats['avg_chunk'] / self.gear_stats['target_avg'] * 100) if self.gear_stats['target_avg'] > 0 else 0
            output += f"   Target Accuracy:  {accuracy:.1f}%\n"
            
            # Deduplication statistics
            output += f"\n💾 Deduplication:\n"
            output += f"   Total Chunks:     {self.gear_stats['num_chunks']}\n"
            output += f"   Unique Chunks:    {self.gear_stats['unique_chunks']}\n"
            output += f"   Duplicate Chunks: {self.gear_stats['num_chunks'] - self.gear_stats['unique_chunks']}\n"
            output += f"   Total Size:       {self.gear_stats['total_size']:,} bytes\n"
            output += f"   Unique Size:      {self.gear_stats['unique_size']:,} bytes\n"
            output += f"   Dedup Ratio:      {self.gear_stats['dedup_rate']:.2f}x\n"
            savings = (1 - self.gear_stats['unique_size'] / self.gear_stats['total_size']) * 100 if self.gear_stats['total_size'] > 0 else 0
            output += f"   Space Savings:    {savings:.1f}%\n"
            
            # Size distribution
            bins = [self.gear_chunker.min_size, self.gear_stats['target_avg'], self.gear_chunker.max_size]
            hist, _ = np.histogram(self.gear_sizes, bins=bins)
            
            output += f"\n📉 Size Distribution:\n"
            output += f"   < {self.gear_stats['target_avg']:,} bytes:  {hist[0]:3d} chunks ({hist[0]/self.gear_stats['num_chunks']*100:.1f}%)\n"
            output += f"   ≥ {self.gear_stats['target_avg']:,} bytes:  {hist[1]:3d} chunks ({hist[1]/self.gear_stats['num_chunks']*100:.1f}%)\n"
            output += f"\n{'='*50}\n\n"
        
        # Display FastCDC statistics if available
        if self.fastcdc_stats:
            output += f"""{'='*50}
FASTCDC STATISTICS
{'='*50}

📊 Configuration:
   Hash Length:      {self.fastcdc_chunker.hash_lenght} bits
   Target Avg Size:  {self.fastcdc_stats['target_avg']:,} bytes
   Min/Max Bounds:   {self.fastcdc_chunker.min_size:,} / {self.fastcdc_chunker.max_size:,} bytes
   Normalized Size:  {self.fastcdc_chunker.normalized_size:,} bytes
   Norm Level:       {self.fastcdc_chunker.norm_level}
   Large Mask Bits:  {self.fastcdc_chunker.large_mask_bits}
   Small Mask Bits:  {self.fastcdc_chunker.small_mask_bits}
   Large Mask:       0x{self.fastcdc_chunker.large_mask:X}
   Small Mask:       0x{self.fastcdc_chunker.small_mask:X}

📈 Results:
   Total Data Size:  {self.fastcdc_stats['total_size']:,} bytes
   Number of Chunks: {self.fastcdc_stats['num_chunks']}
   Smallest Chunk:   {self.fastcdc_stats['min_chunk']:,} bytes
   Largest Chunk:    {self.fastcdc_stats['max_chunk']:,} bytes
   Average Size:     {self.fastcdc_stats['avg_chunk']:.2f} bytes
   Median Size:      {self.fastcdc_stats['median_chunk']:.2f} bytes
   Std Deviation:    {self.fastcdc_stats['std_dev']:.2f}
"""
            
            accuracy = (self.fastcdc_stats['avg_chunk'] / self.fastcdc_stats['target_avg'] * 100) if self.fastcdc_stats['target_avg'] > 0 else 0
            output += f"   Target Accuracy:  {accuracy:.1f}%\n"
            
            # Deduplication statistics
            output += f"\n💾 Deduplication:\n"
            output += f"   Total Chunks:     {self.fastcdc_stats['num_chunks']}\n"
            output += f"   Unique Chunks:    {self.fastcdc_stats['unique_chunks']}\n"
            output += f"   Duplicate Chunks: {self.fastcdc_stats['num_chunks'] - self.fastcdc_stats['unique_chunks']}\n"
            output += f"   Total Size:       {self.fastcdc_stats['total_size']:,} bytes\n"
            output += f"   Unique Size:      {self.fastcdc_stats['unique_size']:,} bytes\n"
            output += f"   Dedup Ratio:      {self.fastcdc_stats['dedup_rate']:.2f}x\n"
            savings = (1 - self.fastcdc_stats['unique_size'] / self.fastcdc_stats['total_size']) * 100 if self.fastcdc_stats['total_size'] > 0 else 0
            output += f"   Space Savings:    {savings:.1f}%\n"
            
            # Size distribution
            bins = [self.fastcdc_chunker.min_size, self.fastcdc_stats['target_avg'], self.fastcdc_chunker.max_size]
            hist, _ = np.histogram(self.fastcdc_sizes, bins=bins)
            
            output += f"\n📉 Size Distribution:\n"
            output += f"   < {self.fastcdc_stats['target_avg']:,} bytes:  {hist[0]:3d} chunks ({hist[0]/self.fastcdc_stats['num_chunks']*100:.1f}%)\n"
            output += f"   ≥ {self.fastcdc_stats['target_avg']:,} bytes:  {hist[1]:3d} chunks ({hist[1]/self.fastcdc_stats['num_chunks']*100:.1f}%)\n"
            output += f"\n{'='*50}\n\n"
        
        # Display comparison if multiple algorithms were run
        num_algos = sum([bool(self.rabin_stats), bool(self.gear_stats), bool(self.fastcdc_stats)])
        
        if num_algos >= 2:
            output += f"""{'='*50}
ALGORITHM COMPARISON
{'='*50}

"""
            
            if self.rabin_stats:
                output += f"📊 Rabin Fingerprinting:\n"
                output += f"   Chunks: {self.rabin_stats['num_chunks']}, Avg: {self.rabin_stats['avg_chunk']:.2f} bytes, "
                output += f"Accuracy: {(self.rabin_stats['avg_chunk'] / self.rabin_stats['target_avg'] * 100):.1f}%\n"
            
            if self.gear_stats:
                output += f"⚙️  Gear Hashing:\n"
                output += f"   Chunks: {self.gear_stats['num_chunks']}, Avg: {self.gear_stats['avg_chunk']:.2f} bytes, "
                output += f"Accuracy: {(self.gear_stats['avg_chunk'] / self.gear_stats['target_avg'] * 100):.1f}%\n"
            
            if self.fastcdc_stats:
                output += f"⚡ FastCDC:\n"
                output += f"   Chunks: {self.fastcdc_stats['num_chunks']}, Avg: {self.fastcdc_stats['avg_chunk']:.2f} bytes, "
                output += f"Accuracy: {(self.fastcdc_stats['avg_chunk'] / self.fastcdc_stats['target_avg'] * 100):.1f}%\n"
            
            output += f"\n📐 Standard Deviation:\n"
            if self.rabin_stats:
                output += f"   Rabin:   {self.rabin_stats['std_dev']:.2f}\n"
            if self.gear_stats:
                output += f"   Gear:    {self.gear_stats['std_dev']:.2f}\n"
            if self.fastcdc_stats:
                output += f"   FastCDC: {self.fastcdc_stats['std_dev']:.2f}\n"
            
            output += f"\n{'='*50}"""
        
        self.stats_text.insert(1.0, output)
    
    def display_histogram(self):
        # Clear previous plot
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        
        # Create new plot
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot histogram(s) based on what data is available
        num_algos = sum([bool(self.rabin_sizes), bool(self.gear_sizes), bool(self.fastcdc_sizes)])
        
        if num_algos > 1:
            # Multiple algorithms - overlapping histograms
            if self.rabin_sizes:
                ax.hist(self.rabin_sizes, bins=20, color='skyblue', edgecolor='blue', 
                       alpha=0.5, label='Rabin')
                ax.axvline(self.rabin_stats['avg_chunk'], color='blue', linestyle='--', 
                          linewidth=2, label=f'Rabin Avg: {self.rabin_stats["avg_chunk"]:.0f}')
            
            if self.gear_sizes:
                ax.hist(self.gear_sizes, bins=20, color='lightcoral', edgecolor='red', 
                       alpha=0.5, label='Gear')
                ax.axvline(self.gear_stats['avg_chunk'], color='red', linestyle='--', 
                          linewidth=2, label=f'Gear Avg: {self.gear_stats["avg_chunk"]:.0f}')
            
            if self.fastcdc_sizes:
                ax.hist(self.fastcdc_sizes, bins=20, color='lightgreen', edgecolor='green', 
                       alpha=0.5, label='FastCDC')
                ax.axvline(self.fastcdc_stats['avg_chunk'], color='green', linestyle='--', 
                          linewidth=2, label=f'FastCDC Avg: {self.fastcdc_stats["avg_chunk"]:.0f}')
            
            # Show target line
            target = self.rabin_stats.get('target_avg') if self.rabin_stats else (self.gear_stats.get('target_avg') if self.gear_stats else self.fastcdc_stats.get('target_avg'))
            if target:
                ax.axvline(target, color='black', linestyle=':', 
                          linewidth=2, label=f'Target: {target}')
        elif self.rabin_sizes:
            # Only Rabin
            ax.hist(self.rabin_sizes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(self.rabin_stats['avg_chunk'], color='red', linestyle='--', 
                      linewidth=2, label=f'Actual Avg: {self.rabin_stats["avg_chunk"]:.0f}')
            ax.axvline(self.rabin_stats['target_avg'], color='green', linestyle='--', 
                      linewidth=2, label=f'Target Avg: {self.rabin_stats["target_avg"]}')
        elif self.gear_sizes:
            # Only Gear
            ax.hist(self.gear_sizes, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
            ax.axvline(self.gear_stats['avg_chunk'], color='red', linestyle='--', 
                      linewidth=2, label=f'Actual Avg: {self.gear_stats["avg_chunk"]:.0f}')
            ax.axvline(self.gear_stats['target_avg'], color='green', linestyle='--', 
                      linewidth=2, label=f'Target Avg: {self.gear_stats["target_avg"]}')
        elif self.fastcdc_sizes:
            # Only FastCDC
            ax.hist(self.fastcdc_sizes, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            ax.axvline(self.fastcdc_stats['avg_chunk'], color='red', linestyle='--', 
                      linewidth=2, label=f'Actual Avg: {self.fastcdc_stats["avg_chunk"]:.0f}')
            ax.axvline(self.fastcdc_stats['target_avg'], color='green', linestyle='--', 
                      linewidth=2, label=f'Target Avg: {self.fastcdc_stats["target_avg"]}')
        
        ax.set_xlabel('Chunk Size (bytes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Chunk Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_chunks(self):
        self.chunks_text.delete(1.0, tk.END)
        
        # Use whichever chunks are available (prefer rabin, then gear, then fastcdc)
        if self.rabin_chunks:
            chunks = self.rabin_chunks
            algo = "Rabin"
        elif self.gear_chunks:
            chunks = self.gear_chunks
            algo = "Gear"
        elif self.fastcdc_chunks:
            chunks = self.fastcdc_chunks
            algo = "FastCDC"
        else:
            return
        
        output = f"📦 Chunk Preview - {algo} ({len(chunks)} total chunks):\n"
        output += "-" * 70 + "\n\n"
        
        max_display = min(20, len(chunks))
        for i, chunk in enumerate(chunks[:max_display]):
            try:
                preview = chunk.decode('utf-8', errors='replace')
                if len(preview) > 60:
                    preview = preview[:60] + "..."
                preview = preview.replace('\n', '\\n')
            except:
                preview = f"<binary data: {len(chunk)} bytes>"
            
            output += f"Chunk {i:3d} | Size: {len(chunk):5d} bytes\n"
            output += f"         | {preview}\n\n"
        
        if len(chunks) > max_display:
            output += f"... and {len(chunks) - max_display} more chunks\n"
        
        self.chunks_text.insert(1.0, output)
    
    def update_chunk_spinbox(self):
        if self.rabin_chunks:
            self.chunk_spinbox.config(to=len(self.rabin_chunks)-1)
        elif self.gear_chunks:
            self.chunk_spinbox.config(to=len(self.gear_chunks)-1)
        elif self.fastcdc_chunks:
            self.chunk_spinbox.config(to=len(self.fastcdc_chunks)-1)
    
    def show_chunk_detail(self):
        if not self.rabin_chunks and not self.gear_chunks and not self.fastcdc_chunks:
            return
        
        idx = self.chunk_index_var.get()
        chunks = self.rabin_chunks if self.rabin_chunks else (self.gear_chunks if self.gear_chunks else self.fastcdc_chunks)
        
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            try:
                content = chunk.decode('utf-8', errors='replace')
            except:
                content = f"<binary data>"
            
            self.chunk_detail_text.delete(1.0, tk.END)
            detail = f"Chunk {idx} Details:\n"
            detail += f"Size: {len(chunk)} bytes\n"
            detail += f"Content:\n{'-'*70}\n{content}\n"
            self.chunk_detail_text.insert(1.0, detail)
    
    def clear_results(self):
        # Clear text displays
        self.stats_text.delete(1.0, tk.END)
        self.chunks_text.delete(1.0, tk.END)
        self.chunk_detail_text.delete(1.0, tk.END)
        
        # Clear histogram
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        
        # Reset all chunker objects and data
        self.rabin_chunker = None
        self.gear_chunker = None
        self.fastcdc_chunker = None
        
        # Clear all chunks
        self.rabin_chunks = []
        self.gear_chunks = []
        self.fastcdc_chunks = []
        
        # Clear all statistics
        self.rabin_stats = {}
        self.gear_stats = {}
        self.fastcdc_stats = {}
        
        # Clear all sizes
        self.rabin_sizes = []
        self.gear_sizes = []
        self.fastcdc_sizes = []
        
        # Reset chunk spinbox
        self.chunk_spinbox.config(to=0)
        self.chunk_index_var.set(0)

def main():
    root = tk.Tk()
    app = RabinChunkerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()