import secrets as s
import numpy as np

class fastCDC:

    def __init__(self, min_size=2048, avg_size=8192, max_size=16384, norm_level=3):


        self.hash_lenght = 64
        self.min_size = min_size
        self.avg_size = avg_size
        self.max_size = max_size
        self.norm_level = norm_level
        # Calculate normalized chunk size (skip region)
        self.normalized_size = min_size + (avg_size - min_size) // 4
        
        # Calculate mask sizes
        avg_bits = (avg_size - 1).bit_length()
        self.large_mask_bits = avg_bits + norm_level
        self.small_mask_bits = avg_bits - norm_level

        self.large_mask = self.create_mask(self.large_mask_bits)
        self.small_mask = self.create_mask(self.small_mask_bits)
        
        self.gear_table = self.initialize_gear_table()
        self.hash = 0

    def initialize_gear_table(self):
        return [s.randbits(self.hash_lenght) for _ in range(256)] 
    

    def create_mask(self, length):
        #padded mask 
        
        pattern_length = 64 // length    

        pattern = 1<<(pattern_length -1)

        mask = 0 
        for i in range(length):
            mask |= pattern << (i*pattern_length) 

        mask = mask & 0xFFFFFFFFFFFFFFFF  # Ensure mask is within 64 bits     
        return mask

    def reset_hash(self):
        self.hash = 0
        return self.hash    
    
    def compute_hash(self, data):
        data = data.encode('utf-8')     
        for byte in data:
            self.hash = ((self.hash << 1) + self.gear_table[byte]) & ((1 << self.window_size) - 1)
        return self.hash
    def reset_hash(self):
        self.hash = 0
        return self.hash
    
    def hash_expand(self, new_byte: int):
        self.hash = ((self.hash << 1) + self.gear_table[new_byte]) & ((1 << self.hash_lenght) - 1)
        return self.hash  
    


    def chunk_data(self, data: str):
        data = data.encode('utf-8')
        chunks = []
        start = 0
        position = 0
        self.reset_hash()
        
        for i in range(len(data)):
            position = i - start
            byte = data[i]
            
            # Region 1: Skip region (no boundary detection)
            if position < self.normalized_size:
                self.hash_expand(byte)
                continue
            
            # Update hash for boundary detection
            self.hash_expand(byte)
            # Region 2: Normal chunking with large mask
            if position < self.avg_size:
                if (self.hash & self.large_mask) == self.large_mask:
                    # Cut point found - hash matches pattern
                    chunks.append(data[start:i+1])
                    start = i + 1
                    position = 0
                    self.reset_hash()
            # Region 3: Emergency cut with small mask
            else:
                if (self.hash & self.small_mask) == self.small_mask or position >= self.max_size:
                    # Emergency cut or max size reached
                    chunks.append(data[start:i+1])
                    start = i + 1
                    position = 0
                    self.reset_hash()
        
        # Add remaining data
        if start < len(data):
            chunks.append(data[start:])
        
        return chunks
    
    def analyze_chunks(self, chunks):
        """Calculate statistics for chunked data."""
        sizes = [len(chunk) for chunk in chunks]
        
        stats = {
            'num_chunks': len(chunks),
            'total_size': sum(sizes),
            'min_chunk': min(sizes) if sizes else 0,
            'max_chunk': max(sizes) if sizes else 0,
            'avg_chunk': np.mean(sizes) if sizes else 0,
            'median_chunk': np.median(sizes) if sizes else 0,
            'std_dev': np.std(sizes) if sizes else 0,
            'target_avg': self.avg_size,
        }
        
        return stats, sizes

fs=fastCDC ()