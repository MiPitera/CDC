import secrets as s
import numpy as np
class GearHashing:
    def __init__(self, window_size=64):
        self.window_size = window_size
        self.gear_table = self._initialize_gear_table()
        self.hash = 0

    def _initialize_gear_table(self):
        return [s.randbits(self.window_size-1) for _ in range(256)] 

    def compute_hash(self, data):
        data = data.encode('utf-8')     
        for byte in data:
            self.hash = ((self.hash << 1) + self.gear_table[byte]) &  ((1 << ((self.window_size-1).bit_length())) - 1)
        return self.hash
    def reset_hash(self):
        self.hash = 0
        return self.hash
    
    def hash_expand(self, new_byte: int):
        self.hash = ((self.hash << 1) + self.gear_table[new_byte]) &  ((1 << ((self.window_size-1).bit_length())) - 1)
        return self.hash        
    


class GearChunker:
    def __init__(self, min_size=2048, avg_size=8192, max_size=16384,window_mode='sqrt'):
        
        if window_mode == 'log':
            self.window_size = (avg_size-1).bit_length()
        else : 
            self.window_size = int(np.sqrt(avg_size))   
        self.gear = GearHashing(window_size=self.window_size*8)
        self.min_size = min_size
        self.avg_size = avg_size
        self.max_size = max_size
        self.mask = (1 << ((avg_size-1).bit_length())) - 1

    def chunk_data(self, data: str):
        data = data.encode('utf-8')
        chunks = []
        start = 0
        self.gear.reset_hash()

        for i in range(len(data)):
            byte = data[i]
            self.gear.hash_expand(byte)

            if (i - start >= self.min_size) and ((self.gear.hash & self.mask) == 0 or (i - start) >= self.max_size):
                chunks.append(data[start:i+1])
                start = i + 1
                self.gear.reset_hash()

        if start < len(data):
            chunks.append(data[start:])

        return chunks
    
    def analyze_chunks(self, chunks):
        """Calculate statistics for chunked data."""
        import numpy as np
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
            'window_size': self.window_size,
        }
        
        return stats, sizes
