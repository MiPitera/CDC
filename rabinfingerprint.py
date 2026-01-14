import numpy as np
import galois as g
import logging 

class RabinFingerprint: 
    def __init__(self, window_size=64):
        self.degree = window_size
        self.poly = g.irreducible_poly(2,self.degree,None,"random")
        self.poly_field = g.GF(2**self.degree, irreducible_poly=self.poly)
        logging.info(f"Generated random irreducible polynomial: {self.poly}")
        self.fingerprint = self.poly_field(0)   
        

    def compute_fingerprint(self, data: str):    
        data = data.encode('utf-8') 
        fingerprint = self.poly_field(0)
        for byte in data:
            fingerprint = fingerprint* self.poly_field(256) # shift byte 
            fingerprint = fingerprint + self.poly_field(byte)
        self.fingerprint = fingerprint                  
    
        return (self.fingerprint)

    def fingerprint_expand(self, new_byte: int):
        self.fingerprint = self.fingerprint * self.poly_field(256) + self.poly_field(new_byte)
        return (self.fingerprint)    
    
    def fingerprint_roll(self, old_byte: int, new_byte: int):
        self.fingerprint = self.fingerprint - self.poly_field(old_byte )* self.poly_field((2 ** (self.degree -8)))
        self.fingerprint = self.fingerprint * self.poly_field(256)
        self.fingerprint = self.fingerprint + self.poly_field(new_byte)    
        return (self.fingerprint)    
    
    def reset_fingerprint(self):
        self.fingerprint = self.poly_field(0)
        return (self.fingerprint)



class RabinChunker:
    def __init__(self, min_size=2048, avg_size=8192, max_size=16384, window_mode='sqrt'):
        '''
        input in bytes
        modes log,sqrt

        '''
        if window_mode == 'log':
            self.window_size = (avg_size-1).bit_length()
        else : 
            self.window_size = int(np.sqrt(avg_size))        

        self.rabin = RabinFingerprint(window_size=self.window_size*8)
        self.min_size = min_size
        self.avg_size = avg_size
        self.max_size = max_size
        self.mask = (1 << ((avg_size-1).bit_length())) - 1

    def chunk_data(self, data: str):
        data = data.encode('utf-8')
        chunks = []
        start = 0
        self.rabin.reset_fingerprint()

        for i in range(len(data)):
            byte = data[i]
            self.rabin.fingerprint_expand(byte)

            if (i - start >= self.min_size) and ((self.rabin.fingerprint & self.mask) == 0 or (i - start) >= self.max_size):
                chunks.append(data[start:i+1])
                start = i + 1
                self.rabin.reset_fingerprint()

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
            'window_size': self.window_size,
        }
        
        return stats, sizes

