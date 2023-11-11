import numpy as np
from basenji import dna_io
import pysam
import pyBigWig



def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = 'N'*(-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    
    #Extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += 'N'*(seq_len-len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot

#Helper function to get (padded) one-hot
def process_sequence(fasta_file, chrom, start, end, seq_len=524288) :

    fasta_open = pysam.Fastafile(fasta_file)
    seq_len_actual = end - start

    #Pad sequence to input window size
    start -= (seq_len - seq_len_actual) // 2
    end += (seq_len - seq_len_actual) // 2

    #Get one-hot
    sequence_one_hot = make_seq_1hot(fasta_open, chrom, start, end, seq_len)
    
    return sequence_one_hot.astype('float32')

def compute_cov(seqnn_model, chr, start, end):
    seq_len = seqnn_model.model.layers[0].input.shape[1]
    seq1hot = process_sequence('/home/yuanh/programs/genomes/hg38/hg38.fa', chr, start, end, seq_len=seq_len) 
    out = seqnn_model.model(seq1hot[None, ])
    return out.numpy()

def write_bw(bw_file, chr, start, end, values, span=32):
    bw_out = pyBigWig.open(bw_file, 'w')
    header = []
    header.append((chr, end+1))
    bw_out.addHeader(header)
    bw_out.addEntries(chr, start, values=values, span=span, step=span)
    bw_out.close()

def transform(seq_cov, clip=384, clip_soft=320, scale=0.3):
    seq_cov = scale * seq_cov # scale
    seq_cov = -1 + np.sqrt(1+seq_cov) # variant stabilize
    clip_mask = (seq_cov > clip_soft) # soft clip
    seq_cov[clip_mask] = clip_soft-1 + np.sqrt(seq_cov[clip_mask] - clip_soft+1)
    seq_cov = np.clip(seq_cov, -clip, clip) # hard clip
    return seq_cov

def untransform(cov, scale=0.3, clip_soft=320, pool_width=32):

    # undo clip_soft
    cov_unclipped = (cov - clip_soft + 1)**2 + clip_soft - 1
    unclip_mask = (cov > clip_soft)
    cov[unclip_mask] = cov_unclipped[unclip_mask]

    # undo sqrt
    cov = (cov +1)**2 - 1

    # undo scale
    cov = cov / scale

    # undo sum
    cov = cov / pool_width
    
    return cov


