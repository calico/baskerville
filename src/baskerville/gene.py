# Copyright 2022 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from collections import defaultdict
import gzip
from intervaltree import IntervalTree
import numpy as np
import pybedtools


class Gene:
    """Class for managing genes in an isoform-agnostic way, taking
    the union of exons across isoforms."""

    def __init__(self, chrom, strand, kv, name=None):
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.name = name
        self.exons = IntervalTree()

    def add_exon(self, start, end):
        """BED 0-indexing assumed."""
        self.exons[start:end] = True

    def get_exons(self):
        """Return a sorted list of exons, merging overlapping intervals."""
        self.exons.merge_overlaps()
        return sorted(self.exons)

    def midpoint(self):
        """Return the midpoint of the gene based on exon positions."""
        positions = []
        for exon in self.get_exons():
            positions += range(exon.begin, exon.end)
        midp = int(np.mean(positions))
        return midp

    def span(self):
        """Return the span of the gene based on exon positions."""
        exon_starts = [exon.begin for exon in self.exons]
        exon_ends = [exon.end for exon in self.exons]
        return min(exon_starts), max(exon_ends)

    def output_slice(
        self,
        seq_start,
        seq_len,
        model_stride,
        span=False,
        majority_overlap=False,
    ):
        """Return sorted, unique output bins for the gene span or exon union.

        Spans always round boundaries to the nearest bin (ties to even).
        Exons use that rule only with majority_overlap=True; otherwise any
        overlap counts. Bins are clipped to the prediction window.
        """
        intervals = (
            [self.span()] if span else ((e.begin, e.end) for e in self.get_exons())
        )
        gene_slice = []
        for start, end in intervals:
            slice_start, slice_end = interval_output_slice(
                start,
                end,
                seq_start,
                seq_len,
                model_stride,
                majority_overlap=span or majority_overlap,
            )
            gene_slice.extend(range(slice_start, slice_end))
        return np.unique(gene_slice)

    def coverage_fraction(self, seq_start, seq_len, model_stride, span=False):
        """Fraction of gene length covered by the sequence window."""
        gene_slice = self.output_slice(seq_start, seq_len, model_stride, span=span)
        if len(gene_slice) == 0:
            return 0.0
        covered_length = len(gene_slice) * model_stride
        if span:
            gene_start, gene_end = self.span()
            total_length = gene_end - gene_start
        else:
            total_length = sum(e.end - e.begin for e in self.get_exons())
        return covered_length / total_length


class Transcriptome:
    """Class for managing a transcriptome, which is a collection of genes
    and their exons, read from a GTF file."""

    def __init__(self, gtf_file, keep_readthrough=True):
        self.genes = {}
        self.keep_readthrough = keep_readthrough
        self.read_gtf(gtf_file)

    def read_gtf(self, gtf_file):
        """Read a GTF file and populate the transcriptome with genes and exons."""
        if gtf_file[-3:] == ".gz":
            gtf_in = gzip.open(gtf_file, "rt")
        else:
            gtf_in = open(gtf_file)

        # ignore header
        line = gtf_in.readline()
        while line[0] == "#":
            line = gtf_in.readline()

        while line:
            a = line.split("\t")
            if a[2] == "exon":
                chrom = a[0]
                start = int(a[3])
                end = int(a[4])
                strand = a[6]
                kv = gtf_kv(a[8])
                gene_id = kv["gene_id"]
                gene_name = None
                if "gene_name" in kv:
                    gene_name = kv["gene_name"]

                if gene_id not in self.genes:
                    self.genes[gene_id] = Gene(chrom, strand, kv, gene_name)

                if self.keep_readthrough:
                    self.genes[gene_id].add_exon(start - 1, end)
                else:
                    if "readthrough_transcript" not in kv.get("tag", []):
                        self.genes[gene_id].add_exon(start - 1, end)

            line = gtf_in.readline()

        # remove genes without any exons added
        self.genes = {
            gene_id: gene for gene_id, gene in self.genes.items() if len(gene.exons) > 0
        }

        gtf_in.close()

    def gene_trees(self):
        """Build chromosome-indexed interval trees of Gene objects."""
        trees = defaultdict(IntervalTree)
        for gene in self.genes.values():
            gene_start, gene_end = gene.span()
            if gene_end > gene_start:
                trees[gene.chrom][gene_start:gene_end] = gene
        return trees

    def bedtool_exon(self):
        """Return a pybedtools.BedTool object containing all exons."""
        bed_lines = []
        for gene_id, gene in self.genes.items():
            for exon in gene.get_exons():
                exon_line = "%s %d %d %s . %s" % (
                    gene.chrom,
                    exon.begin,
                    exon.end,
                    gene_id,
                    gene.strand,
                )
                bed_lines.append(exon_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt

    def bedtool_span(self):
        """Return a pybedtools.BedTool object containing the span of each gene."""
        bed_lines = []
        for gene_id, gene in self.genes.items():
            gene_start, gene_end = gene.span()
            span_line = "%s %d %d %s . %s" % (
                gene.chrom,
                gene_start,
                gene_end,
                gene_id,
                gene.strand,
            )
            bed_lines.append(span_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt


################################################################################
# Methods
################################################################################
def gtf_kv(s):
    """Convert the last gtf section of key/value pairs into a dict."""
    d = {}

    a = s.split(";")
    for key_val in a:
        if key_val.strip():
            eq_i = key_val.find("=")
            if eq_i != -1 and key_val[eq_i - 1] != '"':
                kvs = key_val.split("=")
            else:
                kvs = key_val.split()

            key = kvs[0]
            if kvs[1][0] == '"' and kvs[-1][-1] == '"':
                val = (" ".join(kvs[1:]))[1:-1].strip()
            else:
                val = (" ".join(kvs[1:])).strip()

            if key in ["tag"]:  # handle multi-value keys, store values as lists
                if key in d:
                    d[key].append(val)
                else:
                    d[key] = [val]
            else:
                d[key] = val
    return d


def interval_output_slice(
    start, end, seq_start, seq_len, model_stride, majority_overlap=True
):
    """Map a 0-based, half-open genomic interval to clipped output-bin bounds.

    Round to the nearest boundary (ties to even), or use floor/ceil for any
    overlap when majority_overlap=False. seq_start is the origin of bin 0.
    """
    start = max(0, start - seq_start) / model_stride
    end = max(0, end - seq_start) / model_stride
    if majority_overlap:
        start, end = int(np.round(start)), int(np.round(end))
    else:
        start, end = int(np.floor(start)), int(np.ceil(end))
    bin_max = int(seq_len / model_stride)
    return max(0, min(start, bin_max)), max(0, min(end, bin_max))


def find_overlapping_genes(gene_trees, chrom, start, end):
    """Find genes overlapping a genomic interval.

    Args:
        gene_trees (dict): Chromosome -> IntervalTree mapping.
        chrom (str): Chromosome.
        start (int): Interval start.
        end (int): Interval end.

    Returns:
        list[Gene]: Overlapping Gene objects.
    """
    if chrom not in gene_trees:
        return []
    return [iv.data for iv in gene_trees[chrom][start:end]]
