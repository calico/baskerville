# Copyright 2017 Calico Life Sciences LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import gzip
import os
import re
import sys

import numpy as np
import pysam

from baskerville import dna

"""
vcf.py

Methods and classes to support .vcf SNP analysis.
"""


def cap_allele(allele, cap=5):
    """Cap the length of an allele in the figures."""
    if len(allele) > cap:
        allele = allele[:cap] + "*"
    return allele


class VCF:
    """VCF file handler class for loading and processing VCF files.

    This class provides methods to read VCF files, count SNPs, validate
    against reference genomes.

    Attributes:
        vcf_file (str): Path to the VCF file
        snps (list): List of SNP objects loaded from the file
    """

    def __init__(
        self,
        vcf_file,
        require_sorted=False,
        validate_ref_fasta=None,
        flip_ref=False,
        start_i=None,
        end_i=None,
        pregrouped_seqs=False,
    ):
        """Initialize VCF object and load SNPs from the VCF file.

        Args:
            vcf_file (str): Path to the VCF file
            require_sorted (bool): Whether to require sorted input
            validate_ref_fasta (str): Path to reference FASTA for validation
            flip_ref (bool): Whether to flip reference alleles if needed
            start_i (int): Start index for reading SNPs
            end_i (int): End index for reading SNPs
        """
        self.vcf_file = vcf_file
        self.require_sorted = require_sorted
        self.validate_ref_fasta = validate_ref_fasta
        self.flip_ref = flip_ref
        self.start_i = start_i
        self.end_i = end_i
        self.pregrouped_seqs = pregrouped_seqs
        self.snps = self.read_snps()

    def read_snps(self):
        """Read SNPs from the VCF file using instance attributes.

        Returns:
            list: List of SNP objects
        """
        if self.vcf_file[-3:] == ".gz":
            vcf_in = gzip.open(self.vcf_file, "rt")
        else:
            vcf_in = open(self.vcf_file)

        # read through header
        line = vcf_in.readline()
        vcf_cols = None
        while line and line[0] == "#":
            if line[0:2] != "##":  # column header line starts with single #
                vcf_cols = line.strip().split("\t")
            line = vcf_in.readline()

        # to check sorted
        if self.require_sorted:
            seen_chrs = set()
            prev_chr = None
            prev_pos = -1

        # to check reference
        if self.validate_ref_fasta is not None:
            genome_open = pysam.Fastafile(self.validate_ref_fasta)

        # read in SNPs
        snps = []
        si = 0
        while line:
            if self.start_i is None or self.start_i <= si < self.end_i:
                snps.append(
                    SNP(line, vcf_cols=vcf_cols, pregrouped_seqs=self.pregrouped_seqs)
                )

                if self.require_sorted:
                    if prev_chr is not None:
                        # same chromosome
                        if prev_chr == snps[-1].chr:
                            if snps[-1].pos < prev_pos:
                                print(
                                    "Sorted VCF required. Mis-ordered position: %s"
                                    % line.rstrip(),
                                    file=sys.stderr,
                                )
                                exit(1)
                        elif snps[-1].chr in seen_chrs:
                            print(
                                "Sorted VCF required. Mis-ordered chromosome: %s"
                                % line.rstrip(),
                                file=sys.stderr,
                            )
                            exit(1)

                    seen_chrs.add(snps[-1].chr)
                    prev_chr = snps[-1].chr
                    prev_pos = snps[-1].pos

                if self.validate_ref_fasta is not None:
                    ref_n = len(snps[-1].ref_allele)
                    snp_pos = snps[-1].pos - 1
                    ref_snp = genome_open.fetch(
                        snps[-1].chr, snp_pos, snp_pos + ref_n
                    ).upper()
                    if snps[-1].ref_allele != ref_snp:
                        if not self.flip_ref:
                            # bail
                            print(
                                "ERROR: %s does not match reference %s"
                                % (snps[-1], ref_snp),
                                file=sys.stderr,
                            )
                            exit(1)

                        else:
                            alt_n = len(snps[-1].alt_allele)
                            ref_snp = genome_open.fetch(
                                snps[-1].chr, snp_pos, snp_pos + alt_n
                            ).upper()

                            # if alt matches fasta reference
                            if snps[-1].alt_allele == ref_snp:
                                # flip alleles
                                snps[-1].flip_alleles()

                            else:
                                # bail
                                print(
                                    "ERROR: %s does not match reference %s"
                                    % (snps[-1], ref_snp),
                                    file=sys.stderr,
                                )
                                exit(1)

            si += 1
            line = vcf_in.readline()

        vcf_in.close()
        return snps


class SNP:
    """SNP

    Represent SNPs read in from a VCF file

    Attributes:
        vcf_line (str)
    """

    def __init__(self, vcf_line, vcf_cols=None, pregrouped_seqs=False):
        a = vcf_line.split()
        # self.chr = a[0]
        if a[0].startswith("chr"):
            self.chr = a[0]
        else:
            self.chr = "chr%s" % a[0]
        self.pos = int(a[1])
        self.rsid = a[2]
        self.ref_allele = a[3]

        if pregrouped_seqs:
            # add additional info to SNP object
            if (
                "genes" not in vcf_cols
                or "seq_start" not in vcf_cols
                or "seq_end" not in vcf_cols
            ):
                raise ValueError(
                    f"VCF file must contain 'genes', 'seq_start' and 'seq_end' columns in the header when using -pregrouped_seqs."
                )
            if len(a) < len(vcf_cols):
                raise ValueError(
                    f"SNP entry missing required additional columns: {vcf_line.strip()}"
                )
            self.linked_gene_ids = a[vcf_cols.index("genes")].split(",")
            if not self.linked_gene_ids:
                raise ValueError(
                    f"SNP entry has empty 'genes' column: {vcf_line.strip()}"
                )
            self.seq_start = int(a[vcf_cols.index("seq_start")])
            self.seq_end = int(a[vcf_cols.index("seq_end")])
            if not (self.seq_start <= self.pos <= self.seq_end):
                raise ValueError(
                    f"SNP {self.chr}:{self.pos} position is outside its sequence range [{self.seq_start}, {self.seq_end}]"
                )

        # Check for biallelic requirement
        alt_alleles_raw = a[4].split(",")
        if len(alt_alleles_raw) > 1:
            raise ValueError(
                f"Multi-allelic SNP not supported: {self.rsid} at {self.chr}:{self.pos} "
                f"has {len(alt_alleles_raw)} alternative alleles: {','.join(alt_alleles_raw)}. "
                f"Only biallelic SNPs are supported."
            )

        self.alt_allele = alt_alleles_raw[0]
        self.flipped = False

        if self.rsid == ".":
            self.rsid = "%s:%d" % (self.chr, self.pos)

    @classmethod
    def from_fields(cls, chrom, pos, ref, alt, rsid=None):
        """Construct an SNP directly from fields, bypassing VCF-line parsing."""
        if rsid is None:
            rsid = "."
        line = f"{chrom}\t{int(pos)}\t{rsid}\t{ref}\t{alt}"
        return cls(line)

    def to_vcf_line(self):
        """Render this SNP as a single tab-separated VCF data line (with trailing newline)."""
        return (
            f"{self.chr}\t{self.pos}\t{self.rsid}\t"
            f"{self.ref_allele}\t{self.alt_allele}\t.\t.\t.\n"
        )

    def flip_alleles(self):
        """Flip reference and alt allele."""
        self.ref_allele, self.alt_allele = self.alt_allele, self.ref_allele
        self.flipped = True

    def get_alleles(self):
        """Return a list of all alleles"""
        return [self.ref_allele, self.alt_allele]

    def indel_size(self):
        """Return the size of the indel."""
        return len(self.alt_allele) - len(self.ref_allele)

    def __str__(self):
        return "SNP(%s, %s:%d, %s/%s)" % (
            self.rsid,
            self.chr,
            self.pos,
            self.ref_allele,
            self.alt_allele,
        )


_RSID_RE = re.compile(r"^rs\d+$", re.IGNORECASE)
_GTEX_RE = re.compile(
    r"^(chr[\w]+)_(\d+)_([ACGTN]+)_([ACGTN]+)(?:_b\d+)?$", re.IGNORECASE
)


def parse_snp_input(s, rsid_vcf=None, rsid_index=None):
    """Parse a free-form SNP identifier into an `SNP` object.

    Accepts:
      - rsid:           "rs12345"   (requires `rsid_index` or `rsid_vcf` for lookup)
      - GTEx-style:     "chr6_146898162_C_A_b38"
      - colon form:     "chr6:146898162:C:A"
      - whitespace:     "chr6 146898162 C A"

    For rsid inputs, an indexed SQLite lookup (`rsid_index`, built from the full
    gnomAD observed index) is preferred when given, since it resolves rare variants
    that the common-only `rsid_vcf` scan misses. If `rsid_index` is unset, missing
    on disk, or lacks the rsid, the lookup falls back to scanning `rsid_vcf`.
    """
    s = s.strip()

    if _RSID_RE.match(s):
        if rsid_index is not None:
            try:
                snp = lookup_rsid_indexed(s, rsid_index)
            except FileNotFoundError:
                # index not built; fall through to the VCF scan if we have one
                if rsid_vcf is None:
                    raise
                snp = None
            if snp is not None:
                return snp
        if rsid_vcf is None:
            raise ValueError(
                f"rsid input {s!r} requires rsid_index= or rsid_vcf= for lookup"
            )
        return _lookup_rsid(s, rsid_vcf)

    m = _GTEX_RE.match(s)
    if m:
        chrom, pos, ref, alt = m.groups()
        return SNP.from_fields(chrom, int(pos), ref.upper(), alt.upper(), rsid=s)

    parts = re.split(r"[:\s,]+", s)
    if len(parts) == 4:
        chrom, pos, ref, alt = parts
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        return SNP.from_fields(chrom, int(pos), ref.upper(), alt.upper())

    raise ValueError(f"could not parse SNP input {s!r}")


def _lookup_rsid(rsid, rsid_vcf):
    """Look up an rsid in a (potentially huge) VCF. Returns an `SNP`."""
    rsid_vcf = os.path.expanduser(rsid_vcf)
    rsid_l = str(rsid).lower()
    opener = gzip.open if rsid_vcf.endswith(".gz") else open
    with opener(rsid_vcf, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[2].lower() == rsid_l:
                return SNP(line)
    raise ValueError(f"rsid {rsid!r} not found in {rsid_vcf}")


def lookup_rsid_indexed(rsid, db_path):
    """Resolve an rsid to an `SNP` via the SQLite rsid index.

    The index (built from the full gnomAD observed index) has one table `rsid`
    with columns ``(rsid, chrom, pos, ref, alt)`` and an index on ``rsid``. This
    resolves rare variants that the common-only `rsid_vcf` scan misses.

    Returns an `SNP`, or ``None`` if the rsid is not present (so callers can fall
    back to a VCF scan). Raises if the database file itself is missing.
    """
    import sqlite3

    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"rsid index not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT chrom, pos, ref, alt FROM rsid WHERE rsid = ? LIMIT 1",
            (str(rsid),),
        ).fetchone()
    finally:
        con.close()

    if row is None:
        return None
    chrom, pos, ref, alt = row
    return SNP.from_fields(
        chrom, int(pos), str(ref).upper(), str(alt).upper(), rsid=rsid
    )


# INFO fields carried by the gnomAD observed index (build_observed_index.sh).
GNOMAD_INFO_FIELDS = (
    "AC",
    "AF",
    "cadd_raw_score",
    "revel_max",
    "spliceai_ds_max",
    "pangolin_largest_ds",
    "phylop",
)


def fetch_gnomad_annotation(chrom, pos, ref, alt, observed_dir):
    """Fetch gnomAD annotations for a variant from the observed index.

    Tabix-queries ``{observed_dir}/gnomad.v4.1.observed.{chrom}.vcf.bgz`` at the
    1-based `pos`, matching REF/ALT, and parses the INFO column into a dict of
    the fields in `GNOMAD_INFO_FIELDS` (numeric where possible). Works for any
    variant present in gnomAD, including rare ones.

    Returns the annotation dict, or ``None`` if the variant is absent (e.g. a
    novel/fabricated variant) or the per-chromosome index file is missing.
    """
    observed_dir = os.path.expanduser(observed_dir)
    if not str(chrom).startswith("chr"):
        chrom = f"chr{chrom}"
    bgz = os.path.join(observed_dir, f"gnomad.v4.1.observed.{chrom}.vcf.bgz")
    if not os.path.exists(bgz):
        return None

    ref = str(ref).upper()
    alt = str(alt).upper()
    pos = int(pos)
    tbx = pysam.TabixFile(bgz)
    try:
        # tabix is 0-based half-open; query the single base at `pos`.
        rows = tbx.fetch(chrom, pos - 1, pos)
        for line in rows:
            f = line.rstrip("\n").split("\t")
            if int(f[1]) != pos or f[3].upper() != ref or f[4].upper() != alt:
                continue
            info = {}
            for field in f[7].split(";"):
                if "=" not in field:
                    continue
                k, v = field.split("=", 1)
                if k in GNOMAD_INFO_FIELDS:
                    try:
                        info[k] = float(v)
                    except ValueError:
                        info[k] = v
            return info
    finally:
        tbx.close()
    return None


def write_snp_vcf(snps, path):
    """Write one or more `SNP`s to a VCF in baskerville's minimal schema."""
    if isinstance(snps, SNP):
        snps = [snps]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for s in snps:
            f.write(s.to_vcf_line())
    return path


def make_alt_1hot(ref_1hot, snp_seq_pos, ref_allele, alt_allele):
    """Return alternative allele one hot coding.

    Args:
        ref_1hot (np.array): Reference allele one hot coding.
        snp_seq_pos (int): SNP position in sequence.
        ref_allele (str): Reference allele.
        alt_allele (str): Alternative allele.

    Returns:
        np.array: Alternative allele one hot coding.
    """
    ref_n = len(ref_allele)
    alt_n = len(alt_allele)

    # copy reference
    alt_1hot = np.copy(ref_1hot)

    if alt_n == ref_n:
        # SNP
        dna.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

    elif ref_n > alt_n:
        # deletion
        delete_len = ref_n - alt_n
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_delete(alt_1hot, snp_seq_pos + 1, delete_len)
        else:
            print(
                "WARNING: Deletion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    else:
        # insertion
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_insert(alt_1hot, snp_seq_pos + 1, alt_allele[1:])
        else:
            print(
                "WARNING: Insertion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    return alt_1hot


class SNPCluster:
    def __init__(self):
        self.snps = []
        self.chr = None
        self.start = None
        self.end = None

    def add_snp(self, snp):
        """Add SNP to cluster."""
        self.snps.append(snp)

    def delimit(self, seq_len):
        """Delimit sequence boundaries."""
        positions = [snp.pos for snp in self.snps]
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_mid = (pos_min + pos_max) // 2

        self.chr = self.snps[0].chr
        self.start = pos_mid - seq_len // 2
        self.end = self.start + seq_len

        # for snp in self.snps:
        #     snp.seq_pos = snp.pos - 1 - self.start

    def get_1hots(self, genome_open):
        """Get list of one hot coded sequences."""
        seqs1_list = []

        # extract reference
        if self.start < 0:
            ref_seq = (
                "N" * (-self.start) + genome_open.fetch(self.chr, 0, self.end).upper()
            )
        else:
            ref_seq = genome_open.fetch(self.chr, self.start, self.end).upper()

        # extend to full length
        if len(ref_seq) < self.end - self.start:
            ref_seq += "N" * (self.end - self.start - len(ref_seq))

        # verify reference alleles
        for snp in self.snps:
            ref_n = len(snp.ref_allele)
            snp_pos = snp.pos - 1 - self.start
            ref_snp = ref_seq[snp_pos : snp_pos + ref_n]
            if snp.ref_allele != ref_snp:
                print(
                    f"ERROR: {snp} does not match reference {ref_snp}",
                    file=sys.stderr,
                )
                exit(1)

        # 1 hot code reference sequence
        ref_1hot = dna.dna_1hot(ref_seq)
        seqs1_list = [ref_1hot]

        # make alternative 1 hot coded sequences
        # (assuming SNP is 1-based indexed)
        for snp in self.snps:
            snp_pos = snp.pos - 1 - self.start
            alt_1hot = make_alt_1hot(ref_1hot, snp_pos, snp.ref_allele, snp.alt_allele)
            seqs1_list.append(alt_1hot)

        # transpose for torch
        seqs1_list = [seq_1hot.T for seq_1hot in seqs1_list]

        return seqs1_list
