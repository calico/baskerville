import gzip
import os
import pathlib
import tempfile
import pytest
import pysam
import numpy as np

from baskerville.vcf import (
    VCF,
    SNP,
    SNPCluster,
    cap_allele,
    fetch_gnomad_annotation,
    lookup_rsid_indexed,
    parse_snp_input,
    write_snp_vcf,
)


@pytest.fixture
def test_vcf_file():
    """Use the fixed test VCF file with known SNPs and correct reference alleles."""
    test_dir = str(pathlib.Path(__file__).parent)
    vcf_file = f"{test_dir}/data/sc3_snps.vcf"

    # Verify the file exists
    assert os.path.exists(vcf_file), f"Test VCF file not found: {vcf_file}"

    return vcf_file


@pytest.fixture
def temp_vcf_file():
    """Create a temporary VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##reference=test.fa
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	100	rs100	A	T	60	PASS	AF=0.5
chr1	200	rs200	C	G	60	PASS	AF=0.3
chr1	300	rs300	G	A	60	PASS	AF=0.7
chr1	400	rs400	T	C	60	PASS	AF=0.4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
        f.write(vcf_content)
        temp_file = f.name

    yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def temp_gzipped_vcf_file():
    """Create a temporary gzipped VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##reference=test.fa
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	100	rs100	A	T	60	PASS	AF=0.5
chr1	200	rs200	C	G	60	PASS	AF=0.3
"""

    with tempfile.NamedTemporaryFile(suffix=".vcf.gz", delete=False) as f:
        temp_file = f.name

    with gzip.open(temp_file, "wt") as f:
        f.write(vcf_content)

    yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


class TestVCF:
    """Test cases for the VCF class."""

    def test_vcf_initialization(self, test_vcf_file):
        """Test VCF class initialization."""
        vcf = VCF(test_vcf_file)
        assert vcf.vcf_file == test_vcf_file
        assert len(vcf.snps) == 4  # SNPs are loaded automatically
        assert vcf.snps[0].rsid == "rs1000"

    def test_read_snps_basic(self, test_vcf_file):
        """Test basic SNP loading."""
        vcf = VCF(test_vcf_file)

        # SNPs are already loaded during initialization
        assert len(vcf.snps) == 4

        # Test reloading SNPs with different parameters
        snps = vcf.read_snps()
        assert len(snps) == 4
        assert len(vcf.snps) == 4  # Should update instance as well

        # Check first SNP
        assert vcf.snps[0].chr == "chrI"
        assert vcf.snps[0].pos == 1000
        assert vcf.snps[0].rsid == "rs1000"
        assert vcf.snps[0].ref_allele == "A"
        assert vcf.snps[0].alt_allele == "C"

    def test_read_snps_with_range(self, temp_vcf_file):
        """Test loading SNPs with start/end index range."""
        # Load with range parameters during initialization
        vcf = VCF(temp_vcf_file, start_i=1, end_i=3)

        assert len(vcf.snps) == 2
        assert vcf.snps[0].rsid == "rs200"
        assert vcf.snps[1].rsid == "rs300"

    def test_read_snps_biallelic_only(self, temp_vcf_file):
        """Test that only biallelic SNPs are loaded."""
        vcf = VCF(temp_vcf_file)

        # All SNPs should be biallelic
        for snp in vcf.snps:
            assert hasattr(snp, "alt_allele")  # Single alt allele
            assert not hasattr(snp, "alt_alleles")  # No longer exists
            assert len(snp.get_alleles()) == 2  # ref + alt

    def test_multiallelic_snp_error(self):
        """Test that multi-allelic SNPs raise an error."""
        vcf_content = """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	100	rs100	A	T,G	60	PASS	AF=0.5,0.3
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Multi-allelic SNP not supported"):
                VCF(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_snp_cluster_sequences(self, test_vcf_file):
        """Test SNPCluster sequence generation (replacement for get_sequences)."""
        vcf_obj = VCF(test_vcf_file)

        # Create SNPCluster objects with single SNPs
        snp_clusters = []
        all_seqs = []
        all_headers = []

        # Open genome FASTA
        genome_open = pysam.Fastafile("src/tests/data/sc3.fa.gz")

        for variant in vcf_obj.snps:
            # Create a single-SNP cluster
            cluster = SNPCluster()
            cluster.add_snp(variant)
            cluster.delimit(1000)
            snp_clusters.append(cluster)

            # Get one hot coded sequences (reference + alt)
            cluster_seqs = cluster.get_1hots(genome_open)
            all_seqs.extend(cluster_seqs)

            # Create headers for reference and alt sequences
            ref_header = f"{variant.rsid}_{cap_allele(variant.ref_allele)}"
            all_headers.append(ref_header)

            alt_header = f"{variant.rsid}_{cap_allele(variant.alt_allele)}"
            all_headers.append(alt_header)

        genome_open.close()

        # Convert to numpy array
        seq_vecs = np.array(all_seqs)

        # Should have sequences for ref + alt alleles for each SNP
        # 4 SNPs * 2 alleles each = 8 sequences
        assert seq_vecs.shape[0] == 8
        assert len(all_headers) == 8
        assert len(vcf_obj.snps) == 4

        # Check sequence length
        assert seq_vecs.shape[1] == 4  # 4 nucleotides (A, C, G, T)
        assert seq_vecs.shape[2] == 1000  # sequence length


class TestSNP:
    """Test cases for the SNP class."""

    def test_snp_initialization(self):
        """Test SNP object initialization."""
        vcf_line = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp = SNP(vcf_line)

        assert snp.chr == "chr1"
        assert snp.pos == 1000
        assert snp.rsid == "rs1000"
        assert snp.ref_allele == "A"
        assert snp.alt_allele == "T"
        assert snp.flipped == False

    def test_snp_chr_prefix(self):
        """Test SNP chromosome prefix handling."""
        # Without chr prefix
        vcf_line1 = "1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp1 = SNP(vcf_line1)
        assert snp1.chr == "chr1"

        # With chr prefix
        vcf_line2 = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp2 = SNP(vcf_line2)
        assert snp2.chr == "chr1"

    def test_snp_default_rsid(self):
        """Test SNP default rsid generation."""
        vcf_line = "chr1\t1000\t.\tA\tT\t60\tPASS\tAF=0.5"
        snp = SNP(vcf_line)
        assert snp.rsid == "chr1:1000"

    def test_snp_multi_alt_error(self):
        """Test SNP with multiple alternative alleles raises error."""
        vcf_line = "chr1\t1000\trs1000\tA\tT,G\t60\tPASS\tAF=0.3,0.2"

        with pytest.raises(ValueError, match="Multi-allelic SNP not supported"):
            SNP(vcf_line)

    def test_snp_flip_alleles(self):
        """Test flipping reference and alternative alleles."""
        vcf_line = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp = SNP(vcf_line)

        original_ref = snp.ref_allele
        original_alt = snp.alt_allele

        snp.flip_alleles()

        assert snp.ref_allele == original_alt
        assert snp.alt_allele == original_ref
        assert snp.flipped == True

    def test_snp_get_alleles(self):
        """Test getting all alleles."""
        vcf_line = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp = SNP(vcf_line)

        alleles = snp.get_alleles()
        assert alleles == ["A", "T"]

    def test_snp_indel_size(self):
        """Test calculating indel size."""
        # SNV (no indel)
        vcf_line1 = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp1 = SNP(vcf_line1)
        assert snp1.indel_size() == 0

        # Insertion
        vcf_line2 = "chr1\t1000\trs1000\tA\tATG\t60\tPASS\tAF=0.5"
        snp2 = SNP(vcf_line2)
        assert snp2.indel_size() == 2

        # Deletion
        vcf_line3 = "chr1\t1000\trs1000\tATG\tA\t60\tPASS\tAF=0.5"
        snp3 = SNP(vcf_line3)
        assert snp3.indel_size() == -2

    def test_snp_str_representation(self):
        """Test SNP string representation."""
        vcf_line = "chr1\t1000\trs1000\tA\tT\t60\tPASS\tAF=0.5"
        snp = SNP(vcf_line)

        str_repr = str(snp)
        assert "rs1000" in str_repr
        assert "chr1:1000" in str_repr
        assert "A/T" in str_repr


class TestSNPFields:
    """Test cases for SNP.from_fields and SNP.to_vcf_line."""

    def test_from_fields_chr_prefixed(self):
        snp = SNP.from_fields("chr1", 100, "A", "T")
        assert snp.chr == "chr1"
        assert snp.pos == 100
        assert snp.ref_allele == "A"
        assert snp.alt_allele == "T"
        assert snp.rsid == "chr1:100"

    def test_from_fields_chr_unprefixed(self):
        # Regression: default rsid must use the normalized chr.
        snp = SNP.from_fields("1", 100, "A", "T")
        assert snp.chr == "chr1"
        assert snp.rsid == "chr1:100"

    def test_from_fields_explicit_rsid(self):
        snp = SNP.from_fields("chr1", 100, "A", "T", rsid="rs9")
        assert snp.rsid == "rs9"

    def test_to_vcf_line_round_trip(self):
        original = SNP.from_fields("chr2", 250, "G", "C", rsid="rs250")
        line = original.to_vcf_line()
        reparsed = SNP(line)
        assert reparsed.chr == original.chr
        assert reparsed.pos == original.pos
        assert reparsed.rsid == original.rsid
        assert reparsed.ref_allele == original.ref_allele
        assert reparsed.alt_allele == original.alt_allele


class TestParseSnpInput:
    """Test cases for parse_snp_input."""

    def test_colon_form(self):
        snp = parse_snp_input("chr6:146898162:C:A")
        assert snp.chr == "chr6"
        assert snp.pos == 146898162
        assert snp.ref_allele == "C"
        assert snp.alt_allele == "A"

    def test_whitespace_form(self):
        snp = parse_snp_input("chr6 146898162 C A")
        assert snp.chr == "chr6"
        assert snp.pos == 146898162
        assert snp.ref_allele == "C"
        assert snp.alt_allele == "A"

    def test_colon_form_unprefixed_chrom(self):
        snp = parse_snp_input("6:146898162:C:A")
        assert snp.chr == "chr6"
        assert snp.pos == 146898162

    def test_gtex_form(self):
        s = "chr6_146898162_C_A_b38"
        snp = parse_snp_input(s)
        assert snp.chr == "chr6"
        assert snp.pos == 146898162
        assert snp.ref_allele == "C"
        assert snp.alt_allele == "A"
        assert snp.rsid == s

    def test_gtex_form_lowercase_alleles(self):
        snp = parse_snp_input("chr6_146898162_c_a")
        assert snp.ref_allele == "C"
        assert snp.alt_allele == "A"

    def test_rsid_without_vcf_raises(self):
        with pytest.raises(ValueError, match="requires rsid_index= or rsid_vcf="):
            parse_snp_input("rs123")

    def test_rsid_lookup(self, temp_vcf_file):
        snp = parse_snp_input("rs200", rsid_vcf=temp_vcf_file)
        assert snp.chr == "chr1"
        assert snp.pos == 200
        assert snp.ref_allele == "C"
        assert snp.alt_allele == "G"

    def test_rsid_lookup_case_insensitive(self, temp_vcf_file):
        # Regression: _lookup_rsid must match rsids case-insensitively.
        snp = parse_snp_input("RS200", rsid_vcf=temp_vcf_file)
        assert snp.chr == "chr1"
        assert snp.pos == 200

    def test_rsid_lookup_gzipped(self, temp_gzipped_vcf_file):
        snp = parse_snp_input("rs100", rsid_vcf=temp_gzipped_vcf_file)
        assert snp.chr == "chr1"
        assert snp.pos == 100
        assert snp.ref_allele == "A"
        assert snp.alt_allele == "T"

    def test_rsid_missing_raises(self, temp_vcf_file):
        with pytest.raises(ValueError, match="rs9999"):
            parse_snp_input("rs9999", rsid_vcf=temp_vcf_file)

    def test_garbage_input_raises(self):
        with pytest.raises(ValueError, match="could not parse"):
            parse_snp_input("not-a-snp")


@pytest.fixture
def temp_rsid_index():
    """A small SQLite rsid index mirroring the gnomAD observed-index lookup."""
    import sqlite3

    fd, db_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE rsid (rsid TEXT, chrom TEXT, pos INT, ref TEXT, alt TEXT)"
    )
    con.executemany(
        "INSERT INTO rsid VALUES (?, ?, ?, ?, ?)",
        [
            ("rs777", "chr3", 12345, "A", "G"),
            ("rs888", "chr12", 9999, "C", "T"),
        ],
    )
    con.execute("CREATE INDEX idx_rsid ON rsid (rsid)")
    con.commit()
    con.close()

    yield db_path

    if os.path.exists(db_path):
        os.unlink(db_path)


class TestRsidIndex:
    """Test cases for the SQLite rsid index lookup."""

    def test_lookup_hit(self, temp_rsid_index):
        snp = lookup_rsid_indexed("rs777", temp_rsid_index)
        assert snp is not None
        assert snp.chr == "chr3"
        assert snp.pos == 12345
        assert snp.ref_allele == "A"
        assert snp.alt_allele == "G"
        assert snp.rsid == "rs777"

    def test_lookup_miss_returns_none(self, temp_rsid_index):
        assert lookup_rsid_indexed("rs000", temp_rsid_index) is None

    def test_lookup_missing_db_raises(self):
        with pytest.raises(FileNotFoundError):
            lookup_rsid_indexed("rs777", "/nonexistent/rsid_index.sqlite")

    def test_parse_prefers_index(self, temp_rsid_index, temp_vcf_file):
        # index hit should win without consulting the VCF
        snp = parse_snp_input(
            "rs888", rsid_index=temp_rsid_index, rsid_vcf=temp_vcf_file
        )
        assert snp.chr == "chr12"
        assert snp.pos == 9999

    def test_parse_falls_back_to_vcf(self, temp_rsid_index, temp_vcf_file):
        # rs200 isn't in the index; should fall back to the VCF scan
        snp = parse_snp_input(
            "rs200", rsid_index=temp_rsid_index, rsid_vcf=temp_vcf_file
        )
        assert snp.chr == "chr1"
        assert snp.pos == 200

    def test_parse_falls_back_when_index_missing(self, temp_vcf_file):
        # an unbuilt index shouldn't block the VCF scan
        missing = "/nonexistent/rsid_index.sqlite"
        snp = parse_snp_input("rs200", rsid_index=missing, rsid_vcf=temp_vcf_file)
        assert snp.chr == "chr1"
        assert snp.pos == 200

        # ... but with no VCF to fall back to, the missing index still raises
        with pytest.raises(FileNotFoundError):
            parse_snp_input("rs200", rsid_index=missing)


class TestGnomadAnnotation:
    """Test cases for fetch_gnomad_annotation."""

    @pytest.fixture
    def observed_dir(self, tmp_path):
        """A tiny bgzipped + tabixed observed-index file for one chromosome."""
        tmpdir = str(tmp_path)
        vcf_path = os.path.join(tmpdir, "gnomad.v4.1.observed.chr6.vcf")
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##contig=<ID=chr6,length=170805979>\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            f.write(
                "chr6\t146898162\t.\tC\tA\t.\t.\t"
                "AC=17415;AF=0.114541;cadd_raw_score=-0.594951;"
                "pangolin_largest_ds=0.01;phylop=-0.27\n"
            )
        # tabix_index bgzips to .vcf.gz; the real index files are named
        # .vcf.bgz, so rename data + tbi to match (same bgzip format).
        pysam.tabix_index(vcf_path, preset="vcf", force=True)
        os.rename(vcf_path + ".gz", vcf_path + ".bgz")
        os.rename(vcf_path + ".gz.tbi", vcf_path + ".bgz.tbi")
        return tmpdir

    def test_fetch_hit(self, observed_dir):
        info = fetch_gnomad_annotation("chr6", 146898162, "C", "A", observed_dir)
        assert info is not None
        assert info["AC"] == pytest.approx(17415)
        assert info["AF"] == pytest.approx(0.114541)
        assert info["phylop"] == pytest.approx(-0.27)

    def test_fetch_unprefixed_chrom(self, observed_dir):
        info = fetch_gnomad_annotation("6", 146898162, "C", "A", observed_dir)
        assert info is not None and info["AC"] == pytest.approx(17415)

    def test_fetch_allele_mismatch_returns_none(self, observed_dir):
        # right position, wrong ALT -> not a match (novel-ish)
        assert (
            fetch_gnomad_annotation("chr6", 146898162, "C", "T", observed_dir) is None
        )

    def test_fetch_missing_chrom_file_returns_none(self, observed_dir):
        assert fetch_gnomad_annotation("chr9", 100, "A", "T", observed_dir) is None


class TestWriteSnpVcf:
    """Test cases for write_snp_vcf."""

    def test_write_single_snp(self):
        snp = SNP.from_fields("chr1", 500, "A", "G", rsid="rs500")
        with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as f:
            out_path = f.name
        try:
            write_snp_vcf(snp, out_path)
            with open(out_path) as f:
                content = f.read()
            assert "##fileformat=VCFv4.2" in content
            assert "#CHROM\tPOS\tID" in content

            loaded = VCF(out_path).snps
            assert len(loaded) == 1
            assert loaded[0].chr == "chr1"
            assert loaded[0].pos == 500
            assert loaded[0].rsid == "rs500"
            assert loaded[0].ref_allele == "A"
            assert loaded[0].alt_allele == "G"
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_write_multiple_snps(self):
        snps = [
            SNP.from_fields("chr1", 100, "A", "T", rsid="rs100"),
            SNP.from_fields("chr2", 200, "C", "G", rsid="rs200"),
            SNP.from_fields("chr3", 300, "G", "A", rsid="rs300"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as f:
            out_path = f.name
        try:
            write_snp_vcf(snps, out_path)
            loaded = VCF(out_path).snps
            assert len(loaded) == 3
            assert [s.rsid for s in loaded] == ["rs100", "rs200", "rs300"]
            assert [s.pos for s in loaded] == [100, 200, 300]
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
