"""GTF scoring options shared by the evaluators and their fold launchers."""


def add_gene_options(parser):
    parser.add_argument("--span", action="store_true", help="Score whole gene spans")
    parser.add_argument(
        "--save_span", action="store_true", help="Write per-gene bin profiles"
    )
    parser.add_argument(
        "--seq_step", type=int, default=1, help="Score genes on every Nth dataset row"
    )
    parser.add_argument(
        "--pseudo_qtl", type=float, help="Gene expression pseudocount quantile"
    )


def check_gene_options(parser, args):
    if args.seq_step < 1:
        parser.error("--seq_step must be positive")
    if args.pseudo_qtl is not None and not 0 <= args.pseudo_qtl <= 1:
        parser.error("--pseudo_qtl must be between 0 and 1")
    if not args.genes_gtf and (
        args.span or args.save_span or args.seq_step != 1 or args.pseudo_qtl is not None
    ):
        parser.error("GTF scoring options require --genes_gtf")


def gene_options(args):
    opts = f" --seq_step {args.seq_step}"
    if args.pseudo_qtl is not None:
        opts += f" --pseudo_qtl {args.pseudo_qtl}"
    if args.span:
        opts += " --span"
    if args.save_span:
        opts += " --save_span"
    return opts


def per_dataset(value, datasets, option):
    paths = value.split(",") if value else [None]
    if len(paths) == 1:
        return paths * len(datasets)
    if len(paths) != len(datasets):
        raise ValueError(f"{option} needs one file, or one per data dir")
    return paths
