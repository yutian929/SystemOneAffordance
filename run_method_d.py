"""Command-line entry for the standalone MethodD class."""
import argparse

from method_d import MethodD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/umd_debug.json')
    parser.add_argument('--output', default='outputs/method_d_run')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--stage', choices=['prepare', 'run'], default='run')
    parser.add_argument('--candidate-model', choices=['sam2', 'unsamv2-plus'], default='sam2')
    parser.add_argument('--granularity', type=float, default=0.15,
                        help='UnSAMv2+ target granularity in [0.1, 1]; smaller is finer')
    parser.add_argument('--max-candidates', type=int, default=24,
                        help='Maximum candidates after deduplication; 0 keeps all')
    parser.add_argument('--candidate-cache', help='Optional prior A/D run containing compatible candidates')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must lie in [0, 1]')
    with MethodD(device=args.device, threshold=args.threshold, output_dir=args.output,
                 candidate_cache=args.candidate_cache, candidate_model=args.candidate_model,
                 granularity=args.granularity, max_candidates=args.max_candidates) as pipeline:
        if args.stage == 'prepare':
            pipeline.prepare_manifest(args.manifest)
            print(f'Candidates prepared: {pipeline.output_dir}', flush=True)
        else:
            report = pipeline.run_manifest(args.manifest)
            print(f"Finished: {pipeline.output_dir}; debug mean IoU={report['mean_iou']}", flush=True)


if __name__ == '__main__':
    main()
