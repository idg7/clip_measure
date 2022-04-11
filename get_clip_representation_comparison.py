import clip
import torch
from representation.pairs_behaviour_setup import get_pairs_comparison
from argparse import ArgumentParser
import plotly.express as px
import plotly.graph_objs as go


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--reps_cache_path', default='/home/ssd_storage/experiments/clip/clip_measurements/results/reps_cache', type=str,
                        help='Path to experiment output directory')
    parser.add_argument('--experiment_name', type=str, default='clip_txt_decoder',
                        help='The specific name of the experiment')
    parser.add_argument('--arch', default='RN50x16', type=str,
                        help='ViT-B/32, RN50x16')
    parser.add_argument('--dist', default='cos', type=str,
                        help='l2, cos')
    parser.add_argument('--output_csv', type=str,
                        help='where to save the csv output (full path)')
    parser.add_argument('--output_html', type=str,
                        help='where to save the html output (full path)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model, preprocess = clip.load(args.arch)
    pairs_paths = {
        "diff_pairs": "/home/administrator/datasets/high_low_ps_images/image_pairs_lists/diff_pairs.txt",
        "high_ps_pairs": "/home/administrator/datasets/high_low_ps_images/image_pairs_lists/high_ps_pairs.txt",
        "low_ps_pairs": "/home/administrator/datasets/high_low_ps_images/image_pairs_lists/low_ps_pairs.txt",
        "same_pairs": "/home/administrator/datasets/high_low_ps_images/image_pairs_lists/same_pairs.txt",
        # "frontal-ref": "/home/administrator/datasets/faces_in_views/frontal_ref.txt",
        # "frontal-quarter_left": "/home/administrator/datasets/faces_in_views/frontal_quarter_left.txt",
        # "frontal-half_left": "/home/administrator/datasets/faces_in_views/frontal_half_left.txt",
        # "half_left-half_right": "/home/administrator/datasets/faces_in_views/half_left-half_right.txt",
        # "quarter_left-quarter_right": "/home/administrator/datasets/faces_in_views/quarter_left-quarter_right.txt"
        }
    pairs_image_dirs = {
        "diff_pairs": "/home/administrator/datasets/high_low_ps_images/joined",
        "high_ps_pairs": "/home/administrator/datasets/high_low_ps_images/joined",
        "low_ps_pairs": "/home/administrator/datasets/high_low_ps_images/joined",
        "same_pairs": "/home/administrator/datasets/high_low_ps_images/joined",
        # "frontal-ref": "/home/administrator/datasets/faces_in_views",
        # "frontal-quarter_left": "/home/administrator/datasets/faces_in_views",
        # "frontal-half_left": "/home/administrator/datasets/faces_in_views",
        # "half_left-half_right": "/home/administrator/datasets/faces_in_views",
        # "quarter_left-quarter_right": "/home/administrator/datasets/faces_in_views"
    }
    pairs_comparison = get_pairs_comparison(pairs_image_dirs, pairs_paths, args.reps_cache_path, args.dist, model, preprocess)
    df = pairs_comparison.compare_lists(model.visual)
    #f'/home/ssd_storage/experiments/clip/{args.arch.replace("/", "")}, {args.dist} dist, features.csv'
    df.to_csv(args.output_csv)
    if args.arch == 'RN50x16':
        df = df[['input'] + [f'Layer 1, Bottleneck {i}' for i in range(1, 7)] + [f'Layer 2, Bottleneck {i}' for i in range(1, 9)] + [f'Layer 3, Bottleneck {i}' for i in range(1, 19)] + [f'Layer 4, Bottleneck {i}' for i in range(1, 9)] + ['output', 'type']]
    if args.arch == 'ViT-B/32':
        df = df[['input'] + [f'Residual attention block {i}' for i in range(1, 13)] + ['output', 'type']]
    for x in df.drop(columns=['type']).columns:
        df[x] = ((df[x]) / (df[x].max()))
    means_df = df.groupby(['type']).mean()

    fig = go.Figure()
    fig.update_layout(width=1000, height=700)
    for pairs_type in means_df.index:
        fig.add_trace(go.Scatter(
            x=means_df.columns,
            y=means_df.query(f'type=="{pairs_type}"').values[0],
            mode='lines+markers',
            name=pairs_type))
        fig.update_yaxes(range=[0, 1])
    fig.show()
    fig.write_html(args.output_html)
