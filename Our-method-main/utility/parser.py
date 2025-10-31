import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--data_path', nargs='?', default='./Data/', help='Input data path.')

    parser.add_argument(
        '--dataset',
        nargs='?',
        default='foursquare',
        help='Choose a dataset from {ml-100k, lastfm, foursquare, Yelp, amazon-electro}',
    )

    parser.add_argument(
        '--model_type',
        nargs='?',
        default='GCN',
    )

    parser.add_argument('--epoch', type=int, default=250, help='Number of epoch.')
    parser.add_argument('--patience', type=int, default=30, help='the number of epoch to wait before early stopping.')

    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')

    parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64]', help='Output sizes of every layer.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')

    parser.add_argument('--groups', type=int, default=2, help='Number of subgraphs for GCN model.')

    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for federated server.')
    parser.add_argument('--clients', type=int, default=128, help='Number of federated clients each round.')

    parser.add_argument('--regs', nargs='?', default='[1e-4]', help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')

    # argument for adversaria loss
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for adversarial weights.')
    parser.add_argument('--reg_adv', type=float, default=0.01, help='Regularization for adversarial loss.')

    parser.add_argument(
        '--node_dropout_flag',
        type=int,
        default=1,
        help='0: Disable node dropout, 1: Activate node dropout',
    )
    parser.add_argument(
        '--node_dropout',
        nargs='?',
        default='[0.5]',
        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.',
    )
    parser.add_argument(
        '--mess_dropout',
        nargs='?',
        default='[0.0,0.0,0.0]',
        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.',
    )

    parser.add_argument(
        '--adj_type',
        nargs='?',
        default='norm',
        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.',
    )
    parser.add_argument(
        '--loss_function',
        nargs='?',
        default='bpr',
        help='Specify the loss function from {bpr, apr}.',
    )

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument(
        '--Ks',
        nargs='?',
        default='[5,10,20,30,40,50,100]',
        help='Ks for evaluation',
    )

    parser.add_argument(
        '--save_flag',
        type=int,
        default=1,
        help='0: Disable model saver, 1: Activate model saver',
    )

    parser.add_argument(
        '--test_flag',
        nargs='?',
        default='part',
        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch',
    )

    parser.add_argument(
        '--report',
        type=int,
        default=1,
        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels',
    )

    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')

    return parser.parse_args()
