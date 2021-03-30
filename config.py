import argparse
import torch
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ag_news/', help='the path of dataset')
    parser.add_argument('--train_file', type=str, default='processed_train.csv', help='train_dataset\'s name')
    parser.add_argument('--val_file', type=str, default='processed_val.csv', help='val_dataset\'s name')
    parser.add_argument('--test_file', type=str, default='processed_test.csv', help='test_dataset\'s name')

    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--output_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--r', type=int, default=5)


    parser.add_argument('--word_vectors', type=str, default='w2v/glove/glove.6B.50d.txt')

    args = parser.parse_args()

    return args