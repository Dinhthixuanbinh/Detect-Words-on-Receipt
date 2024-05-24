# config.py

import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser(description='BERT Sentiment Analysis')

    # Model hyperparameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pre-trained BERT model name')
    parser.add_argument('--num_labels', type=int, default=5, help='Number of labels in the classification task')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')

    # Data paths
    parser.add_argument('--data_path', type=str, default='path/to/data', help='Path to training data')
    # parser.add_argument('--val_data_path', type=str, default='path/to/val/data', help='Path to validation data')
    parser.add_argument('--xml_path', type=str, default='path/to/xml/file', help='Path to XML file')


    # Save model path
    parser.add_argument('--save_model_path', type=str, default='path/to/save/model', help='Path to save the trained model')



    args = parser.parse_args()
    return args
