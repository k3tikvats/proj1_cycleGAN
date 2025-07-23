#!/usr/bin/env python3
"""
Main entry point for SAR to EO CycleGAN project.
Provides a unified interface for training, evaluation, and testing.
"""

import argparse
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *


def run_training(args):
    """Run training with specified parameters."""
    from train_cycleGAN import main as train_main
    
    # Update config if needed
    if args.data_path:
        global DATA_PATH
        DATA_PATH = args.data_path
    
    if args.epochs:
        global EPOCHS
        EPOCHS = args.epochs
    
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    
    print(f"Starting training with:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    
    train_main()


def run_evaluation(args):
    """Run evaluation with specified parameters."""
    from evaluate_results import main as eval_main
    
    # Set up arguments for evaluation
    sys.argv = ['evaluate_results.py']
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    if args.num_samples:
        sys.argv.extend(['--num_samples', str(args.num_samples)])
    if args.data_path:
        sys.argv.extend(['--data_path', args.data_path])
    if args.output_dir:
        sys.argv.extend(['--output_dir', args.output_dir])
    
    eval_main()


def run_test(args):
    """Run environment and setup tests."""
    from utils import main as utils_main
    utils_main()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='SAR to EO CycleGAN - Main Interface')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train the CycleGAN model')
    train_parser.add_argument('--data_path', type=str, help='Path to dataset')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size for training')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--num_samples', type=int, default=10,
                           help='Number of samples to evaluate')
    eval_parser.add_argument('--data_path', type=str, help='Path to dataset')
    eval_parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Run environment and setup tests')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'eval':
        run_evaluation(args)
    elif args.command == 'test':
        run_test(args)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python run.py test                    # Test environment setup")
        print("  python run.py train                   # Start training")
        print("  python run.py eval --checkpoint path  # Evaluate model")


if __name__ == "__main__":
    main()
