"""Compatibility entry point for the canonical LSTM trainer.

The project previously had two different TrafficLSTM architectures writing to
the same checkpoint. This wrapper keeps one source of truth.
"""

from training.train_lstm import train


if __name__ == "__main__":
    train()
