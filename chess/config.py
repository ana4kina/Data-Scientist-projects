MODEL_CONFIG = {
    'board_shape': (8, 8, 12),
    'additional_features_dim': 5,
    'conv_filters': [64, 128, 256],
    'dense_units': [512, 256],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

TRAINING_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42
}