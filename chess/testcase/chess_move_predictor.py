import chess
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import argparse
import os
from typing import List, Tuple, Dict, Any
from collections import Counter
import gc


class ChessDataProcessor:    
    def __init__(self, min_move_count: int = 2):
        self.move_to_index = {}
        self.index_to_move = {}
        self.move_vocab_size = 0
        self.min_move_count = min_move_count
        
    def create_move_vocabulary(self, moves: List[str]) -> None:
        move_counts = Counter(moves)
        filtered_moves = [move for move, count in move_counts.items() if count >= self.min_move_count]
        print(f"Всего уникальных ходов: {len(move_counts)}")
        print(f"Ходов после фильтрации (min_count={self.min_move_count}): {len(filtered_moves)}")
        self.move_counts = move_counts
        self.filtered_moves = filtered_moves
        self.move_to_index = {move: idx for idx, move in enumerate(sorted(filtered_moves))}
        self.index_to_move = {idx: move for move, idx in self.move_to_index.items()}
        self.move_vocab_size = len(self.move_to_index)
        print(f"Создан словарь из {self.move_vocab_size} уникальных ходов")
    
    def board_to_matrix(self, board: chess.Board) -> np.ndarray:
        matrix = np.zeros((8, 8, 12), dtype=np.float32)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - square // 8
                col = square % 8
                piece_index = piece_types.index(piece.piece_type)
                if piece.color == chess.WHITE:
                    matrix[row, col, piece_index] = 1
                else:
                    matrix[row, col, piece_index + 6] = 1
        
        return matrix
    
    def encode_move(self, move_uci: str) -> int:
        return self.move_to_index.get(move_uci, -1)
    
    def decode_move(self, move_index: int) -> str:
        return self.index_to_move.get(move_index, "")
    
    def prepare_training_data(self, df: pd.DataFrame, sample_size: int = None) -> Tuple:
        print("Создание словаря ходов...")
        self.create_move_vocabulary(df['move'].values)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Взята выборка из {sample_size} позиций")
        X_board = []
        X_additional = []
        y = []
        skipped_moves = 0
        processed = 0
        
        for _, row in df.iterrows():
            try:
                move_encoded = self.encode_move(row['move'])
                if move_encoded == -1:
                    skipped_moves += 1
                    continue
                    
                board = chess.Board(row['fen'])
                board_matrix = self.board_to_matrix(board)
                
                additional_features = np.array([
                    int(board.turn),
                    int(board.has_kingside_castling_rights(chess.WHITE)),
                    int(board.has_queenside_castling_rights(chess.WHITE)),
                    int(board.has_kingside_castling_rights(chess.BLACK)),
                    int(board.has_queenside_castling_rights(chess.BLACK)),
                ], dtype=np.float32)
                
                X_board.append(board_matrix)
                X_additional.append(additional_features)
                y.append(move_encoded)
                
                processed += 1
                if processed % 10000 == 0:
                    print(f"Обработано {processed} позиций...")
                
            except Exception as e:
                skipped_moves += 1
                continue
        print(f"Пропущено ходов (редкие/ошибки): {skipped_moves}")
        print(f"Итоговый размер датасета: {len(X_board)}")
        X_board = np.array(X_board)
        X_additional = np.array(X_additional)
        y = np.array(y)
        return X_board, X_additional, y

class ChessModel:
    def __init__(self, move_vocab_size: int):
        self.model = None
        self.move_vocab_size = move_vocab_size
        self.history = None
    
    def build_model(self, board_shape: Tuple[int, int, int] = (8, 8, 12), 
                   additional_features_dim: int = 5) -> keras.Model:
        board_input = keras.Input(shape=board_shape, name='board_input')
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(board_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        additional_input = keras.Input(shape=(additional_features_dim,), name='additional_input')
        combined = layers.concatenate([x, additional_input])
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.move_vocab_size, activation='softmax', name='move_output')(x)
        model = keras.Model(
            inputs=[board_input, additional_input],
            outputs=output
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model
    
    def train(self, X_train_board: np.ndarray, X_train_additional: np.ndarray, 
              y_train: np.ndarray, X_val_board: np.ndarray, 
              X_val_additional: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> None:
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
        ]
        print("Обучение модели")
        self.history = self.model.fit(
            [X_train_board, X_train_additional],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_val_board, X_val_additional], y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    def save_model(self, filepath: str) -> None:
        if self.model:
            self.model.save(filepath)
            print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> None:
        self.model = keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")


class ChessMovePredictor:
    def __init__(self, model: keras.Model, processor: ChessDataProcessor):
        self.model = model
        self.processor = processor
    
    def predict_moves(self, fen_position: str, top_k: int = 5) -> List[Tuple[str, float]]:
        try:
            board = chess.Board(fen_position)
            board_matrix = self.processor.board_to_matrix(board)
            additional_features = np.array([
                int(board.turn),
                int(board.has_kingside_castling_rights(chess.WHITE)),
                int(board.has_queenside_castling_rights(chess.WHITE)),
                int(board.has_kingside_castling_rights(chess.BLACK)),
                int(board.has_queenside_castling_rights(chess.BLACK)),
            ], dtype=np.float32)
            board_input = np.expand_dims(board_matrix, axis=0)
            additional_input = np.expand_dims(additional_features, axis=0)
            predictions = self.model.predict([board_input, additional_input], verbose=0)[0]
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            top_moves = []
            for idx in top_indices:
                move = self.processor.decode_move(idx)
                if move:
                    probability = predictions[idx]
                    top_moves.append((move, probability))
            return top_moves
        except Exception as e:
            print(f"Ошибка предсказания для FEN: {fen_position}, {e}")
            return []
    def get_legal_moves_with_probabilities(self, fen_position: str, top_k: int = 10) -> List[Tuple[str, float]]:
        try:
            board = chess.Board(fen_position)
            all_predictions = self.predict_moves(fen_position, top_k=50)
            
            legal_moves = []
            for move_uci, prob in all_predictions:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        legal_moves.append((move_uci, prob))
                        if len(legal_moves) >= top_k:
                            break
                except:
                    continue
            total_prob = sum(prob for _, prob in legal_moves)
            if total_prob > 0:
                legal_moves = [(move, prob/total_prob) for move, prob in legal_moves]
            return legal_moves
        except Exception as e:
            print(f"Ошибка получения легальных ходов: {e}")
            return []

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model: keras.Model, X_test_board: np.ndarray, 
                      X_test_additional: np.ndarray, y_test: np.ndarray,
                      processor: ChessDataProcessor) -> Dict[str, float]:        
        y_pred_proba = model.predict([X_test_board, X_test_additional], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Top-K accuracy
        top_3_accuracy = ModelEvaluator.top_k_accuracy(y_test, y_pred_proba, k=3)
        top_5_accuracy = ModelEvaluator.top_k_accuracy(y_test, y_pred_proba, k=5)
        
        print("--- метрики качества ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top_3_accuracy': top_3_accuracy,
            'top_5_accuracy': top_5_accuracy
        }
    
    @staticmethod
    def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)


class DataLoader:
    @staticmethod
    def load_data_from_csv(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)
    
    @staticmethod
    def load_data_from_content(content: str) -> pd.DataFrame:
        lines = content.strip().split('\n')[1:]
        data = []
        for line in lines:
            if line.strip():
                parts = line.split('",')
                if len(parts) == 2:
                    fen = parts[0].replace('"', '').strip()
                    move = parts[1].strip()
                    data.append({'fen': fen, 'move': move})
        return pd.DataFrame(data)
    
    @staticmethod
    def prepare_datasets(X_board: np.ndarray, X_additional: np.ndarray, 
                        y: np.ndarray, test_size: float = 0.2, 
                        val_size: float = 0.1) -> Tuple:
        indices = np.arange(len(X_board))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=42
        )
        X_train_board = X_board[train_idx]
        X_train_additional = X_additional[train_idx]
        y_train = y[train_idx]
        
        X_val_board = X_board[val_idx]
        X_val_additional = X_additional[val_idx]
        y_val = y[val_idx]
        
        X_test_board = X_board[test_idx]
        X_test_additional = X_additional[test_idx]
        y_test = y[test_idx]
        
        print(f"Обучающая выборка: {len(y_train)}")
        print(f"Валидационная выборка: {len(y_val)}")
        print(f"Тестовая выборка: {len(y_test)}")
        
        return (X_train_board, X_train_additional, y_train), \
               (X_val_board, X_val_additional, y_val), \
               (X_test_board, X_test_additional, y_test)


def main():
    parser = argparse.ArgumentParser(description='Шахматный предсказатель ходов')
    parser.add_argument('--train', action='store_true', help='Запустить обучение модели')
    parser.add_argument('--predict', type=str, help='FEN позиция для предсказания')
    parser.add_argument('--file', type=str, help='Путь к CSV файлу с данными')
    parser.add_argument('--model_path', type=str, default='chess_model.h5', 
                       help='Путь для сохранения/загрузки модели')
    parser.add_argument('--processor_path', type=str, default='chess_processor.pkl',
                       help='Путь для сохранения/загрузки процессора')
    parser.add_argument('--min_move_count', type=int, default=5,
                       help='Минимальное количество повторений хода для включения в словарь')
    parser.add_argument('--sample_size', type=int, default=50000,
                       help='Размер выборки для обучения (для больших датасетов)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох обучения')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args)
    elif args.predict:
        predict_move(args)
    else:
        print("Используйте --train для обучения или --predict FEN_POSITION для предсказания")


def train_model(args):
    """Функция обучения модели"""
    print("--- загрузка данных ---")
    
    if args.file and os.path.exists(args.file):
        df = DataLoader.load_data_from_csv(args.file)
    else:
        from demo_data import get_demo_data
        df = DataLoader.load_data_from_content(get_demo_data())
    
    print(f"Загружено {len(df)} позиций")
    
    print("--- подготовка данных ---")
    processor = ChessDataProcessor(min_move_count=args.min_move_count)
    X_board, X_additional, y = processor.prepare_training_data(df, sample_size=args.sample_size)
    if len(X_board) == 0:
        print("Нет данных для обучения после фильтрации, уменьшите min_move_count.")
        return
    
    print("--- разделдение данных ---")
    (X_train_board, X_train_add, y_train), \
    (X_val_board, X_val_add, y_val), \
    (X_test_board, X_test_add, y_test) = DataLoader.prepare_datasets(
        X_board, X_additional, y
    )
    
    del X_board, X_additional, y
    gc.collect()
    
    print("--- создание модели ---")
    chess_model = ChessModel(processor.move_vocab_size)
    chess_model.build_model()
    chess_model.model.summary()
    
    print("--- обучение модели ---")
    chess_model.train(
        X_train_board, X_train_add, y_train,
        X_val_board, X_val_add, y_val,
        epochs=args.epochs, batch_size=32
    )
    
    print("--- оценка модели ---")
    metrics = ModelEvaluator.evaluate_model(
        chess_model.model, 
        X_test_board, X_test_add, y_test, 
        processor
    )
    
    print("--- сохранение модели ---")
    chess_model.save_model(args.model_path)
    
    with open(args.processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print("Обучение завершено")


def predict_move(args):
    if not os.path.exists(args.model_path) or not os.path.exists(args.processor_path):
        print("Модель не найдена, необходимо сначала её обучить.")
        return
    print("--- загрузка модели ---")
    chess_model = ChessModel(0)
    chess_model.load_model(args.model_path)
    with open(args.processor_path, 'rb') as f:
        processor = pickle.load(f)
    predictor = ChessMovePredictor(chess_model.model, processor)
    print(f"--- предсказание для позиции ---")
    print(f"FEN: {args.predict}")
    legal_moves = predictor.get_legal_moves_with_probabilities(args.predict, top_k=5)
    board = chess.Board(args.predict)
    print(f"Чей ход: {'Белые' if board.turn else 'Черные'}")
    print("Топ 5 ходов:")
    for i, (move, prob) in enumerate(legal_moves, 1):
        print(f"{i}. {move} (вероятность: {prob:.4f})")

if __name__ == "__main__":
    main()