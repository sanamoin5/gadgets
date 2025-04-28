from multiprocessing import freeze_support

from recommendation_engine.experiments.evaluation.sbert_evaluator import test_evaluate_sbert
from recommendation_engine.training.sbert_trainer import train_and_evaluate_sbert

if __name__ == '__main__':
    freeze_support()
    train_and_evaluate_sbert()
    test_evaluate_sbert()