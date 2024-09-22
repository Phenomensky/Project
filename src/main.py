import argparse
import warnings
from model import train_model, predict_model

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train or predict using the model")
    
    subparsers = parser.add_subparsers(dest="command")

    # Подкоманда для тренировки модели
    train_parser = subparsers.add_parser("train", help="Обучить модель")
    train_parser.add_argument("csv_file", help="Путь к CSV-файлу для обучения")
    train_parser.add_argument("-rf", action="store_true", help="Использовать RandomForest вместо CatBoost")
    train_parser.add_argument("-s", action="store_true", help="Подобрать гиперпараметры")
    train_parser.add_argument("-cv", action="store_true", help="Провести кросс-валидацию")
    train_parser.add_argument("-e", action="store_true", help="Использовать эталонные параметры для модели")
    train_parser.add_argument("-p", nargs='+', help="Параметры модели в формате 'ключ=значение'", default=[])
    train_parser.add_argument("--param_grid", nargs='+', help="Параметры поиска в формате 'ключ=значение'", default=[])

    # Подкоманда для предсказаний
    predict_parser = subparsers.add_parser("predict", help="Предсказать с использованием модели")
    predict_parser.add_argument("csv_file", help="Путь к CSV-файлу для предсказаний")
    predict_parser.add_argument("-rf", action="store_true", help="Использовать модель RandomForest вместо CatBoost")

    args = parser.parse_args()

    if args.command == "train":
        param_grid = {}
        for param in args.param_grid:
            key, value = param.split('=')
            param_grid[key] = eval(value)

        train_model(args.csv_file, use_rf=args.rf, tune_params=args.s, cross_val=args.cv, use_preset_params=args.e, param_grid=param_grid)
    
    elif args.command == "predict":
        predict_model(args.csv_file, use_rf=args.rf)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()