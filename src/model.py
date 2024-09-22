import joblib
import pandas as pd
import os
from process import clean_data
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

# Эталонные параметры для CatBoost
CATBOOST_PRESET_PARAMS = {
    'learning_rate': 0.2,
    'l2_leaf_reg': 5,
    'iterations': 500,
    'depth': 10,
    'bagging_temperature': 0.5
}

# Эталонные параметры для RandomForest
RF_PRESET_PARAMS = {
    'max_depth': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 200
}

# Папка для хранения моделей и кодировщиков
MODEL_DIR = './models'

def get_encoder_file(model_name):
    """Возвращает имя файла для кодировщика, связанного с конкретной моделью."""
    return os.path.join(MODEL_DIR, f"preprocessor_{model_name}.pkl")

def get_target_scaler_file(model_name):
    """Возвращает имя файла для сохранения TargetScaler для каждой модели."""
    return os.path.join(MODEL_DIR, f"target_scaler_{model_name}.pkl")

def preprocess_data(df, model_name, train=True):
    onehot_columns = ['status', 'propertyType', 'stories_numeric', 'baths_category', 'beds_category', 'Year built']
    label_columns = ['city', 'state', 'street_type']
    numerical_columns = ['sqft_numeric', 'lotsize_numeric', 'Price/sqft_numeric']
    passthrough_columns = [
        'fireplace', 'private_pool', 'Remodeled year', 'Heating',
        'Cooling', 'Parking', 'High_Rated_School_Nearby',
        'Low_Rated_School_Nearby', 'Has_Primary_School', 'Has_Middle_School',
        'Has_High_School', 'Weighted_Rating'
    ]
    target_scaler = QuantileTransformer()

    if train:
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_columns),
                ('ordinal', OrdinalEncoder(), label_columns),
                ('scaler', QuantileTransformer(), numerical_columns),
                ('passthrough', 'passthrough', passthrough_columns)
            ]
        )

        X = df.drop(columns=['target_cleaned'])
        y = df['target_cleaned'].values.reshape(-1, 1)

        y_scaled = target_scaler.fit_transform(y)

        X_preprocessed = preprocessor.fit_transform(X)

        # Сохраняем кодировщики и трансформеры для предсказаний
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(preprocessor, get_encoder_file(model_name))
        joblib.dump(target_scaler, get_target_scaler_file(model_name))

    else:
        # Загружаем кодировщики и трансформеры, сохранённые при обучении
        preprocessor = joblib.load(get_encoder_file(model_name))
        target_scaler = joblib.load(get_target_scaler_file(model_name))
        
        X_preprocessed = preprocessor.transform(df)
        y_scaled = None

    return X_preprocessed, y_scaled

def train_model(csv_file, use_rf=False, tune_params=False, cross_val=False, use_preset_params=False, custom_params=None, param_grid=None):
    print("Загрузка данных...", flush=True)
    data = pd.read_csv(csv_file)
    data = clean_data(data)
    X, y = preprocess_data(data, 'rf' if use_rf else 'cb')

    print("Начало обучения модели...", flush=True)
    if use_rf:
        model = RandomForestRegressor()
        if use_preset_params:
            print("Используются эталонные параметры для RandomForest.", flush=True)
            model.set_params(**RF_PRESET_PARAMS)
        elif custom_params:
            print("Используются переданные параметры для RandomForest.", flush=True)
            model.set_params(**custom_params)
    else:
        if use_preset_params:
            print("Используются эталонные параметры для CatBoost.", flush=True)
            model = CatBoostRegressor(**CATBOOST_PRESET_PARAMS, verbose=0)
        elif custom_params:
            print("Используются переданные параметры для CatBoost.", flush=True)
            model = CatBoostRegressor(**custom_params, verbose=0)
        else:
            model = CatBoostRegressor(verbose=0)

    # Поиск параметров
    if tune_params and not use_preset_params:
        print("Начало поиска лучших параметров...", flush=True)
        if param_grid:
            param_grid_eval = param_grid
        else:
            param_grid_eval = {'n_estimators': [100, 200], 'max_depth': [10, 20]} if use_rf else {'iterations': [100, 200], 'depth': [6, 10]}
        
        grid = GridSearchCV(model, param_grid_eval, cv=5)
        model = grid.fit(X, y).best_estimator_
        print(f"Лучшие параметры найдены: {grid.best_params_}", flush=True)
    else:
        model.fit(X, y)
        print("Обучение завершено.", flush=True)

    # Кросс-валидация
    if cross_val:
        print("Начало кросс-валидации...", flush=True)
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation scores: {scores}", flush=True)

    # Сохранение модели
    model_file = os.path.join(MODEL_DIR, f"model_rf.pkl" if use_rf else f"model.pkl")
    joblib.dump(model, model_file)
    print(f"Модель сохранена в {model_file}", flush=True)

def predict_model(csv_file, use_rf=False):
    print("Загрузка данных для предсказания...", flush=True)
    original_data = pd.read_csv(csv_file)
    data = original_data.copy()
    data = clean_data(data, train=False)
    data, _ = preprocess_data(data, 'rf' if use_rf else 'cb', train=False)

    # Загрузка модели
    model_file = os.path.join(MODEL_DIR, f"model_rf.pkl" if use_rf else f"model.pkl")
    print(f"Загрузка модели из {model_file}...", flush=True)
    model = joblib.load(model_file)

    # Загрузка сохранённого скейлера целевой переменной
    print("Загрузка целевого скейлера...", flush=True)
    target_scaler = joblib.load(get_target_scaler_file('rf' if use_rf else 'cb'))
    
    print("Выполнение предсказания...", flush=True)
    predictions_scaled = model.predict(data)

    # Обратная трансформация предсказаний (в тысячи долларов)
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    original_data['predictions'] = predictions
    
    # Сохранение результатов
    result_file_path = os.path.join(os.path.dirname(csv_file), f"predictions.csv")
    original_data.to_csv(result_file_path, index=False)
    print(f"Предсказания сохранены в {result_file_path}", flush=True)
