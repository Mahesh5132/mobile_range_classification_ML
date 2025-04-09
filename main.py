from config import TRAIN_FILE, TEST_FILE, MODEL_DIR
from data.data_loader import load_data, inspect_data
from preprocessing.preprocessing import preprocess_data, split_data
from models.train_models import get_classic_models, train_classic_models, train_xgboost, train_softmax
from utils.model_utils import save_models

if __name__ == "__main__":
    df_train, df_test = load_data(TRAIN_FILE, TEST_FILE)
    inspect_data(df_train)

    X_scaled, y, scaler = preprocess_data(df_train)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    models = get_classic_models()
    trained_models = train_classic_models(models, X_train, X_test, y_train, y_test)

    # Add XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    trained_models["xgboost_model"] = xgb_model

    # Add Multinomial Softmax
    mcl_model = train_softmax(X_train, y_train, X_test, y_test)
    trained_models["multinomial_model"] = mcl_model

    # Add scaler
    trained_models["scaler"] = scaler

    save_models(trained_models, MODEL_DIR)
