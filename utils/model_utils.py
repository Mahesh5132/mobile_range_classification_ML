import os
import pickle

def save_models(model_dict, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for name, model in model_dict.items():
        with open(os.path.join(model_dir, f"{name}.pkl"), 'wb') as f:
            pickle.dump(model, f)
