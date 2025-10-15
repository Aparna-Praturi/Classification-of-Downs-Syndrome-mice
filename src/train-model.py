import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import os
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def get_model_and_params(trial, model_name):
    if trial is None:
            if model_name == "logreg":
                return LogisticRegression(max_iter=3000, random_state=42), {}

            elif model_name == "rf":
                return RandomForestClassifier(random_state=42), {}

            elif model_name == "xgb":
                return xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"), {}

            elif model_name == "lgbm":
                return lgb.LGBMClassifier(random_state=42), {}

            elif model_name == "svc":
                return SVC(kernel="rbf", probability=True, random_state=42), {}

            elif model_name == "mlp":
                return MLPClassifier(max_iter=300, random_state=42), {}

            else:
                raise ValueError(f"Unknown model name: {model_name}")

    if model_name == "logreg":
        model = LogisticRegression(max_iter=3000, random_state=42)

        combo_choices = [
            "l1__liblinear",
            "l1__saga",
            "l2__lbfgs",
            "l2__liblinear",
            "l2__saga",
            "l2__newton-cg",
        ]

        # Suggest the combo — this is NOT returned as a param
        combo = trial.suggest_categorical("penalty_solver_combo", combo_choices)
        penalty, solver = combo.split("__")

        params = {
            "model__C": trial.suggest_float("model__C", 1e-4, 1e2, log=True),
            "model__penalty": penalty,
            "model__solver": solver,
            "model__class_weight": trial.suggest_categorical("model__class_weight", [None, "balanced"]),
        }


    elif model_name == "rf":
        model = RandomForestClassifier(random_state=42)
        params = {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 500),
            "model__max_depth": trial.suggest_int("model__max_depth", 3, 30),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
        }

    elif model_name == "xgb":
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        params = {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 500),
            "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.3),
            "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
            "model__gamma": trial.suggest_float("model__gamma", 0, 5),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 10.0),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-4, 10.0),
        }

    elif model_name == "lgbm":
        model = lgb.LGBMClassifier(random_state=42)
        params = {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 500),
            "model__max_depth": trial.suggest_int("model__max_depth", -1, 15),
            "model__num_leaves": trial.suggest_int("model__num_leaves", 16, 64),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.3),
            "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 10.0),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-4, 10.0),
        }

    elif model_name == "svc":
        
        params = {
            "model__C": trial.suggest_float("model__C", 1e-3, 1e5, log=True),
            "model__gamma": trial.suggest_float("model__gamma", 1e-5, 1e2, log=True),
            
            
        }
        model = SVC(probability=True, random_state=42, class_weight="balanced",decision_function_shape='ovr')

    elif model_name == "mlp":
        model = MLPClassifier(max_iter=300, random_state=42)
        params = {
            "model__hidden_layer_sizes": trial.suggest_categorical(
                "model__hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)]
            ),
            "model__activation": trial.suggest_categorical("model__activation", ["relu", "tanh"]),
            "model__alpha": trial.suggest_float("model__alpha", 1e-5, 1e-2, log=True),
            "model__learning_rate_init": trial.suggest_float("model__learning_rate_init", 1e-4, 1e-2, log=True),
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, params





def nested_mouseid_cv(X, y, groups, model_name="logreg", n_trials=10, random_state=42, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    results_dir = 'results'
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    saved_models = {}

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        groups_tr = groups.iloc[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)

        def objective(trial):
            model, params = get_model_and_params(trial, model_name)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ('pca', PCA(n_components=0.95, whiten=True, random_state=42)),
                ("model", model)
            ])

            pipe.set_params(**params)

            scores = []
            for inner_train_idx, val_idx in inner_cv.split(X_tr, y_tr, groups=groups_tr):
                X_inner_train, X_val = X_tr.iloc[inner_train_idx], X_tr.iloc[val_idx]
                y_inner_train, y_val = y_tr.iloc[inner_train_idx], y_tr.iloc[val_idx]

                pipe.fit(X_inner_train, y_inner_train)
                preds = pipe.predict(X_val)
                scores.append(f1_score(y_val, preds, average="weighted"))

            return np.mean(scores)

        # Hyperparameter tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_param_importances(study)

        #  Retrain best pipeline on full outer train
        best_model, _ = get_model_and_params(study.best_trial, model_name)
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ('pca', PCA(n_components=0.95, whiten=True, random_state=42)),
            ("model", best_model)
        ])
        best_params_clean = {k: v for k, v in study.best_params.items() if "__" in k}
        best_pipe.set_params(**best_params_clean)
        best_pipe.fit(X_tr, y_tr)

        #  Evaluate
        preds = best_pipe.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")

        results.append({
            "fold": fold,
            "model": model_name,
            "best_params": study.best_params,
            "accuracy": acc,
            "f1_score": f1
        })

        #  Save model (both in memory and on disk)
        saved_models[(model_name, fold)] = best_pipe
        filename = os.path.join(save_dir, f"{model_name}_fold{fold}.joblib")
        joblib.dump(best_pipe, filename)

        
        
        cm = confusion_matrix(y_te, preds, normalize='true')
        print("\nConfusion Matrix:")
        print(cm)
        fig_path = os.path.join(results_dir, f"{model_name}_fold{fold}_confusion_matrix.png")
        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"{model_name}_fold{fold}_Confusion Matrix")
        plt.savefig(fig_path, dpi=300)
    

    results_df = pd.DataFrame(results)
    return results_df, saved_models

def find_best_model(X, y, groups,model_list, output_path, summary_path):

    all_results = []

    for m in model_list:
        print(f"\n=== Running {m.upper()} ===")
        df_res, models = nested_mouseid_cv(X, y, groups, model_name=m, n_trials=30)
        all_results.append(df_res)

    all_results_df = pd.DataFrame(pd.concat(all_results))

    # Summary across folds
    summary = all_results_df.groupby("model")[["accuracy","f1_score"]].agg(["mean","std"])
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    print(summary)

    best_model = summary['f1_score_mean'].idxmax()
    print("Best model based on F1 score:", best_model)


    try:
        all_results_df.to_csv(output_path, index=False) 
        print(f"\nSuccessfully exported processed data to: {output_path}")

        summary.to_csv(summary_path, index=True)
     

    except Exception as e:
        print(f"\nError exporting data: {e}")

    return best_model



def train_best_model(model_name, X, y, mouse_groups, save_path):

    def objective(trial):
       
        # model_name = best_model.__class__.__name__.lower()
        # print(model_name)
        model, params = get_model_and_params(trial, model_name)
        
        pipe = Pipeline([
                ("scaler", StandardScaler()),
                ('pca', PCA(n_components=0.95, whiten=True, random_state=42)),
                ("model", model)
            ])

        pipe.set_params(**params)

        # Cross-validation setup
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, groups=mouse_groups, scoring="f1_weighted")

        return scores.mean()

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params_clean = {k: v for k, v in study.best_params.items() if "__" in k}
   
    print("Best hyperparameters:", study.best_params)
    print("Best CV F1 score:", study.best_value)

    # Train final model on full dataset
    final_model, _ = get_model_and_params(None, model_name)
    
    best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ('pca', PCA(n_components=0.95, whiten=True, random_state=42)),
            ("model", final_model)
        ])
    best_params_clean = {k: v for k, v in study.best_params.items() if "__" in k}
    best_pipe.set_params(**best_params_clean)
    best_pipe.fit(X, y)

    trained_model = best_pipe.named_steps["model"]
   
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "best_pipeline.joblib")
    joblib.dump(best_pipe, file_path)

    print(f" Final {model_name.upper()} model saved to {file_path}")
           
    joblib.dump(trained_model, os.path.join(save_path, "best_trained_model.joblib"))

    # # 5Save final trained model
    # joblib.dump(final_model, save_path)
    print(f" Final model saved to {save_path}")

    y_pred = best_pipe.predict(X)
    cm = confusion_matrix(y, y_pred,  normalize='true')
    print("\nConfusion Matrix:")
    print(cm)
    fig_path = os.path.join(save_path, f"confusion_matrix.png")
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - ")
    plt.savefig(fig_path, dpi=300)
   

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return trained_model, study


def final_pipeline():
    SCRIPT_DIR = Path(__file__).resolve().parent

    data_path = SCRIPT_DIR.parent/'data'/'processed_data.csv'
    
    output_path = SCRIPT_DIR.parent /'results'/'all_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = SCRIPT_DIR.parent /'results'/'summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    saved_models_path = SCRIPT_DIR.parent /'saved_models'

    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")

    # Define X, y

    df['MouseID_cleaned'] = df['MouseID'].str.split("_").str[0]

    X = df.drop(['class',"MouseID", "Genotype", "Treatment", 'Behavior',
                        'MouseID_cleaned'], axis=1)
    y_raw = df['class']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    try:
        label_encoder_path = os.path.join(saved_models_path, "label_encoder.joblib")
        joblib.dump(le, label_encoder_path)
        print(f'saved encoder in {label_encoder_path}')
    except:
        print('couldnt save encoder')

    y = pd.Series(y)
    mouse_groups = df["MouseID_cleaned"].copy()

    model_list = ["logreg", "rf", "xgb", "lgbm", "svc", "mlp"]

    best_model = find_best_model(X, y, mouse_groups, model_list, output_path, summary_path)
   
    final_model = train_best_model(best_model, X, y, mouse_groups=mouse_groups, save_path=saved_models_path)

    return




if __name__ == "__main__":
    final_pipeline() 













