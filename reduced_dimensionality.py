import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV
import yaml

t_ds = xr.open_dataset("../../inputs/processed_data/training_dataset1.nc")

model_name = "xgb_test"

vars_to_use = ["to", "zos", "mld"]
per_variable_pca = True
n_components = 5


def fit_pca_scores(da, n_components):
    feature_dims = [d for d in da.dims if d != "time"]
    X = da.stack(feature=feature_dims).transpose("time", "feature")
    X = X.dropna("feature", how="any")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    pc = xr.DataArray(
        scores,
        dims=("time", "pc"),
        coords={"time": X["time"], "pc": np.arange(1, pca.n_components_ + 1)},
        name="pc_scores",
    )
    return pc, pca

if per_variable_pca:
    pcs = []
    for var in vars_to_use:
        pc_var, pca_var = fit_pca_scores(t_ds[var], n_components)
        pc_var = pc_var.assign_coords(
            pc=[f"{var}_pc{idx}" for idx in pc_var["pc"].values]
        )
        pcs.append(pc_var)
        print(f"{var} explained variance ratio: {pca_var.explained_variance_ratio_}")

    pc_scores = xr.concat(pcs, dim="pc")
else:
    pca_ds = t_ds[vars_to_use]
    feature_dims = [d for d in pca_ds.dims if d != "time"]
    print("Feature dimensions:", feature_dims)

    X = (
        pca_ds.to_array("variable")
        .stack(feature=("variable", *feature_dims))
        .transpose("time", "feature")
    )

    # Drop any spatial points with NaNs across time before PCA.
    X = X.dropna("feature", how="any")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    pc_scores = xr.DataArray(
        scores,
        dims=("time", "pc"),
        coords={"time": X["time"], "pc": np.arange(1, pca.n_components_ + 1)},
        name="pc_scores",
    )

    loadings = xr.DataArray(
        pca.components_,
        dims=("pc", "feature"),
        coords={"pc": pc_scores["pc"], "feature": X["feature"]},
        name="pc_loadings",
    )

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    loadings_unstacked = loadings.unstack("feature")

xgb_model = xgb.XGBClassifier()

param_space_edges = {
    'objective': ['binary:logistic', 'binary:hinge'],
    'max_depth': [4, 8],
    'learning_rate': [0.005, 0.1],
    'subsample': [0.6, 0.9],
    'n_estimators' : [20, 200],
    # regularisation parameters, should be >0
    'alpha' : [0.1, 0.5],
    'lambda' : [0.1, 0.5],
    'gamma' : [0.1, 0.5],
}

my_search = BayesSearchCV(
            xgb_model,
            param_space_edges,
            n_jobs=-1,
            n_iter = 100
        )

x_train = pc_scores.isel(time=slice(0,400)).values
y_train = t_ds.target.isel(time=slice(0,400)).values

my_search.fit(x_train, y_train)

best_params = my_search.best_params_
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(x_train, y_train)

if model_name is not None:
    print(f"Saving best model trained on x_train to {model_name}.json")
    best_model.save_model(f"outputs/xgboost/{model_name}.json")

print("Best set of hyperparameters: ", my_search.best_params_)
print("Best score: ", my_search.best_score_)

print("test set performance:")
x_test = pc_scores.isel(time=slice(400, None)).values
y_test = t_ds.target.isel(time=slice(400, None)).values
y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:, 1]

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_proba))
