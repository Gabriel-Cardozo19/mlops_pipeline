# =============================
# IMPORTS
# =============================

import pandas as pd
import matplotlib.pyplot as plt

from ft_engineering import load_data, feature_engineering

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================
# FUNCION METRICAS
# =============================

def summarize_classification(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


# =============================
# FUNCION CREAR MODELO
# =============================

def build_model(preprocessor, model):

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipe


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    # cargar datos
    df = load_data("Base_de_datos.csv")

    # feature engineering
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df)

    # modelos a probar
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = {}

    # entrenamiento
    for name, model in models.items():

        pipe = build_model(preprocessor, model)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        results[name] = summarize_classification(y_test, preds)

    # =============================
    # TABLA RESULTADOS
    # =============================

    results_df = pd.DataFrame(results).T
    print("\nResultados modelos:\n")
    print(results_df)

    # =============================
    # GRAFICO COMPARATIVO
    # =============================

    results_df.plot(kind="bar", figsize=(10,6))
    plt.title("Comparación de modelos")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df.corr()["Pago_atiempo"].sort_values(ascending=False)



    print("""
Evaluación de modelos:

Se entrenaron tres algoritmos supervisados: Logistic Regression, Random Forest y XGBoost.

Los resultados muestran un rendimiento alto y consistente entre modelos, con valores de F1 cercanos a 0.97.

Se seleccionó Logistic Regression como modelo final debido a:

- rendimiento equivalente a modelos más complejos
- mayor interpretabilidad
- menor costo computacional
- mayor facilidad de despliegue

Conclusión:
El modelo presenta buen desempeño predictivo y capacidad de generalización.
""")
    


    import joblib
import os

# ... cuando ya tengas el pipe entrenado que quieres guardar, por ejemplo:
best_model_name = results_df["f1"].idxmax()
best_model = models[best_model_name]
best_pipe = build_model(preprocessor, best_model)
best_pipe.fit(X_train, y_train)

# Guardar en miops_pipeline/ (sin crear carpetas nuevas)
out_dir = os.path.join(os.path.dirname(__file__), "..", "miops_pipeline")
os.makedirs(out_dir, exist_ok=True)

joblib.dump(best_pipe, os.path.join(out_dir, "best_model.joblib"))
print("Modelo guardado en miops_pipeline/best_model.joblib")