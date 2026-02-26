Proyecto MLOps — Predicción de Pago de Créditos

Sistema de Machine Learning que predice si un cliente pagará su crédito a tiempo utilizando datos históricos financieros.

Objetivo
Anticipar comportamiento de nuevos clientes y mejorar decisiones de riesgo crediticio.
Arquitectura del proyecto
Incluye pipeline completo:
carga de datos
análisis exploratorio
feature engineering
entrenamiento
evaluación
monitoreo
dashboard

Modelos evaluados
Logistic Regression ✅ seleccionado
Random Forest
XGBoost
Criterio de selección: mejor balance entre rendimiento, interpretabilidad y costo computacional.

Métricas modelo final
Accuracy ≈ 95%
F1 Score ≈ 0.97

Monitoreo
Se implementaron métricas de drift:
PSI
KS Test
Jensen–Shannon
Chi-cuadrado
El sistema genera alertas automáticas cuando detecta desviaciones.

Dashboard

Se desarrolló una app Streamlit que permite:
visualizar drift
monitorear variables
detectar anomalías
sugerir reentrenamiento

Ejecución
pip install -r requirements.txt
python src/model_trainig_evaluation.py
python src/model_monitoring.py
streamlit run streamlit_app.py


Resultado final
.Pipeline listo para despliegue en producción con monitoreo continuo.




MLOps Pipeline — Predicción de Pago a Tiempo (Créditos)

1) Objetivo del negocio
Este proyecto simula un caso real de una empresa financiera donde el equipo de Datos y Analítica necesita **anticipar el comportamiento de nuevos usuarios** utilizando información histórica de créditos.

**Problema a resolver:**  
Predecir si un cliente pagará a tiempo un crédito (`Pago_atiempo`), donde:
- `Pago_atiempo = 1` → paga a tiempo
- `Pago_atiempo = 0` → no paga a tiempo

**Impacto esperado:**  
- Reducir el riesgo crediticio.
- Apoyar decisiones de otorgamiento y condiciones (monto, plazo, tasas).
- Identificar señales tempranas de deterioro en la población (drift).

---

2) Alcance técnico (qué se construyó)
El proyecto implementa un flujo completo estilo MLOps:

1. **Carga de datos** (dataset de ejemplo no productivo).
2. **EDA (Exploración y comprensión)**: univariable, bivariable y multivariable.
3. **Feature Engineering** mediante pipelines reproducibles (`ColumnTransformer`).
4. **Entrenamiento y evaluación de varios modelos** (comparación por métricas).
5. **Selección y persistencia del mejor modelo** (`best_model.joblib`).
6. **Monitoreo de data drift** (métricas: KS, PSI, JS, Chi²) + alertas.
7. **Dashboard en Streamlit** para visualizar drift y evolución.
8. **Despliegue del modelo con FastAPI** (endpoint `/predict` batch).
9. **Dockerización** (imagen lista con dependencias + API).
10. **Integración con SonarCloud** (calidad/seguridad/estilo vía GitHub Actions).

---

3) Estructura del repositorio
La estructura se mantiene fija para compatibilidad con pipelines automatizados.

```txt
mlops_pipeline/
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── Comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_trainig_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── Dockerfile
├── streamlit_app.py
├── baseline_profile.json
├── latest_drift_metrics.csv
├── monitoring_log.csv
└── best_model.joblib

4) Gestión del proyecto en Git (ramas y versionado)

El flujo de trabajo se realiza por ramas y merges a la rama estable.
developer: desarrollo de notebooks y scripts.
main: rama estable e integrable a producción.

La integración se hace mediante Pull Request hacia main.

Dataset: descripción general


5) El dataset cuenta con:

Registros: 10,763

Columnas: 23

Tipos: 12 numéricas enteras, 8 numéricas float, 3 tipo object (categóricas/fechas)

Ejemplo de columnas relevantes:
fecha_prestamo (fecha)
tipo_credito (categoría codificada)
tipo_laboral (categoría)
capital_prestado, plazo_meses, cuota_pactada
puntaje, puntaje_datacredito
saldos (mora/total/principal)
promedio_ingresos_datacredito
Target: Pago_atiempo


6) EDA (Exploración y análisis de datos)

Notebook principal:
src/Comprension_eda.ipynb

6.1 Exploración inicial (info, nulos, tipos)
Se encontró presencia de nulos principalmente en:

Columna	Nulos aprox.
tendencia_ingresos	2932
promedio_ingresos_datacredito	2930
saldo_mora_codeudor	590
saldo_principal	405
saldo_mora	156
saldo_total	156
puntaje_datacredito	6

Se recomendó:

Unificar representación de nulos.
Imputación en pipeline (numéricas y categóricas).
Conversión de tipos (por ejemplo, fecha_prestamo a datetime).


6.2 Análisis univariable

Se generaron distribuciones (histogramas) para variables numéricas y conteos para categóricas.

Hallazgos:
Muchas variables presentan alta asimetría (skewness muy alto), por ejemplo:
salario_cliente, total_otros_prestamos, saldo_mora, saldo_mora_codeudor
Se detectaron posibles outliers extremos (kurtosis alta), lo cual sugiere:
necesidad de escalado robusto o transformaciones logarítmicas (opcional)
validaciones de rango para datos productivos
Ejemplo de estadísticas adicionales (calculadas sobre columnas numéricas):
Skewness alto: salario_cliente ~ 43, saldo_mora_codeudor ~ 95
Kurtosis alto: saldo_mora_codeudor ~ 9279 (colas pesadas)

Interpretación:
Hay valores extremos que pueden afectar ciertos modelos sensibles.
Modelos basados en árboles suelen tolerarlos mejor.



6.3 Análisis bivariable (vs target Pago_atiempo)

Se revisaron relaciones con la variable objetivo:
Comparación de distribuciones (por clase).
Variables con señales importantes:
puntaje, puntaje_datacredito
variables de saldos (mora)
comportamiento de ingresos (promedio_ingresos_datacredito)

Hallazgo clave:
El dataset muestra alta concentración de Pago_atiempo=1 (clase mayoritaria), lo cual sugiere revisar:
balance de clases
métricas más informativas (F1, PR-AUC)


6.4 Análisis multivariable

Se generó matriz de correlación y relaciones entre variables.

Hallazgos:
Se identifican correlaciones esperadas:
capital_prestado ↔ cuota_pactada (relación positiva)
saldo_total ↔ saldo_principal (relación alta)
Se detectó que variables no numéricas (ej. fecha_prestamo) requieren conversión o exclusión para correlación.
Se definieron reglas base para validación futura:
Campos monetarios no deben ser negativos (excepto donde aplique).
Edad debe estar dentro de rangos razonables.
Variables de fecha deben ser parseables.


7) Feature Engineering (pipelines reproducibles)

Script:
src/ft_engineering.py
Se construyó un pipeline con ColumnTransformer:
Numéricas: SimpleImputer(strategy="median")
Categóricas nominales: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
Ordinales (si aplica): OrdinalEncoder

Resultado:

Dataset transformado listo para entrenamiento.
Procesamiento consistente tanto en entrenamiento como en inferencia (API).


8) Entrenamiento y evaluación de modelos

Script:
src/model_trainig_evaluation.py

8.1 Modelos entrenados
Se entrenaron y compararon:
Logistic Regression
Random Forest
XGBoost

8.2 Métricas de evaluación

Se midieron (mínimo):
Accuracy
Precision
Recall
F1-score

Resultados obtenidos (ejecución final observada):

Modelo	Accuracy	Precision	Recall	F1
LogisticRegression	0.952624	0.952624	1.000000	0.975737
RandomForest	0.952624	0.952624	1.000000	0.975737
XGBoost	0.951231	0.953825	0.997075	0.974970

Interpretación:

Los modelos muestran métricas altas, influenciadas por posible desbalance (clase mayoritaria).
Se recomienda complementar con ROC-AUC / PR-AUC si se quiere evaluación aún más robusta.


8.3 Persistencia del mejor modelo

El mejor modelo se guarda como:
best_model.joblib

Este archivo se usa en:
API (FastAPI)
monitoreo (para asociar predicciones)
Docker (imagen)


9) Monitoreo de Drift (Data Drift)

Script:
src/model_monitoring.py

9.1 ¿Qué hace el monitoreo?

Crea un perfil baseline con estadísticas históricas.
Simula llegada de nuevos datos.
Calcula métricas de drift por variable:
KS test (numéricas)
PSI (numéricas)
Jensen-Shannon divergence
Chi-cuadrado (categóricas)
Genera alertas si supera umbrales.

9.2 Archivos que genera

baseline_profile.json: perfil histórico (baseline)
latest_drift_metrics.csv: drift actual por variable
monitoring_log.csv: registro temporal de ejecuciones

9.3 Resultado observado

Ejemplo real:
Se detectó drift crítico en promedio_ingresos_datacredito
Mensaje:
“ALERTA: Drift crítico detectado… Recomendación: revisar calidad de datos, umbrales y considerar reentrenamiento.”

Interpretación:

Cambios en ingresos promedio pueden indicar:
nueva población distinta
cambios macroeconómicos
errores de calidad de datos
necesidad de recalibrar o reentrenar


## 10) Dashboard en Streamlit
Archivo: `streamlit_app.py`

La aplicación permite visualizar:

- Métricas de drift por variable
- Alertas de drift crítico
- Evolución temporal del drift
- Comparación distribución histórica vs actual

### Ejecución
python -m streamlit run streamlit_app.py

IMPORTANTE: primero ejecutar monitoreo
python src/model_monitoring.py


## 11) API del modelo (FastAPI)

El modelo se expone mediante una API REST para predicciones batch.

Levantar servidor:

uvicorn src.model_deploy:app --reload

Abrir documentación:
http://127.0.0.1:8000/docs

Endpoint:
POST /predict

Ejemplo de request

{
  "data": [
    {
      "tipo_credito": 4,
      "capital_prestado": 2000000,
      "plazo_meses": 12,
      "edad_cliente": 35,
      "salario_cliente": 3000000,
      "total_otros_prestamos": 1000000,
      "cuota_pactada": 180000,
      "cant_creditosvigentes": 2,
      "huella_consulta": 3,
      "creditos_sectorFinanciero": 2,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 1
    }
  ]
}


## 12) Dockerización

Construir imagen:
docker build -t credit-model .

Ejecutar:
docker run -p 8000:8000 credit-model

Abrir Swagger:
http://127.0.0.1:8000/docs


## 13) Calidad de código (SonarCloud)

Se integró SonarCloud para análisis automático de:

- calidad
- seguridad
- estilo
- mantenibilidad

El análisis se ejecuta automáticamente en cada push y pull request mediante GitHub Actions.

Resultado:
Quality Gate → PASSED

## 14) Resultados y hallazgos

Hallazgos principales:

- Dataset con valores extremos en variables financieras.
- Alta asimetría en ingresos y saldos.
- Variables de mora y score crediticio muestran mayor relación con el target.
- Se detectó drift en promedio_ingresos_datacredito.

Conclusión:
El modelo es útil para predecir comportamiento de pago, pero requiere monitoreo continuo.

## 15) Autor

Proyecto desarrollado por:
[Gabriel Esteban Cardozo Orjuela]

Repositorio:
[https://github.com/Gabriel-Cardozo19/mlops_pipeline.git]