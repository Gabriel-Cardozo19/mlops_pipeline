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
Pipeline listo para despliegue en producción con monitoreo continuo.