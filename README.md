# 📈 Income Predictor – Prototipo escalable listo para vender

**Stack:** Python + scikit-learn + Streamlit + GitHub (Actions opcional)  
**Caso:** Clasificación binaria con datos públicos (*Adult* – OpenML id=1590): predecir si el ingreso anual > USD 50k.

## 🚀 Demo en 5 minutos

```bash
# 1) Clonar y entrar
git clone <TU_REPO_URL> income-predictor
cd income-predictor

# 2) Crear entorno e instalar
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) (Primera vez) Entrenar el modelo
python train.py

# 4) Correr la app
streamlit run app.py
```

> La primera vez, `train.py` descargará el dataset desde OpenML (internet requerido).  
> Para probar sin internet, podés jugar con `data/sample_adult_small.csv` y ajustar el código para entrenar con ese CSV.

## 🧠 Qué hace
- Descarga y prepara datos públicos reales (*Adult*).
- Compara 3 modelos (LogReg, RandomForest, XGBoost) con **ROC AUC (CV)** y elige el mejor.
- Persiste: `models/best_model.joblib`, `models/metadata.json`, `models/input_schema.json`.
- **Scoreo en tiempo real**: formulario con los campos del esquema.
- **Scoreo por archivo**: subís CSV y descargás `scored.csv` con columna `score`.
- Métricas de hold-out: ROC AUC, F1, Accuracy.

## 📦 Estructura del repo
```
.
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── data/
│   └── sample_adult_small.csv
├── models/
│   └── .gitkeep
├── .github/workflows/train.yml
├── .gitignore
└── LICENSE
```

## 🏭 Cómo “productizar” rápido (checklist)
- Branding: cambia título, agrega logo y colores.
- Dominio: deploy en Streamlit Cloud o Docker.
- Privacidad: banner de consentimiento y política.
- Modelo: agrega LightGBM/CatBoost o GridSearch.
- Monitoreo: métricas + drift con dashboard.
- Retrain: habilitá Actions o un cron propio.
- Ventas: crea un 1‑pager (problema, solución, métricas, CTA).

## ☁️ Deploy (opciones)
1. **Streamlit Community Cloud**.
2. **Hugging Face Spaces**.
3. **Docker básico** (ver README original o agrega un Dockerfile).

## 🔁 Re-entrenos automáticos (GitHub Actions)
Editá la cron en `.github/workflows/train.yml` y asegurate de que el runner tenga internet.

## 📜 Licencia
MIT.