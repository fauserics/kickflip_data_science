name: retrain-classifier

on:
  schedule:
    - cron: "0 9 * * 1"         # Lunes 09:00 UTC (ajusta a gusto)
  workflow_dispatch: {}          # Permite lanzarlo manualmente

permissions:
  contents: write                # necesario para git push

concurrency:
  group: retrain-classifier      # evita carreras si se lanza dos veces
  cancel-in-progress: false

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: true   # usa GITHUB_TOKEN para pushear

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install --prefer-binary -r requirements.txt

      - name: Train classifier
        run: python train.py

      # Sube como artifacts por si querés descargarlos desde la UI de Actions (opcional)
      - name: Upload classifier artifacts (optional)
        uses: actions/upload-artifact@v4
        with:
          name: classifier-artifacts
          path: |
            models/best_model.joblib
            models/metadata.json
            models/input_schema.json
          if-no-files-found: ignore

      - name: Commit & Push artifacts
        run: |
          set -e
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          # Si .gitignore ignora 'models/*', asegurate de permitir .joblib/.json o usa -f:
          git add models/best_model.joblib models/metadata.json models/input_schema.json || true
          git diff --cached --quiet || CHANGED=1
          if [ "${CHANGED:-0}" -eq 1 ]; then
            git commit -m "Auto: retrain classifier"
            git pull --rebase origin ${{ github.ref_name }} || true
            git push origin HEAD:${{ github.ref_name }}
          else
            echo "No changes"
          fi
