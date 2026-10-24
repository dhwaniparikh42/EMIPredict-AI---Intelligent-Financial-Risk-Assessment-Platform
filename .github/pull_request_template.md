## What does this PR do?
<!-- one or two sentences -->

## Type
- [ ] Data / pipeline   - [ ] EDA / features   - [ ] Classification   - [ ] Regression   - [ ] App   - [ ] Docs / CI

## Checklist
- [ ] `python -m pytest -q` passes locally
- [ ] No data files, `mlflow.db` or `mlartifacts/` committed
- [ ] Anything learned (scaler / encoder / imputer) is fitted on **train only**
- [ ] Test split used only for the final report
- [ ] MLflow runs logged to the shared server (if this PR trains models)
- [ ] README / docs updated if behaviour changed
