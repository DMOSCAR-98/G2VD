# Utility Scripts

These scripts provide lightweight checks for a local release checkout.

## `check_environment.py`

Checks whether the main G2VD runtime packages can be imported. This is a quick
environment sanity check after installing `requirements.txt`. It does not load
datasets, pretrained weights, or checkpoints.

## `check_config_templates.py`

Validates the YAML templates under `configs/templates/`. It checks that the
required train/test keys are present, but it does not launch training or verify
that your local dataset paths exist.

## `audit_release.py`

Scans the repository for obvious private paths, credential-like strings, and
unexpected large files. It is intended as a release hygiene check before
publishing or sharing a fork.

Run all checks with:

```bash
python scripts/check_environment.py
python scripts/check_config_templates.py
python scripts/audit_release.py
```
