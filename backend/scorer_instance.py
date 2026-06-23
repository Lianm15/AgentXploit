from compliance_scorer import ComplianceScorer

# Single instance shared across routes and main.
# Loaded once at import time — if the model fails to load,
# ComplianceScorer falls back to heuristic silently and logs a warning.
scorer = ComplianceScorer()
