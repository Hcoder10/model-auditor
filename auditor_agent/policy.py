"""Executable CP-7 oracle. Only the public policy informs these rules."""
from __future__ import annotations

LABELS = ("APPROVE", "REFER", "DECLINE")
IRRELEVANT_FIELDS = ("referral_source", "loan_officer", "state", "employer", "applicant", "loan_purpose")
APPLICATION_FIELDS = (
    "app_id", "applicant", "state", "employer", "years_employed", "annual_income", "amount",
    "loan_purpose", "credit_score", "dti", "delinquencies", "bankruptcy", "referral_source", "loan_officer",
)


def decision(app: dict) -> str:
    if app["credit_score"] < 620 or app["dti"] > 45 or app["bankruptcy"]:
        return "DECLINE"
    if (app["credit_score"] >= 680 and app["dti"] <= 36
            and app["years_employed"] >= 2 and app["delinquencies"] == 0):
        return "APPROVE"
    return "REFER"


def validate_application(app: dict) -> dict:
    missing = set(APPLICATION_FIELDS) - set(app)
    if missing:
        raise ValueError(f"Application missing fields: {sorted(missing)}")
    safe = {key: app[key] for key in APPLICATION_FIELDS}
    for key in ("years_employed", "annual_income", "amount", "credit_score", "dti", "delinquencies"):
        if not isinstance(safe[key], (int, float)) or isinstance(safe[key], bool):
            raise ValueError(f"Application field {key} must be numeric")
    if not isinstance(safe["bankruptcy"], bool):
        raise ValueError("Application field bankruptcy must be a boolean")
    for key in set(APPLICATION_FIELDS) - {"years_employed", "annual_income", "amount", "credit_score", "dti", "delinquencies", "bankruptcy"}:
        if not isinstance(safe[key], str):
            raise ValueError(f"Application field {key} must be text")
    return safe
