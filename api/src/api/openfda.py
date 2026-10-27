import re

import httpx

URL = "https://api.fda.gov/drug/label.json"

# Label fields worth answering questions from (Rx and OTC labels use different ones).
# Packaging, ingredient lists and tables are left out: noise for retrieval.
SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "purpose",
    "active_ingredient",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "do_not_use",
    "warnings_and_cautions",
    "warnings",
    "precautions",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "when_using",
    "stop_use",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "pregnancy",
    "pregnancy_or_breast_feeding",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
    "storage_and_handling",
)


def fetch_label(client: httpx.Client, generic_name: str) -> dict | None:
    """Newest single-ingredient label for a generic name.

    A name search also matches combinations ("amlodipine" -> "amlodipine and benazepril"),
    so skip names that look like combinations, then take the shortest one.
    """
    r = client.get(
        URL,
        params={
            "search": f'openfda.generic_name:"{generic_name}"',
            "sort": "effective_time:desc",
            "limit": 50,
        },
    )
    if r.status_code == 404:  # openFDA answers 404 when nothing matches
        return None
    r.raise_for_status()
    labels = [x for x in r.json()["results"] if x.get("openfda", {}).get("generic_name")]

    def rank(label: dict) -> tuple[bool, int]:
        name = label["openfda"]["generic_name"][0]
        return bool(re.search(r" and |/|,", name, re.IGNORECASE)), len(name)

    return min(labels, key=rank, default=None)
