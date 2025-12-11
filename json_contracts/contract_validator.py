import json

def validate_contract(data, contract_path="json_contract/audit_report_contract.json"):
    """
    Validate the generated report JSON against the formal contract.
    
    Returns:
        missing_fields: list of dotted paths describing missing keys
    """
    with open(contract_path, "r") as f:
        contract = json.load(f)

    missing = []

    def _check(d, c, path=""):
        if isinstance(c, dict):
            if not isinstance(d, dict):
                missing.append(path.rstrip("."))
                return
            for key in c:
                if key not in d:
                    missing.append(path + key)
                else:
                    _check(d[key], c[key], path + key + ".")
        elif isinstance(c, list):
            if not isinstance(d, list):
                missing.append(path.rstrip("."))
        else:
            pass  # contract uses human-readable descriptors, not actual types

    _check(data, contract)
    return missing


if __name__ == "__main__":
    # Manual test
    sample_json_path = "audit_output.json"
    with open(sample_json_path, "r") as f:
        data = json.load(f)
    missing = validate_contract(data)
    if missing:
        print("Contract validation failed:")
        for m in missing:
            print(" -", m)
    else:
        print("JSON contract valid.")
