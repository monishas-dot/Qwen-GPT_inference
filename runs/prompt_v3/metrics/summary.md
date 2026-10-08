# Extraction Metrics Report

## Ground Truth Overview

| Metric | Value |
|---|---|
| Total annotated files (GT) | 129 |
| Top-level fields (schema) | 19 |
| Scoring keys per file (sub-fields expanded) | 22 |
| Total (file x top-level field) instances | 2451 |
| Total (file x scoring key) instances | 2838 |

### Field-instance counts by GT severity (all top-level fields pooled)

| Severity | Count |
|---|---|
| none | 1331 |
| low | 744 |
| medium | 275 |
| high | 101 |

## Model: Qwen

| Metric | Value |
|---|---|
| Files scored | 125 |
| Excluded (id missing in file) | 0 |
| Excluded (id not in GT) | 3 |
| Excluded (bad status/errors) | 4 |
| Field instances w/ no GT severity (skipped) | 0 |

### Overall accuracy, all fields pooled, by severity

| Severity | n | Strict correct | Strict % | Weighted % |
|---|---|---|---|---|
| none | 1408 | 807 | 57.3% | 57.6% |
| low | 918 | 314 | 34.2% | 35.6% |
| medium | 308 | 88 | 28.6% | 30.1% |

### Per-field accuracy by severity

| Field | Severity | n | Strict correct | Strict % | Weighted % |
|---|---|---|---|---|---|
| acuity | none | 78 | 43 | 55.1% | 55.1% |
| acuity | low | 33 | 19 | 57.6% | 57.6% |
| acuity | medium | 12 | 4 | 33.3% | 33.3% |
| date_of_birth | none | 76 | 51 | 67.1% | 67.1% |
| date_of_birth | low | 36 | 17 | 47.2% | 47.2% |
| date_of_birth | medium | 10 | 5 | 50.0% | 50.0% |
| diagnoses.diagnosisCode | none | 57 | 20 | 35.1% | 35.1% |
| diagnoses.diagnosisCode | low | 49 | 13 | 26.5% | 26.5% |
| diagnoses.diagnosisCode | medium | 10 | 1 | 10.0% | 15.0% |
| lob | none | 91 | 38 | 41.8% | 41.8% |
| lob | low | 28 | 14 | 50.0% | 50.0% |
| lob | medium | 4 | 1 | 25.0% | 25.0% |
| member_id | none | 73 | 48 | 65.8% | 65.8% |
| member_id | low | 35 | 10 | 28.6% | 28.6% |
| member_id | medium | 13 | 1 | 7.7% | 7.7% |
| member_name | none | 58 | 47 | 81.0% | 81.0% |
| member_name | low | 41 | 33 | 80.5% | 80.5% |
| member_name | medium | 17 | 7 | 41.2% | 41.2% |
| place_of_service | none | 95 | 43 | 45.3% | 45.3% |
| place_of_service | low | 22 | 5 | 22.7% | 22.7% |
| place_of_service | medium | 6 | 2 | 33.3% | 33.3% |
| procedures.dateOfServiceEnd | none | 35 | 12 | 34.3% | 36.7% |
| procedures.dateOfServiceEnd | low | 67 | 10 | 14.9% | 18.7% |
| procedures.dateOfServiceEnd | medium | 17 | 2 | 11.8% | 14.2% |
| procedures.dateOfServiceStart | none | 35 | 13 | 37.1% | 39.5% |
| procedures.dateOfServiceStart | low | 67 | 17 | 25.4% | 30.4% |
| procedures.dateOfServiceStart | medium | 17 | 5 | 29.4% | 33.8% |
| procedures.numberOfVisits | none | 35 | 19 | 54.3% | 57.6% |
| procedures.numberOfVisits | low | 67 | 36 | 53.7% | 59.7% |
| procedures.numberOfVisits | medium | 17 | 5 | 29.4% | 40.2% |
| procedures.procedureCode | none | 35 | 20 | 57.1% | 59.5% |
| procedures.procedureCode | low | 67 | 24 | 35.8% | 39.8% |
| procedures.procedureCode | medium | 17 | 2 | 11.8% | 18.6% |
| requesting_provider_address | none | 85 | 53 | 62.4% | 62.4% |
| requesting_provider_address | low | 24 | 9 | 37.5% | 37.5% |
| requesting_provider_address | medium | 10 | 5 | 50.0% | 50.0% |
| requesting_provider_fax | none | 59 | 34 | 57.6% | 57.6% |
| requesting_provider_fax | low | 43 | 6 | 14.0% | 14.0% |
| requesting_provider_fax | medium | 18 | 5 | 27.8% | 27.8% |
| requesting_provider_name | none | 56 | 40 | 71.4% | 71.4% |
| requesting_provider_name | low | 42 | 21 | 50.0% | 50.0% |
| requesting_provider_name | medium | 18 | 8 | 44.4% | 44.4% |
| requesting_provider_npi | none | 66 | 33 | 50.0% | 50.0% |
| requesting_provider_npi | low | 38 | 9 | 23.7% | 23.7% |
| requesting_provider_npi | medium | 17 | 2 | 11.8% | 11.8% |
| requesting_provider_phone | none | 55 | 37 | 67.3% | 67.3% |
| requesting_provider_phone | low | 45 | 13 | 28.9% | 28.9% |
| requesting_provider_phone | medium | 19 | 4 | 21.1% | 21.1% |
| servicing_provider_address | none | 68 | 47 | 69.1% | 69.1% |
| servicing_provider_address | low | 35 | 11 | 31.4% | 31.4% |
| servicing_provider_address | medium | 16 | 5 | 31.2% | 31.2% |
| servicing_provider_fax | none | 77 | 57 | 74.0% | 74.0% |
| servicing_provider_fax | low | 32 | 5 | 15.6% | 15.6% |
| servicing_provider_fax | medium | 11 | 5 | 45.5% | 45.5% |
| servicing_provider_name | none | 61 | 47 | 77.0% | 77.0% |
| servicing_provider_name | low | 40 | 23 | 57.5% | 57.5% |
| servicing_provider_name | medium | 16 | 8 | 50.0% | 50.0% |
| servicing_provider_npi | none | 65 | 36 | 55.4% | 55.4% |
| servicing_provider_npi | low | 38 | 8 | 21.1% | 21.1% |
| servicing_provider_npi | medium | 18 | 2 | 11.1% | 11.1% |
| servicing_provider_phone | none | 75 | 49 | 65.3% | 65.3% |
| servicing_provider_phone | low | 32 | 9 | 28.1% | 28.1% |
| servicing_provider_phone | medium | 14 | 8 | 57.1% | 57.1% |
| type_of_service | none | 73 | 20 | 27.4% | 27.4% |
| type_of_service | low | 37 | 2 | 5.4% | 5.4% |
| type_of_service | medium | 11 | 1 | 9.1% | 9.1% |

## Model: GPT

| Metric | Value |
|---|---|
| Files scored | 129 |
| Excluded (id missing in file) | 0 |
| Excluded (id not in GT) | 3 |
| Excluded (bad status/errors) | 0 |
| Field instances w/ no GT severity (skipped) | 0 |

### Overall accuracy, all fields pooled, by severity

| Severity | n | Strict correct | Strict % | Weighted % |
|---|---|---|---|---|
| none | 1436 | 1098 | 76.5% | 76.8% |
| low | 954 | 555 | 58.2% | 59.7% |
| medium | 329 | 157 | 47.7% | 49.9% |

### Per-field accuracy by severity

| Field | Severity | n | Strict correct | Strict % | Weighted % |
|---|---|---|---|---|---|
| acuity | none | 80 | 65 | 81.2% | 81.2% |
| acuity | low | 34 | 25 | 73.5% | 73.5% |
| acuity | medium | 13 | 6 | 46.2% | 46.2% |
| date_of_birth | none | 77 | 61 | 79.2% | 79.2% |
| date_of_birth | low | 39 | 20 | 51.3% | 51.3% |
| date_of_birth | medium | 10 | 4 | 40.0% | 40.0% |
| diagnoses.diagnosisCode | none | 58 | 46 | 79.3% | 84.5% |
| diagnoses.diagnosisCode | low | 50 | 32 | 64.0% | 69.7% |
| diagnoses.diagnosisCode | medium | 11 | 3 | 27.3% | 45.5% |
| lob | none | 95 | 52 | 54.7% | 54.7% |
| lob | low | 28 | 20 | 71.4% | 71.4% |
| lob | medium | 4 | 2 | 50.0% | 50.0% |
| member_id | none | 74 | 46 | 62.2% | 62.2% |
| member_id | low | 37 | 11 | 29.7% | 29.7% |
| member_id | medium | 14 | 6 | 42.9% | 42.9% |
| member_name | none | 59 | 56 | 94.9% | 94.9% |
| member_name | low | 42 | 31 | 73.8% | 73.8% |
| member_name | medium | 19 | 8 | 42.1% | 42.1% |
| place_of_service | none | 97 | 43 | 44.3% | 44.3% |
| place_of_service | low | 24 | 1 | 4.2% | 4.2% |
| place_of_service | medium | 6 | 1 | 16.7% | 16.7% |
| procedures.dateOfServiceEnd | none | 35 | 31 | 88.6% | 89.3% |
| procedures.dateOfServiceEnd | low | 70 | 40 | 57.1% | 59.7% |
| procedures.dateOfServiceEnd | medium | 18 | 10 | 55.6% | 56.5% |
| procedures.dateOfServiceStart | none | 35 | 29 | 82.9% | 84.5% |
| procedures.dateOfServiceStart | low | 70 | 44 | 62.9% | 66.2% |
| procedures.dateOfServiceStart | medium | 18 | 9 | 50.0% | 59.3% |
| procedures.numberOfVisits | none | 35 | 30 | 85.7% | 85.7% |
| procedures.numberOfVisits | low | 70 | 55 | 78.6% | 83.0% |
| procedures.numberOfVisits | medium | 18 | 8 | 44.4% | 56.7% |
| procedures.procedureCode | none | 35 | 32 | 91.4% | 93.8% |
| procedures.procedureCode | low | 70 | 46 | 65.7% | 72.3% |
| procedures.procedureCode | medium | 18 | 11 | 61.1% | 67.8% |
| requesting_provider_address | none | 87 | 81 | 93.1% | 93.1% |
| requesting_provider_address | low | 25 | 18 | 72.0% | 72.0% |
| requesting_provider_address | medium | 11 | 7 | 63.6% | 63.6% |
| requesting_provider_fax | none | 61 | 48 | 78.7% | 78.7% |
| requesting_provider_fax | low | 43 | 19 | 44.2% | 44.2% |
| requesting_provider_fax | medium | 20 | 9 | 45.0% | 45.0% |
| requesting_provider_name | none | 57 | 52 | 91.2% | 91.2% |
| requesting_provider_name | low | 43 | 30 | 69.8% | 69.8% |
| requesting_provider_name | medium | 20 | 12 | 60.0% | 60.0% |
| requesting_provider_npi | none | 68 | 46 | 67.6% | 67.6% |
| requesting_provider_npi | low | 39 | 13 | 33.3% | 33.3% |
| requesting_provider_npi | medium | 18 | 5 | 27.8% | 27.8% |
| requesting_provider_phone | none | 56 | 39 | 69.6% | 69.6% |
| requesting_provider_phone | low | 46 | 23 | 50.0% | 50.0% |
| requesting_provider_phone | medium | 21 | 8 | 38.1% | 38.1% |
| servicing_provider_address | none | 69 | 63 | 91.3% | 91.3% |
| servicing_provider_address | low | 35 | 28 | 80.0% | 80.0% |
| servicing_provider_address | medium | 18 | 10 | 55.6% | 55.6% |
| servicing_provider_fax | none | 79 | 69 | 87.3% | 87.3% |
| servicing_provider_fax | low | 34 | 21 | 61.8% | 61.8% |
| servicing_provider_fax | medium | 11 | 8 | 72.7% | 72.7% |
| servicing_provider_name | none | 62 | 58 | 93.5% | 93.5% |
| servicing_provider_name | low | 42 | 28 | 66.7% | 66.7% |
| servicing_provider_name | medium | 16 | 9 | 56.2% | 56.2% |
| servicing_provider_npi | none | 66 | 45 | 68.2% | 68.2% |
| servicing_provider_npi | low | 40 | 12 | 30.0% | 30.0% |
| servicing_provider_npi | medium | 19 | 4 | 21.1% | 21.1% |
| servicing_provider_phone | none | 76 | 66 | 86.8% | 86.8% |
| servicing_provider_phone | low | 35 | 20 | 57.1% | 57.1% |
| servicing_provider_phone | medium | 14 | 13 | 92.9% | 92.9% |
| type_of_service | none | 75 | 40 | 53.3% | 53.3% |
| type_of_service | low | 38 | 18 | 47.4% | 47.4% |
| type_of_service | medium | 12 | 4 | 33.3% | 33.3% |