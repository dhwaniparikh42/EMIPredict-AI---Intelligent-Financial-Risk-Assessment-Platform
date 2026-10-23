# Data Quality Report

- Rows entering cleaning (after parsing): **404,800**
- Rows after cleaning: **394,301** (dropped 10,499)

## 1. Missing values (before cleaning)

| column | nulls |
|---|---|
| bank_balance | 2440 |
| monthly_rent | 2426 |
| credit_score | 2420 |
| education | 2404 |
| emergency_fund | 2351 |

## 2. Rule violations: before vs after

| rule | before | after (train+val+test) |
|---|---|---|
| credit_score_out_of_range | 4,776 | 0 |
| max_emi_exceeds_salary | 1,066 | 0 |
| max_monthly_emi_out_of_range | 458 | 0 |
| monthly_salary_out_of_range | 10,065 | 0 |
| rent_on_non_rented_house | 4,369 | 0 |

## 3. Cleaning actions

| action | rows / values affected |
|---|---|
| malformed_numbers_repaired (e.g. '18000.0.0') | 3,948 |
| unparseable_numbers_set_to_null (e.g. 'nan.0') | 14 |
| gender_labels_normalised (MALE/M/male/F/... -> Male/Female) | 9,022 |
| unknown_category_labels_set_to_null | 0 |
| duplicate_rows_dropped | 0 |
| invalid_credit_scores_set_to_null (0/999/1200) | 4,776 |
| rent_set_to_0_for_Own/Family_houses | 5,856 |
| rows_dropped_null_in_critical_column | 0 |
| rows_flagged: age_out_of_range | 0 |
| rows_flagged: monthly_salary_out_of_range | 10,065 |
| rows_flagged: max_monthly_emi_out_of_range | 458 |
| rows_flagged: requested_amount_out_of_range_for_scenario | 0 |
| rows_flagged: requested_tenure_out_of_range_for_scenario | 0 |
| rows_flagged: negative_values | 0 |
| rows_flagged: existing_loans_flag_mismatch | 0 |
| rows_flagged: dependents_ge_family_size | 0 |
| rows_flagged: employment_exceeds_working_age | 0 |
| rows_flagged: max_emi_exceeds_salary | 1,066 |
| rows_dropped_business_rules (unique rows) | 10,499 |

## 4. Train / validation / test split

| split | rows | Eligible | High_Risk | Not_Eligible |
|---|---|---|---|---|
| train | 276,010 | 18.42% | 4.35% | 77.23% |
| val | 59,145 | 18.42% | 4.36% | 77.22% |
| test | 59,146 | 18.42% | 4.35% | 77.23% |

Stratified on `emi_eligibility` x `emi_scenario`, so the class mix is preserved in every split.

## 5. Imputation

Fitted on the **train split only** and applied to val/test (no leakage). Values are stored in `artifacts/imputation_values.json`.
