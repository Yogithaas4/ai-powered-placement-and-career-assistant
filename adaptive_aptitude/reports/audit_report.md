# Question Bank Audit Report

- Clean questions: **8855**
- Image-backed questions: **243**
- Total: **9098**

## Field completeness

- `question_id` missing: 0
- `question` missing: 0
- `question_type` missing: 0
- `correct_answer` missing: 3
- `subject` missing: 0
- `topic` missing: 0
- `subtopic` missing: 0
- `difficulty` missing: 0
- `time_expected_minutes` missing: 0
- Option-dependent questions with no options at all: 135

## Type-specific issues

- **multi_select_suspicious_single_answer**: 22 (e.g. ['da::1.11.2', 'da::5.2.1', 'da::7.17.13', 'da::13.6.7', 'isro_w_cover::4.2.6'])
- **match_following_missing_mapping_data**: 23 (e.g. ['da::1.65.37', 'da::3.41.8', 'da::4.10.50', 'da::13.9.5', 'em::7.24.2'])
- **numerical_answer_not_parseable_as_number**: 92 (e.g. ['da::3.34.2', 'da::13.1.3', 'em::2.29.5', 'em::2.35.1', 'em::2.36.1'])
- **visual_type_without_image_info**: 36 (e.g. ['isro_w_cover::1.3.1', 'isro_w_cover::1.3.2', 'isro_w_cover::1.4.1', 'isro_w_cover::1.8.1', 'isro_w_cover::1.10.2'])
- **image_based_missing_image_reference**: 36 (e.g. ['isro_w_cover::1.3.1', 'isro_w_cover::1.3.2', 'isro_w_cover::1.4.1', 'isro_w_cover::1.8.1', 'isro_w_cover::1.10.2'])

## Duplicates

- Duplicate question_id occurrences: 0
- Duplicate question-text groups: 518 (covering 1099 records)

## Taxonomy

- Unique subject labels: 36
- Unique (subject, topic) pairs: 417
- Likely-duplicate subject-label groups: 2
  - `operating system` <- ['Operating System', 'Operating Systems']
  - `programming and data structure` <- ['Programming and Data Structure', 'Programming and Data Structures']
- Likely-duplicate topic-label groups (within same normalized subject): 17

## Distribution

### By question type
- mcq: 7870
- numerical: 672
- multi_select: 239
- fill_blank: 139
- image_based: 100
- match_following: 65
- diagram_based: 6
- table_based: 6
- graph_based: 1

### By difficulty
- Medium: 4965
- Easy: 2876
- Hard: 1257

### By validation status
- validated: 8636
- valid: 323
- clean: 126
- options_restored_from_pdf: 5
- answer_disputed: 3
- multi_answer_C_or_D: 2
- answer_is_none_of_above: 2
- manual_review_required: 1