# Real LLM Deployment Project Report

## 1. Problem Definition

- Use case:
- Users:
- Why fine-tuning is needed:
- Why a small model is appropriate:

## 2. Data Preparation

Describe each source separately.

| Source | Number of examples | Task type | Cleaning/filtering | Category |
|--------|--------------------|-----------|--------------------|----------|
| | | | | |

Explain how you converted examples into chat format.

## 3. Splitting Strategy

Report train/validation/eval sizes.

Explain why your split prevents data leakage.

## 4. Training Methodology

Explain:

- base model choice;
- LoRA configuration;
- assistant-only loss;
- why labels are set to `-100` for system/user/padding tokens;
- checkpoint policy;
- monitoring.

## 5. Evaluation

Report automatic metrics and qualitative examples.

## 6. Deployment

Include:

- Hugging Face model repo link;
- Hugging Face Space link;
- screenshots or usage examples;
- model card link.

## 7. Limitations and Risks

Discuss:

- hallucination;
- poor categories;
- data bias;
- unsafe/out-of-scope prompts;
- prompt injection if using retrieval or state machine context.

## 8. What You Improved

Explain at least one improvement to:

- data quality;
- `src/context_state_machine.py`;
- deployment app;
- evaluation methodology.

## 9. Individual Technical Notes

Each student writes half a page. Maximum one page per student.

### Student 1: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:

### Student 2: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:

### Student 3: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:
