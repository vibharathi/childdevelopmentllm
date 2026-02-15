# Project Guidelines

## Task

Your task is to build a **local prototype** that can answer caregiver-style questions about early childhood development (birth to 36 months). You will be provided with a small set of milestone reference texts - some of which intentionally contain inconsistencies or noise. The data set is provided in the zip file attached to the email.

## Requirements

### Technical Constraints

All work must:
- Run entirely locally (no cloud APIs)
- Use only free, open-source models and libraries with the programming language of your choice
- Be runnable via instructions included in your GitHub repo

### Core Requirements

1. **Implement a minimal question-answering system.** Your system must include:
   - A retrieval mechanism (embeddings, keyword, hybrid - your choice)
   - An answer generation component (local LLM, template-based generation - your choice)
   - A safety / quality layer that handles inappropriate, medical, or high-risk queries

2. **Compare Two Strategies**
   - Include two distinct retrieval-and/or-generation strategies, for example:
     - Embeddings-based retrieval vs keyword retrieval
     - Different embedding models
     - Different ways of ranking/merging retrieved evidence, etc.
   - Your README should briefly explain:
     - Why you chose the two strategies
     - What trade-offs you observed
     - Which one performed better and why

3. **Implement a Confidence / Uncertainty Fallback**
   - Your system should include a simple mechanism that allows it to:
     - Detect low-confidence or conflicting retrieval results
     - Provide a fallback response, such as:
       - A safe uncertainty statement
       - A refusal when appropriate
       - A generic guidance answer
   - Examples of uncertainty signals you may use (choose any):
     - Low similarity scores
     - Large disagreement between retrieved documents
     - Missing relevant retrieval
     - Triggering safety rules
     - Low LLM likelihood or incoherence
   - You may design the fallback behavior however you see fit.

## Deliverables

Please provide a link to a GitHub repository containing:

### 1. Runnable Source Code

Your repo must include:
- Source code for the prototype
- All required model files or download scripts
- requirements.txt, package.json or equivalent

### 2. README with the Following Sections

- Setup instructions (environment, dependencies, how to run)
- Architecture overview (your reasoning for key decisions)
- Explanation of the two strategies you compared
- Description of your confidence / uncertainty fallback
- 3–5 example questions and your system's actual outputs
- Your evaluation of those outputs (good, bad, concerning, uncertain—your reasoning)
- Known limitations or things you'd improve with more time

### 3. Notes on Use of AI Tools

You may and are encouraged to use AI tools during development. If you do, be prepared to discuss:
- How you guided them
- What you accepted or rejected
- Where you exercised your own judgment
- Which models you picked for various tasks and why
