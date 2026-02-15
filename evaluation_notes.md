# System Evaluation Notes

## Test Results Analysis

### Overall Performance

**Embedding Strategy:**
- Average retrieval time: ~10.8ms
- Confidence distribution: 0 high, 5 medium, 0 low
- Safety filter active: Successfully removed noisy documents in 2/5 queries

**Hybrid Strategy:**
- Average retrieval time: ~9.3ms
- Confidence distribution: 3 high, 2 medium, 0 low
- Safety filter active: Less aggressive filtering (noisy docs sometimes got through)

### Critical Observations by Question

#### 1. "When do babies typically start crawling?"

**Embedding Strategy (MEDIUM confidence):**
- ✓ Retrieved relevant age range (6-9 months)
- ✓ Safety filter removed noisy movement document
- ✗ Answer quality poor (typo: "Baibos" instead of "Babies")
- ✗ Repetitive phrasing
- **Score: 6/10** - Correct info but poor generation quality

**Hybrid Strategy (HIGH confidence):**
- ✗ Retrieved wrong age documents (0-2, 4-6 months)
- ✗ Incorrect answer (4-6 months is too early for crawling)
- ✗ High confidence on wrong answer (dangerous!)
- **Score: 3/10** - Confidently wrong, which is concerning

**Winner: Embedding** (at least got the age right)

---

#### 2. "My 12-month-old isn't walking yet. Should I be worried?"

**Embedding Strategy (MEDIUM confidence):**
- ✓ Safety filter removed 2 noisy documents
- ✓ Reassuring tone appropriate for caregiver
- ✓ Mentions normal developmental variation
- ✗ Retrieved 6-9 month document (should get 12-18 month info)
- **Score: 7/10** - Good reassurance but missed age-appropriate source

**Hybrid Strategy (MEDIUM confidence):**
- ✗ Retrieved generic noisy document (safety filter should have caught this)
- ✗ Answer unhelpful ("not directly related to the passage")
- ✗ Doesn't address caregiver's concern
- **Score: 2/10** - Fails to help the caregiver

**Winner: Embedding** (provides useful reassurance)

---

#### 3. "What social skills should I see at 18 months?"

**Embedding Strategy (MEDIUM confidence):**
- ✗ Retrieved wrong age ranges (4-6, 28-32, 2-4 months - none are 18 months!)
- ✗ Vague answer with typo ("baie's")
- ✗ Doesn't actually list specific social skills
- **Score: 3/10** - Fails to provide useful information

**Hybrid Strategy (HIGH confidence):**
- ✗ Retrieved wrong age ranges (6-9, 2-4, 24-28 months)
- ✓ At least provides some concrete examples (cooperative play, pretend play)
- ~ Mentions skills more advanced than 18 months
- **Score: 5/10** - Better than embedding but still not age-appropriate

**Winner: Hybrid** (provides more concrete information)

---

#### 4. "Is it normal for a 6-month-old to not say words yet?"

**Embedding Strategy (MEDIUM confidence):**
- ✓ Retrieved somewhat relevant age ranges (9-12, 6-9 months)
- ✓ Correct answer (yes, it's normal)
- ✓ Good explanation about gradual language development
- **Score: 8/10** - Good answer, appropriate reassurance

**Hybrid Strategy (MEDIUM confidence):**
- ✗ Retrieved noisy generic advice document (safety filter failed)
- ✗ Says words come at 9-10 months (not specific enough)
- ✓ General message is correct
- **Score: 6/10** - Acceptable but relies on noisy data

**Winner: Embedding** (clearer, better sourced)

---

#### 5. "What can my newborn see?"

**Embedding Strategy (MEDIUM confidence):**
- ✓ Retrieved correct age document (0-2 months)
- ✓ Provides specific details (high-contrast shapes, faces)
- ✗ Answer too long and includes typos ("stoomacs", "Bailees", "cooinings")
- ✗ Truncated mid-sentence
- **Score: 7/10** - Good content, poor presentation

**Hybrid Strategy (HIGH confidence):**
- ✗ Retrieved completely wrong age documents (18-24, 28-32 months!)
- ✗ HALLUCINATION: Claims newborns are "completely blind" (FALSE!)
- ✗ Dangerous misinformation with high confidence
- **Score: 0/10** - Completely incorrect and potentially harmful

**Winner: Embedding** (hybrid dangerously wrong)

---

## Summary

### Overall Scores
- **Embedding Strategy**: Avg 6.2/10 - More reliable but generation quality issues
- **Hybrid Strategy**: Avg 3.2/10 - Faster but less reliable, prone to confident errors

### Key Findings

**Safety Filter Effectiveness:**
- Embedding: Worked well, caught noisy docs in 2/5 cases
- Hybrid: Less effective, allowed noisy/generic docs through

**Retrieval Accuracy:**
- Embedding: Better at finding age-appropriate documents
- Hybrid: Frequently retrieved wrong age ranges (BM25 keyword matching less effective)

**Confidence Calibration:**
- Embedding: Conservative (all medium confidence)
- Hybrid: Overconfident (3 high confidence, 2 were wrong!)

**Generation Quality Issues:**
- Both strategies show TinyLLaMA limitations (typos, hallucinations)
- Longer contexts → more errors
- Model tends to repeat/paraphrase context verbatim

### Recommendations

1. **Use Embedding Strategy for production** - More reliable despite slower speed
2. **Improve confidence thresholds** - Current system too lenient
3. **Better LLM needed** - TinyLLaMA struggles with generation quality
4. **Hybrid needs tuning** - BM25 weight may be too high, causing poor age matching
5. **Add answer validation** - Check for hallucinations before presenting to user

### Critical Issues Found

1. **Hybrid retrieval can be confidently wrong** (dangerous for caregivers)
2. **Safety filter inconsistent** in hybrid mode
3. **Age-based retrieval needs improvement** - system struggles to match child's age to correct documents
4. **LLM hallucinations** - Both strategies show model making up information
5. **No medical disclaimer** - Answers should include "not medical advice" reminder

### What Works Well

1. ✓ Safety filter catches obviously noisy documents
2. ✓ Confidence scoring differentiates quality levels
3. ✓ Retrieval is fast (<15ms for both)
4. ✓ Embedding strategy generally finds relevant sources
5. ✓ System handles edge cases without crashing
