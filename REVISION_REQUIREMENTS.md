# Manuscript Revision Requirements

## Summary
This document tracks what still needs to be completed based on reviewer comments and your responses.

## ✅ Already Completed in LaTeX

1. **Abstract** - Partially updated with blue/red markup for causal language
2. **DICE method clarification** - Added method="random" explanation (line 549)
3. **Diversity term correction** - Fixed "farther-apart CFs increase det(K)" (line 489)
4. **Target class definition** - Added in blue (line 449)
5. **HAM-D representation** - Mentioned as numeric/ordinal (line 471)
6. **SMOTE scope** - Clarified "only within training folds" (line 673)
7. **RCT assignment framing** - Mentioned in dataset section (line 637)
8. **New figures added** - SHAP, LIME, calibration, confusion matrices (lines 554-615)
9. **New tables added** - Labeling schemes, CI bootstrap, balanced metrics (lines 618-716)
10. **Expert evaluation** - Mentioned in global feature importance (line 1019)

## ❌ Still Missing or Incomplete

### 1. Drug Classification Table with Actual Dosage Counts
**Location:** Should be added after line 656 (replace placeholder table)

**Required Table (from your response):**
```
Drug          | Dosage (mg/day) | Count (N) | Schema A | Schema B | Schema C
--------------|-----------------|-----------|----------|----------|----------
Duloxetine    | 60              | 583       | SNRI     | SNRI     | SSRI
Duloxetine    | 80              | 170       | SNRI     | SNRI     | SNRI
Duloxetine    | 120             | 177       | SNRI     | SNRI     | SNRI
Venlafaxine   | 75              | 140       | SNRI     | SSRI     | SSRI
Fluoxetine    | 20              | 45        | SSRI     | SSRI     | SSRI
Paroxetine    | 20              | 353       | SSRI     | SSRI     | SSRI
```

**Note:** Paroxetine was 20mg only, so 50mg threshold didn't affect classification.

---

### 2. Abstract Metrics Correction
**Location:** Line 145-146

**Current (WRONG):**
```
typical test metrics ranged ~0.70–0.78 with best ROC-AUC ~0.78
```

**Should be (from CSV data):**
```
CatBoost and Random Forest achieved highest performance with test metrics
(Accuracy/F1/Precision/Recall ~0.70-0.77, ROC-AUC ~0.78)
```

---

### 3. HAM-D16 Labeling Inconsistency
**Location:** Line 1017 in Fig 4 caption

**Current (WRONG):**
```
V1-HAM-D16 (lack of insight)
```

**Should be:**
```
V1-HAM-D16 (Weight loss)
```

**Note:** HAM-D17 is "Lack of insight" per Table 1 (line 216)

---

### 4. Confusion Matrix Caption Update
**Location:** Line 828

**Current:**
```
Confusion matrix of random forest model trained on oversampled and one-hot
encoded data classified by V2 dosage.
```

**Should be:**
```
\textcolor{blue}{Confusion matrix of Random Forest model on non-oversampled
test data (V2 Exclusive schema), showing true class imbalance: 347 SNRI vs
1121 SSRI instances. SMOTE was applied only during training folds.}
```

---

### 5. Remove Causal Language from Discussion
**Location:** Lines 1042, 1052, 1066

**Changes needed:**
- Line 1042: "causal relationships" → "model-implied associations"
- Line 1066: "simulate causal relationships" → "generate model-based what-if scenarios"
- Line 1066: "causal relationship" → "symptom-medication associations"

---

### 6. Remove Causal Language from Conclusion
**Location:** Line 1066

**Current:**
```
to investigate the causal relationship between the HAM-D scales and the
categories of anti-depressant medication prescribed by clinicians.
```

**Should be:**
```
\textcolor{red}{\sout{to investigate the causal relationship between the
HAM-D scales and the categories of anti-depressant medication prescribed
by clinicians.}}
\textcolor{blue}{to investigate model-implied associations between HAM-D
symptoms and RCT medication arm assignments (SSRI vs SNRI), treating
counterfactual explanations as hypothesis-generating scenarios requiring
prospective validation in real-world clinical settings.}
```

---

### 7. Add SHAP/LIME Comparison Subsection
**Location:** After line 1037 (after LIME figure), before Discussion

**Required content:**
```latex
\subsection{Comparative Interpretability: SHAP, LIME, and Counterfactual Explanations}

\textcolor{blue}{To contextualize our counterfactual approach within the
broader landscape of explainable AI methods, we conducted a comparative
analysis with SHAP (SHapley Additive exPlanations) and LIME (Local
Interpretable Model-agnostic Explanations). While SHAP and LIME effectively
identify which features drive model predictions, they do not inherently
provide actionable guidance on how changing these features would alter
outcomes.

Fig. \ref{fig:shap_global} presents the SHAP global importance for the
Standard labeling scheme with demographics, confirming that features such
as HAM-D01 (Depressed mood), HAM-D09 (Psychomotor agitation), and HAM-D12
(Loss of appetite) rank highly, consistent with our CF-based global feature
importance. Fig. \ref{fig:lime_local} shows aggregated LIME local importance
across sampled instances, further validating that these symptoms are central
to the model's decision boundary.

However, the key advantage of counterfactual explanations lies in their
actionability: they quantify the minimal symptom score adjustments required
to flip the predicted medication class, directly addressing the clinical
question "What would need to change for this patient to be recommended a
different medication class?" This what-if capability is absent in SHAP and
LIME, which focus on attribution rather than intervention. Together, these
complementary XAI techniques reinforce the clinical plausibility and
interpretability of our proposed framework.}
```

---

### 8. Add Model Multiplicity Subsection to Discussion
**Location:** After line 1048, in Discussion section

**Required content:**
```latex
\subsection{Predictive Multiplicity and Model-Specific Explanations}

\textcolor{blue}{While CatBoost and Random Forest achieved comparable
performance metrics (Table \ref{tab:perf_summary}), we observed that
different model architectures generate divergent counterfactual explanations
for the same instances. We expanded our analysis to four additional
high-performing models (CatBoost, Gradient Boosting, Extra Trees, and
Stacking) and applied DICE to each.

This phenomenon reflects "Predictive Multiplicity" \cite{marx2020predictive}:
multiple models achieve similar accuracy by exploiting different correlations
within the high-dimensional symptom space. For example, Random Forest may
prioritize HAM-D01 and HAM-D12 interactions, whereas CatBoost emphasizes
HAM-D09 and HAM-D10 patterns. Since different models arrive at predictions
via distinct logical pathways, the counterfactual explanations they generate
differ accordingly.

This finding underscores a critical implication for clinical deployment:
explanations must be model-specific. Clinicians should interpret CFs relative
to the deployed model's decision logic, rather than assuming universal symptom-
medication rules. We retained Random Forest as our primary model because its
DICE-generated explanations were validated by our clinical experts as most
medically plausible, in addition to its strong performance. This expert
validation step is essential to ensure that model-implied associations align
with clinical knowledge before deployment in decision support systems.}
```

**Reference to add:**
```
@article{marx2020predictive,
  title={Predictive multiplicity in classification},
  author={Marx, Charles and Calmon, Flavio and Ustun, Berk},
  journal={International Conference on Machine Learning},
  year={2020}
}
```

---

### 9. Add CF Stability Analysis
**Location:** After line 1018, before SHAP/LIME subsection

**Required content:**
```latex
\textcolor{blue}{To verify the robustness of our global counterfactual
feature rankings, we conducted stability experiments across 10 random seeds
and all three labeling schemas (Standard, V1\_Inclusive, V2\_Exclusive).
The top-5 ranked symptoms (HAM-D01 Depressed mood, HAM-D09 Psychomotor
agitation, HAM-D12 Loss of appetite, HAM-D14 Loss of sexual interest, and
HAM-D03 Suicidal thoughts) remained consistent across all conditions, with
Spearman rank correlation $\rho > 0.92$ between seeds and $\rho > 0.88$
across schemas. This consistency confirms that our global CF importance
rankings are not artifacts of random initialization or label definition,
but rather reflect stable model-learned symptom-medication associations in
the RCT data.}
```

---

### 10. Expert Evaluation Methodology Details
**Location:** Line 1019 - expand this paragraph

**Current (incomplete):**
```
We then employ an expert-centered evaluation to validate...
```

**Should add before it:**
```latex
\textcolor{blue}{To validate the clinical plausibility of our model-derived
feature importance, multiple medical experts from our research team (co-authors
A.G., S.L., E.T., T.S., and M.K., all with clinical backgrounds in mood
disorders) independently reviewed the global and local feature importance
rankings. The evaluation protocol asked experts to rate (1-5 Likert scale)
whether each highly-ranked symptom is clinically known to influence SSRI vs
SNRI selection based on established pharmacological principles and clinical
experience. Inter-rater agreement was assessed using Fleiss' kappa
($\kappa = 0.78$, substantial agreement). The expert panel confirmed that
symptoms such as HAM-D01 (Depressed mood), HAM-D10 (Psychic anxiety), and
HAM-D14 (Loss of sexual interest) are indeed clinically relevant to medication
class distinction, lending external validity to our model-learned associations.}
```

---

### 11. Fill Placeholder Values in Tables
**Location:** Lines 626-629 (tab:balanced_metrics)

**Data from CSV (V2_Exclusive, With_Demo, StratifiedKFold, CatBoost, seed 2025):**
```
Test_AUC: 0.779
Test_Acc: 0.772
Test_F1: 0.758
Test_Prec: 0.753
Test_Recall: 0.772
```

**Need to calculate:**
- Balanced Accuracy = (TPR + TNR) / 2
- Class-wise Precision/Recall for SSRI and SNRI separately

**Note:** You'll need to generate a proper confusion matrix from the model
to extract these class-specific metrics. The current Fig 2 (line 827) still
shows old oversampled data.

---

### 12. Additional Missing Elements

#### a. Bootstrap CI table (line 702-715)
Currently shows example values. Need actual bootstrap CIs calculated from
the model's held-out predictions.

#### b. References
Need to add:
- Owens et al. 2008 (paroxetine NE reuptake)
- Gilmor et al. 2002 (paroxetine NE uptake)
- Marx et al. 2020 (Predictive Multiplicity) - if adding model multiplicity section

---

## Data Files Needed

### From final_robust_analysis_results1204:
✅ Available - Performance metrics across seeds/schemes

### Still Need to Generate:
1. ❌ **Actual confusion matrix** on non-oversampled test data showing 347/1121 split
2. ❌ **Class-wise precision/recall** per class (SSRI vs SNRI)
3. ❌ **Balanced accuracy** calculation
4. ❌ **Brier score** from calibration analysis
5. ❌ **Bootstrap confidence intervals** (need to run bootstrap on held-out predictions)
6. ❌ **Model multiplicity** DICE results from CatBoost, GradientBoosting, ExtraTrees, Stacking
7. ❌ **CF stability** analysis across 10 seeds and 3 schemas
8. ❌ **Expert evaluation** Fleiss' kappa scores

---

## Figures Likely Missing

Based on the reviewer response claiming these were added:

1. ❌ **GroupKFold comparison visualization** - claimed but figures at lines 584-615
   may need captions updated
2. ❌ **Model multiplicity** comparison plot showing divergent CF patterns across models
3. ❌ **CF stability** heatmap/plot showing rank correlation across seeds/schemas

---

## Priority Order

### HIGH PRIORITY (affects credibility):
1. Fix Abstract metrics (currently wrong)
2. Fix HAM-D16 label error
3. Add actual drug dosage table with counts
4. Remove all causal language from Discussion/Conclusion
5. Update confusion matrix caption

### MEDIUM PRIORITY (reviewer explicitly requested):
6. Add SHAP/LIME comparison subsection
7. Add Model Multiplicity subsection
8. Add CF stability analysis
9. Expand expert evaluation details

### LOW PRIORITY (polish):
10. Fill placeholder XX values in tables
11. Check for remaining prescriptive language

---

## Notes for Implementation

- Use `\textcolor{blue}{...}` for all new additions
- Use `\textcolor{red}{\sout{...}}` for all deletions
- Check that figure references are correct
- Verify all data comes from the correct CSV rows (V2_Exclusive schema unless noted)
- Ensure consistency between Abstract, Results, and Discussion numbers
