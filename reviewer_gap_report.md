# 审稿意见未满足点排查报告

以下项目依据 `REVISION_REQUIREMENTS.md` 中的要求检查，仍需后续修改：

1. **药物分类剂量表缺失**：在数据集描述部分（表\ref{tab:labeling_schemes}附近）仍未加入审稿人要求的“Drug | Dosage | Count | Schema A/B/C”剂量统计表，现有表格只列出了分类方案与总计数。
2. **摘要指标表述仍未按要求修改**：摘要结果段仍为“典型测试指标~0.70–0.78，最佳ROC-AUC~0.78”，尚未替换为审稿人指定的“Accuracy/F1/Precision/Recall ~0.70-0.77, ROC-AUC ~0.78”的表述。
3. **HAM-D16 标签不一致**：全局特征重要性图（Fig.\ref{fig4}）的图注仍写“V1-HAM-D16 (lack of insight)”，未改为“V1-HAM-D16 (Weight loss)”。
4. **混淆矩阵图注需更新**：Fig.\ref{fig2} 图注仍描述为“oversampled and one-hot encoded data”，未改为审稿人要求的“非过采样测试集 V2 Exclusive 架构，显示真实类别不平衡，并声明 SMOTE 仅用于训练折”。
5. **讨论区因果措辞未调整**：讨论段仍多处使用“causal relationships”等措辞（如讨论开头及结论），未按要求改写为“model-implied associations”等非因果表述，并未加入模型假设情境的说明。
6. **结论区因果措辞未替换**：结论仍描述“investigate the causal relationship...”，未按要求改为“model-implied associations between HAM-D symptoms and RCT medication arm assignments...”。
7. **缺少 SHAP/LIME 对比小节文字**：在 Fig.\ref{fig:shap_global} 与 Fig.\ref{fig:lime_local} 之后，尚未加入审稿人要求的“Comparative Interpretability: SHAP, LIME, and Counterfactual Explanations”小节及对应文字说明。
8. **缺少模型多重性小节**：讨论区未添加“Predictive Multiplicity and Model-Specific Explanations”小节，也未引入 marx2020predictive 文献引用，未阐述不同模型产生不同 CF 的临床解释影响。
9. **表格占位数据未替换**：平衡准确率表（Table~\ref{tab:balanced_metrics}）仍为占位符“0.XX”，未填入真实数值。

以上内容需根据审稿意见继续完善，当前未进行修改以便后续集中处理。
