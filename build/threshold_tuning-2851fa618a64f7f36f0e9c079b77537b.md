# Threshold Tuning

Every transcript now has a continuous `output_prob` score. The threshold (the cutoff above which we call something an incident) is something we get to choose.

The **Precision-Recall curve** shows the tradeoff across every possible threshold at once:

```python
precision, recall, thresholds = precision_recall_curve(df['label'], df['output_prob'])
pr_auc = auc(recall, precision)
positive_class_ratio = df['label'].mean()
```

```python
pred_05 = (df['output_prob'] >= 0.5).astype(int)
precision_05 = precision_score(df['label'], pred_05, zero_division=0)
recall_05 = recall_score(df['label'], pred_05, zero_division=0)
```

```python
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.axhline(y=positive_class_ratio, color='r', linestyle='--', label=f'Random baseline')
plt.scatter(recall_05, precision_05, color='green', s=100,
            label=f'Threshold 0.5 (P={precision_05:.2f}, R={recall_05:.2f})', zorder=5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()
```

0.5 is just a default, not the right answer. Where you set it depends on what's more costly: missing a real incident, or acting on a false one.
