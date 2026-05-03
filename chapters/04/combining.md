# Putting It All Together

We now have six features per row: two logprob confidence scores, two similarity scores, and two judge scores. We combine them with a simple threshold rule and use Optuna to find the best thresholds via multi-objective optimization.

## Baseline

```python
df['WRONG'] = ((df['address WRONG'] == 1) | (df['title WRONG'] == 1)).astype(int)
df['CORRECT'] = (df['WRONG'] == 0).astype(int)

print(f"Total wrong: {df['WRONG'].sum()} out of {len(df)}")
print(f"Baseline precision (assume all correct): {df['CORRECT'].mean():.2%}")
```

## Threshold rule

```python
pred_correct = (
    (df['address_prob_norm'] >= addr_conf_th) &
    (df['title_prob_norm'] >= title_conf_th) &
    (df['similarity_transcription_address'] >= addr_sim_th) &
    (df['similarity_transcription_title'] >= title_sim_th) &
    (df['llm_judge_address_prob'] <= addr_judge_th) &
    (df['llm_judge_title_prob'] <= title_judge_th)
).astype(int)
```

## Optimizing thresholds with Optuna

:::{admonition} What is Optuna?
:class: note dropdown
Optuna is a hyperparameter optimization framework that automatically searches a parameter space using sampling strategies like TPE (Tree-structured Parzen Estimator). Instead of trying all combinations, it learns from previous trials to focus on promising regions. Here we use it in multi-objective mode, maximizing both precision and recall simultaneously, which produces a Pareto front of optimal trade-offs rather than a single best solution.
:::

Manual tuning six thresholds is tedious. Optuna explores the space automatically and finds the Pareto front between precision and recall:

```python
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    addr_conf_th = trial.suggest_float('addr_conf_th', 0.0, 1.0, step=0.05)
    title_conf_th = trial.suggest_float('title_conf_th', 0.0, 1.0, step=0.05)
    addr_sim_th = trial.suggest_float('addr_sim_th', 0.0, 1.0, step=0.05)
    title_sim_th = trial.suggest_float('title_sim_th', 0.0, 1.0, step=0.05)
    addr_judge_th = trial.suggest_float('addr_judge_th', 0.0, 1.0, step=0.05)
    title_judge_th = trial.suggest_float('title_judge_th', 0.0, 1.0, step=0.05)

    pred_correct = (
        (df['address_prob_norm'] >= addr_conf_th) &
        (df['title_prob_norm'] >= title_conf_th) &
        (df['similarity_transcription_address'] >= addr_sim_th) &
        (df['similarity_transcription_title'] >= title_sim_th) &
        (df['llm_judge_address_prob'] <= addr_judge_th) &
        (df['llm_judge_title_prob'] <= title_judge_th)
    ).astype(int)

    return precision_score(df['CORRECT'], pred_correct, zero_division=0), \
           recall_score(df['CORRECT'], pred_correct, zero_division=0)

study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(objective, n_trials=500, show_progress_bar=True)
```

```python
fig = vis.plot_pareto_front(study, target_names=['Precision', 'Recall'])
fig.update_layout(xaxis=dict(range=[0.8, 1.0]))
fig.show()
```

Pick the best parameters for a given minimum precision requirement:

```python
def get_best_recall_for_precision(study, min_precision=0.8):
    valid_trials = [t for t in study.best_trials if t.values[0] >= min_precision]
    if not valid_trials:
        print(f"No trials found with precision >= {min_precision}")
        return None
    best_trial = max(valid_trials, key=lambda t: t.values[1])
    print(f"Precision: {best_trial.values[0]:.2%}, Recall: {best_trial.values[1]:.2%}")
    print(json.dumps(best_trial.params, indent=2))
    return best_trial.params

best_params = get_best_recall_for_precision(study, min_precision=0.95)
```

## Feature importance with SHAP

:::{admonition} What is SHAP?
:class: note dropdown
SHAP (SHapley Additive exPlanations) assigns each feature a contribution score for a given prediction, based on game-theory Shapley values. It answers "how much did this feature push the prediction up or down?", giving you interpretable, consistent feature importance across any model. More: [shap.readthedocs.io](https://shap.readthedocs.io)
:::

Which of the six signals actually matters? SHAP tells us:

```python
features = [
    'address_prob_norm', 'title_prob_norm',
    'similarity_transcription_address', 'similarity_transcription_title',
    'llm_judge_address_prob', 'llm_judge_title_prob'
]

def model_predict(data_array):
    return (
        (data_array[:, 0] >= best_params['addr_conf_th']) &
        (data_array[:, 1] >= best_params['title_conf_th']) &
        (data_array[:, 2] >= best_params['addr_sim_th']) &
        (data_array[:, 3] >= best_params['title_sim_th']) &
        (data_array[:, 4] <= best_params['addr_judge_th']) &
        (data_array[:, 5] <= best_params['title_judge_th'])
    ).astype(float)

X = df[features].values
explainer = shap.KernelExplainer(model_predict, shap.sample(X, 50))
shap_values = explainer.shap_values(X, nsamples=100)

shap.summary_plot(shap_values, X, feature_names=features)
```
