import numpy as np
from prophet import Prophet
import optuna


def compute_mase(training_series, testing_series, prediction_series):
    prediction_series.index = np.arange(len(training_series), len(training_series) + len(testing_series))

    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)
    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


# Prophet related functions
def p_model(params, df):
    yearly = False
    if df.shape[0] > 52:
        yearly = True

    # Model
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=True,
        daily_seasonality=False,
        # Make Prophet run a lot faster by disabling the confidence interval
        uncertainty_samples=0,
        **params
    )

    m.add_country_holidays(country_name='FR')

    return m


def p_model_df(df):
    df = df.reset_index(drop=True)

    return df


def make_future(model, df, periods=60, freq='1D'):
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)

    return future


def suggest_params(trial):
    return {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.8),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
        'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
    }


def objective(trial, train, valid, horizon):
    try:
        params = suggest_params(trial)

        df = p_model_df(train)
        m = p_model(params, df)

        if params['growth'] == 'logistic':
            df['cap'] = df['y'].max()
            df['floor'] = 0.0

        m.fit(df)

        future = make_future(m, df, periods=horizon)
        if params['growth'] == 'logistic':
            future['cap'] = df['y'].max()
            future['floor'] = 0.0

        forecast = m.predict(future)

        metric = compute_mase(train['y'], valid['y'], forecast['yhat'])

    except Exception as e:
        print(e)
        print(f"Encountered an error while performing trial {trial.number}, trial got pruned.")
        raise optuna.TrialPruned()

    return metric


def optimize(train, valid, num_trials, horizon):
    # Create a sampler and set seed for repeatability.
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=10)
    pruner = optuna.pruners.SuccessiveHalvingPruner()

    # Create a study
    direction = "minimize"
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    objective_wrapper = lambda trial: objective(trial, train, valid, horizon)

    study.optimize(objective_wrapper, n_trials=num_trials, show_progress_bar=False)

    print(f"Keeping best trial: {study.best_trial.number}")
    print(f"Keeping best params: {study.best_params}")
    print(f"Study best metric value: {study.best_value}")

    return study.best_params
