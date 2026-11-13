"""
Feature Engineering Module
TODO: Implement the functions below
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def scale_features(X_train, X_test, method="standard"):
    """Scale features using the specified method.
    TODO: Implement this function
    - method: "standard", "minmax", or "none"
    - Return (X_train_scaled, X_test_scaled, scaler)
    """
    if method == "none":
        return X_train.copy(), X_test.copy(), None

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError(
            "Scaling method must be "
            "'standard', 'minmax', or 'none'"
        )

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return ( X_train_scaled , X_test_scaled , scaler )


def select_features(X_train, y_train, X_test, k=10):
    """Select top-k features using ANOVA F-value.
    TODO: Implement this function
    - Return (X_train_selected, X_test_selected, selector)
    """
    if k > X_train.shape[1]:
        raise ValueError(
            "k cannot be greater than "
            "the number of features"
        )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform( X_train,y_train)

    X_test_selected = selector.transform(X_test)

    selected_columns = X_train.columns[ selector.get_support() ]

    X_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_columns,
        index=X_train.index
    )

    X_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_columns,
        index=X_test.index
    )

    return (
        X_train_selected,
        X_test_selected,
        selector
    )