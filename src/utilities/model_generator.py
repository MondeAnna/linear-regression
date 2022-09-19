import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np


def generate_models():
    df = _generate_df()
    models = _get_models(df)

    return {
        "Model 1": models[0],
        "Model 2": models[1],
        "Model 3": models[2],
    }


def generate_feature_sets():
    df = _generate_df()

    base_features = df.columns
    non_base_features = [
        "salary",
        "prior_experience",
        "manager_position",
        "executive_position",
    ]

    model_1_features = base_features.drop(
        non_base_features
    ).drop(
        "prior_experience_original",
    ).insert(
        0,
        "prior_experience_original"
    )

    model_2_features = model_1_features.drop(
        "years_at_rank"
    )

    model_3_features = [
        "years_in_field",
        "executive_position",
        "market_value",
        "engineering_department"
    ]

    return {
        "Model 1": model_1_features,
        "Model 2": model_2_features,
        "Model 3": model_3_features,
    }


def _generate_df():
    df = read_in_df()
    df = clean_df(df)
    df = normalize_data(df)
    return df


def _get_models(df):
    model_features = generate_feature_sets()
    model_1 = create_model(df, model_features["Model 1"])
    model_2 = create_model(df, model_features["Model 2"])
    model_3 = create_model(df, model_features["Model 3"])

    return model_1, model_2, model_3


def create_model(df, features):
    X = df[features]
    y = df["salary"]

    X = sm.add_constant(X)

    return sm.OLS(y, X).fit()


def read_in_df():
    df = pd.read_csv(r"../data/salary_raw.csv")
    return df.rename(columns={
        "exprior": "prior_experience",
        "yearsworked": "years_in_field",
        "yearsrank": "years_at_rank",
        "market": "market_value",
        "degree": "has_degree",
        "otherqual": "has_other_qualification",
        "position": "employee_position",
        "male": "is_male",
        "Field": "field_of_work",
        "yearsabs": "years_absent",
    })


def clean_df(df):
    df = df.copy()
    df["salary"] = df["salary"].fillna(df["salary"].median())
    df = remove_outliers(df)
    df = append_original_prior_experience(df)
    return df


def normalize_data(df):
    discrete_features = DISCRETE.copy()
    discrete_features.append("prior_experience_original")
    df.copy()
    df[discrete_features] = df[discrete_features].add(0.001).apply(np.log)
    return df


def get_outlier_bounds(df):
    q1 = df[DISCRETE].quantile(0.25)
    q3 = df[DISCRETE].quantile(0.75)

    iqr = q3 - q1
    iqr_range = iqr * 1.5

    lower_bound = q1 - iqr_range
    upper_bound = q3 + iqr_range

    return lower_bound, upper_bound


def fill_corrupt_values(df):
    df = df.copy()
    for feat in DISCRETE:
        df[feat] = df[feat].apply(
            lambda x: df[feat].median() if np.isnan(x) else x
        )
    return df


def get_fields_of_work(df):
    fields_of_work = pd.get_dummies(df["field_of_work"])
    fields_of_work.rename(columns={
        1: "engineering_department",
        2: "finance_department",
        3: "human_resources_department",
        4: "marketing_department",
    }, inplace=True)
    return fields_of_work.drop(columns=["human_resources_department"])


def get_employee_positions(df):
    employee_positions = pd.get_dummies(df["employee_position"])
    employee_positions.rename(columns={
        1: "junior_position",
        2: "manager_position",
        3: "executive_position",
    }, inplace=True)
    return employee_positions.drop(columns=["junior_position"])


def hot_encode(df):
    fields = get_fields_of_work(df)
    positions = get_employee_positions(df)
    df_no_fields = df.drop(columns=["field_of_work"])
    return pd.concat([df_no_fields, fields, positions], axis="columns")


def remove_outliers(df):
    df = df.copy()
    lower_bound, upper_bound = get_outlier_bounds(df)

    min_greater_than = df[DISCRETE] > lower_bound
    max_less_than = df[DISCRETE] < upper_bound

    df[DISCRETE] = df[DISCRETE][min_greater_than & max_less_than]

    df = fill_corrupt_values(df)
    df = hot_encode(df)

    return df


def append_original_prior_experience(df):
    prior_experience = pd.read_csv("../data/salary_raw.csv")[["exprior"]]
    df = df.copy()
    df["prior_experience_original"] = prior_experience
    return df


SMALL = 10
MEDIUM = 15
LARGE = 20
GIGANTIC = 25
COLLOSAL = 30
BOLD = "bold"


NOMINAL = [
    "field_of_work",
    "employee_position",
]


BINARY = [
    "is_male",
    "has_degree",
    "has_other_qualification",
]


DISCRETE = [
    "salary",
    "prior_experience",
    "years_in_field",
    "years_at_rank",
    "years_absent",
    "market_value",
]


sns.set(rc={
    "figure.figsize": (LARGE, SMALL),
    "figure.titlesize": COLLOSAL,
    "figure.titleweight": BOLD,
    "axes.labelpad": GIGANTIC,
    "axes.labelsize": GIGANTIC,
    "axes.labelweight": BOLD,
    "axes.titlepad": COLLOSAL,
    "axes.titlesize": COLLOSAL,
    "axes.titleweight": BOLD,
    "xtick.labelsize": LARGE,
    "ytick.labelsize": LARGE,
    "legend.fontsize": LARGE,
})

