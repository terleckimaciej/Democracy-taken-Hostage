import pandas as pd
import numpy as np
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def add_interactions(df: pd.DataFrame, base: str, partners: dict) -> pd.DataFrame:
    base_alias = base[:-2] if base.endswith('_n') else base
    for suffix, col in partners.items():
        interaction = df[base] * df[col]
        df[f"{base}_X_{suffix}"] = interaction
        if base_alias != base:
            df[f"{base_alias}_X_{suffix}"] = interaction
    return df


def compute_vif(frame: pd.DataFrame, features: list) -> pd.DataFrame:
    df = frame.copy()

    def _ensure_interaction(name: str) -> bool:
        if name in df.columns:
            return True
        if name.startswith('polar_X_'):
            suffix = name[len('polar_X_'):]
            alt = 'polar_n_X_' + suffix
            if alt in df.columns:
                df[name] = df[alt]
                return True
            mapping = {
                'gdp_pc_log': 'log_gdp_pc', 'mulpar': 'mulpar_n', 'hc_indep': 'hc_indep',
                'Equal': 'equal', 'Prop': 'Prop', 'Mixed': 'Mixed', 'feduni': 'feduni'
            }
            partner = mapping.get(suffix, suffix)
            if 'polar_n' in df.columns and partner in df.columns:
                df[name] = df['polar_n'] * df[partner]
                return True
        if name.startswith('polar_n_X_'):
            suffix = name[len('polar_n_X_'):]
            alt = 'polar_X_' + suffix
            if alt in df.columns:
                df[name] = df[alt]
                return True
            mapping = {
                'gdp_pc_log': 'log_gdp_pc', 'mulpar': 'mulpar_n', 'hc_indep': 'hc_indep',
                'Equal': 'equal', 'Prop': 'Prop', 'Mixed': 'Mixed', 'feduni': 'feduni'
            }
            partner = mapping.get(suffix, suffix)
            if 'polar_n' in df.columns and partner in df.columns:
                df[name] = df['polar_n'] * df[partner]
                return True
        return False

    available = []
    for feat in features:
        if feat in df.columns:
            available.append(feat)
        else:
            if _ensure_interaction(feat):
                available.append(feat)
            else:
                print(f"Skipping missing feature: {feat}")

    if not available:
        raise KeyError(f"None of the requested features are present in the frame: {features}")

    X = add_constant(df[available])
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data


# Build a small synthetic DataFrame with necessary columns
np.random.seed(0)
N = 100
df = pd.DataFrame({
    'polar_n': np.random.rand(N),
    'Prop': np.random.randint(0,2,size=N),
    'Mixed': np.random.randint(0,2,size=N),
    'log_gdp_pc': np.random.rand(N)*10,
    'mulpar_n': np.random.rand(N),
    'feduni': np.random.rand(N),
    'equal': np.random.rand(N),
    'hc_indep': np.random.rand(N),
    'pol_of_society_n': np.random.rand(N)
})

# Do not pre-create interaction columns to test creation logic
features = ['polar_n', 'polar_X_Prop', 'polar_X_Mixed', 'polar_X_gdp_pc_log', 'polar_X_mulpar', 'polar_X_feduni',
            'polar_X_Equal', 'equal', 'feduni', 'mulpar_n', 'Prop', 'Mixed', 'log_gdp_pc', 'hc_indep', 'polar_X_hc_indep']

vif = compute_vif(df, features)
print(vif)
