import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import ast
    return np, pd


@app.cell
def _(np, pd):
    df = pd.read_csv("./datasets/dataset_2/dataset.csv")

    df.country = df.country.fillna("Unkown")

    data = (
        pd.concat(
            [
                df.country.str.split(" - ", expand=True),
                df.year,
            ], axis=1)
            .melt(id_vars="year")
            .dropna(subset="value")
            .drop(columns="variable")
            .rename(columns={"value": "country"})
    )

    data = data.replace("Unkown", np.nan)
    data = data[data.year.gt(1919)]
    data = data[data.year.lt(2026)]
    data
    return data, df


@app.cell
def _(data, pd):
    last_decade = data[data.year.between(2020, 2025, inclusive="right")].groupby("country").size()
    previous_decade = data[data.year.between(2015, 2020, inclusive="both")].groupby("country").size()

    most_profilic_last_decade = last_decade.nlargest(10)
    most_profilic_previous_decade = previous_decade.reindex(most_profilic_last_decade.index)

    final = pd.concat([
        most_profilic_last_decade, 
        most_profilic_previous_decade], axis=1, keys=["2021-2025","2016-2020"]
    )

    final["pct_change"] = (
        final.loc[:, "2021-2025"]
            .div(final.loc[:, "2016-2020"])
            .mul(100)
            .round(1)
    )
    final.sort_values(by="2021-2025", ascending=False)
    return


@app.cell
def _(data):
    data.country.isna().sum() / data.shape[0]
    return


@app.cell
def _(df):
    df[df.country.eq("Unkown")].affiliations
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
