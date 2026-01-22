import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from lib.utils_base import configure_matplotlib_environment

    plt = configure_matplotlib_environment()
    return Path, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("./datasets/dataset_2/dataset.csv")
    df = df[df.year.gt(1919)]
    df = df[df.year.lt(2026)]
    df.head()
    return (df,)


@app.cell
def _(Path, df, plt):
    BASE_COLOR = "#3A4F43"
    COLOR_1 = "orange"
    COLOR_2 = "#00A2FF"
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.tick_params(color=BASE_COLOR, labelcolor=BASE_COLOR)
    ax.spines[:].set_color(BASE_COLOR)
    ax.xaxis.label.set_color(BASE_COLOR)
    ax.yaxis.label.set_color(BASE_COLOR)
    BASE_COLOR
    counts = df.year.value_counts(sort=False)
    counts.plot(ax=ax, c=COLOR_1, label="conteggio")
    counts.sort_index(ascending=True).rolling(10).mean().plot(ax=ax, color=COLOR_2, label="media mobile a 10 anni")
    ax.set_xlabel("anni")
    ax.set_ylabel("Nr pubblicazioni")
    ax.legend(frameon=False)
    fig.savefig(Path("./imgs/dataset_2_img_1.svg"), format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.show()
    return


@app.cell
def _(df, np, pd):
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
    return (data,)


@app.cell
def _(data, pd):
    last_decade = data[data.year.between(2020, 2025, inclusive="right")].groupby("country").size()
    previous_decade = data[data.year.between(2015, 2020, inclusive="both")].groupby("country").size()

    most_profilic_last_decade = last_decade.nlargest(10)
    most_profilic_previous_decade = previous_decade.reindex(most_profilic_last_decade.index)

    final = pd.concat([
        most_profilic_previous_decade,
        most_profilic_last_decade], axis=1, keys=["2016-2020", "2021-2025"]
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
def _():
    return


@app.cell
def _(df):
    "Number of articles withouth country information", round(df.country.eq("Unkown").sum() / df.shape[0] * 100, 1)
    return


@app.cell
def _(df):
    df[df.country.eq("Italy")]
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
