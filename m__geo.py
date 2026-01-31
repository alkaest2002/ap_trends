import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from lib.utils_base import configure_matplotlib_environment
    from lib.utils_base import get_eu_countries

    # Get configured plt environment
    plt, colors = configure_matplotlib_environment()

    # Define constants
    BASE_COLOR = colors["base"]
    COLOR_1 = colors["color_1"]
    COLOR_2 = colors["color_2"]

    EU_COUNTRIES = get_eu_countries()
    return BASE_COLOR, COLOR_1, COLOR_2, EU_COUNTRIES, Path, np, pd, plt


@app.cell
def _(Path):
    # Define PATHS
    DATASET_FOLDER = Path("./dataset/titles_with_excerpts_2/")
    IMGS_FOLDER = Path("out/_base") / "imgs"
    OTHER_FOLDER = Path("out/_base") / "other"

    for folder in {IMGS_FOLDER, OTHER_FOLDER}:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
    return DATASET_FOLDER, IMGS_FOLDER, OTHER_FOLDER


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")

    # Restrict period to 1900-2025
    df = df[df.year.between(1920, 2025)]
    df.shape
    return (df,)


@app.cell
def _(BASE_COLOR, COLOR_1, COLOR_2, IMGS_FOLDER, df, plt):
    # Init figure
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Colorize features
    ax.tick_params(color=BASE_COLOR, labelcolor=BASE_COLOR)
    ax.spines[:].set_color(BASE_COLOR)
    ax.xaxis.label.set_color(BASE_COLOR)
    ax.yaxis.label.set_color(BASE_COLOR)

    # Plot data
    counts = df.year.value_counts(sort=False)
    counts.plot(ax=ax, c=COLOR_1, label="conteggio")

    (
        counts
            .reindex(range(counts.index.min(),counts.index.max()+1), fill_value=0)
            .sort_index(ascending=True)
            .rolling(10).mean()
            .plot(ax=ax, color=COLOR_2, label="media mobile a 10 anni")
    )

    # Customize plot
    xticks = range(1920, 2026, 10)
    xlabels = [f'{x}' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_xlabel("Anni", labelpad=10)
    ax.set_ylabel("Nr pubblicazioni", labelpad=0)
    ax.legend(frameon=False)

    # Save plot as svg
    fig.savefig(IMGS_FOLDER / "img_publications_per_year.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.show()
    return


@app.cell
def _(df, np, pd):
    # Fill articles with no country with Unknown
    df.country = df.country.fillna("Unkown")

    # Separate multi-country articles
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

    #
    data = data.replace("Unkown", np.nan)
    return (data,)


@app.cell
def _(EU_COUNTRIES, OTHER_FOLDER, data, pd):
    # Compute stats for MOST PROLIFIC
    y_most_recent = data[data.year.between(2001, 2025, inclusive="both")].groupby("country").size()
    y_least_recent = data[data.year.between(1925, 2000, inclusive="both")].groupby("country").size()

    # Compute most profilic countries
    most_recent_profilic = y_most_recent.nlargest(50)
    least_recent_profilic = y_least_recent.nlargest(50)

    # Combine results
    final = pd.concat([
            most_recent_profilic,
            least_recent_profilic
        ], 
        axis=1, keys=["most_recent", "least_recent"]
    ).reset_index(drop=False)

    # Compute % of change
    final["pct_change"] = (
        final.loc[:, "most_recent"]
            .div(final.loc[:, "least_recent"])
            .mul(100)
            .round(1)
    )

    # Create CSV tables
    for nlargest_column in {"most_recent", "pct_change"}:
        (
            final
                .nlargest(10, columns=nlargest_column)
                .loc[~final.country.isin(EU_COUNTRIES)]
                .iloc[:5, :]
                .set_index("country")
                .apply(lambda x: pd.to_numeric(x, downcast="integer"))
                .to_csv(OTHER_FOLDER / f"largest_{nlargest_column}.csv")
        )
    return


@app.cell
def _(df):
    "Number of articles withouth country information", round(df.country.eq("Unkown").sum() / df.shape[0] * 100, 1)
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
