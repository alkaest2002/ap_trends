import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from lib.utils_base import configure_matplotlib_environment

    # Get configured plt env
    plt, colors = configure_matplotlib_environment()
    return Path, np, pd, plt, colors


@app.cell
def _(Path, colors):
    # Define PATHS
    DATASET_FOLDER = Path("./datasets/dataset_2/")
    IMGS_FOLDER = DATASET_FOLDER / "imgs"

    # Define other constants
    BASE_COLOR = colors["base"]
    COLOR_1 = colors["color_1"]
    COLOR_2 = colors["color_2"]
    return BASE_COLOR, COLOR_1, COLOR_2, DATASET_FOLDER, IMGS_FOLDER


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
    ax.set_xlabel("anni")
    xticks = range(1920, 2026, 10)
    xlabels = [f'{x}' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_ylabel("Nr pubblicazioni")
    ax.legend(frameon=False)

    # Save plot as svg
    fig.savefig(IMGS_FOLDER / "img_1.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
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
def _(data, pd):
    # Compute stats for last 10 years (5+5)
    y_2021_2025 = data[data.year.between(2021, 2025, inclusive="both")].groupby("country").size()
    y_2016_2020 = data[data.year.between(2016, 2020, inclusive="both")].groupby("country").size()

    # Compute most profilic countries
    most_profilic_2021_2025 = y_2021_2025.nlargest(10)
    most_profilic_2016_2020 = y_2016_2020.reindex(most_profilic_2021_2025.index)

    # Combine results
    final = pd.concat([
            most_profilic_2016_2020,
            most_profilic_2021_2025
        ], 
        axis=1, keys=["2016-2020", "2021-2025"]
    )

    # Compute % of change
    final["pct_change"] = (
        final.loc[:, "2021-2025"]
            .div(final.loc[:, "2016-2020"])
            .mul(100)
            .round(1)
    )

    # Sort data
    final.sort_values(by="2021-2025", ascending=False)
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
