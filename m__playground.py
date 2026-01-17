import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df1 = pd.read_csv("./acme/datasets/scopus/scopus_1.csv")
    df2 = pd.read_csv("./acme/datasets/scopus/scopus_1.csv")
    df3 = pd.read_csv("./acme/datasets/scopus/scopus_1.csv")
    df = pd.concat([df1,df2,df2])

    df.shape
    return (df,)


@app.cell
def _(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.columns
    return


@app.cell
def _(df):
    df["title_lowercase"] = df.title.str.lower()
    df_cleaned = df.drop_duplicates(subset="title_lowercase")
    df_cleaned.shape, df_cleaned.title_lowercase
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df.to_csv("./acme/datasets/scopus/scopus.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
