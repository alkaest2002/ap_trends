import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df_title_only = pd.read_csv("./datasets/dataset_2/openai_small/titles_only/dataset_topic.csv")
    df_title_with_excerpts = pd.read_csv("./datasets/dataset_2/openai_small/titles_with_excerpts/dataset_topic.csv")
    df_title_with_abstracts = pd.read_csv("./datasets/dataset_2/openai_small/titles_with_abstracts/dataset_topic.csv")
    return df_title_only, df_title_with_abstracts, df_title_with_excerpts


@app.cell
def _(df_title_only):
    df_title_only.topic.value_counts(normalize=True).nlargest(1).squeeze()
    return


@app.cell
def _(df_title_with_excerpts):
    df_title_with_excerpts.topic.value_counts(normalize=True).nlargest(1).squeeze()
    return


@app.cell
def _(df_title_with_abstracts):
    df_title_with_abstracts.topic.value_counts(normalize=True).nlargest(1).squeeze()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
