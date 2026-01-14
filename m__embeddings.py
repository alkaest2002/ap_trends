import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from lib.embeddings import get_batch_embeddings
    return get_batch_embeddings, pd


@app.cell
def _(pd):
    texts = pd.read_csv("./data/psycarticles_cleaned.csv")
    texts.head()
    return (texts,)


@app.cell
def _(get_batch_embeddings, texts):
    m = get_batch_embeddings(texts.iloc[:2,-1].to_list())
    return (m,)


@app.cell
def _(m):
    m
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
