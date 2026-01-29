import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import itertools
    from os import getenv
    from pathlib import Path
    from bertopic import BERTopic
    from bertopic.backend import OpenAIBackend
    from dotenv import load_dotenv
    from kneed import KneeLocator
    from openai import OpenAI
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from lib.utils_pandas import get_topics_in_period
    from lib.utils_base import configure_matplotlib_environment

    load_dotenv();

    # Get configured plt env
    plt, colors = configure_matplotlib_environment()
    return (
        BERTopic,
        KneeLocator,
        OpenAI,
        OpenAIBackend,
        Path,
        colors,
        getenv,
        np,
        pd,
        plt,
    )


@app.cell
def _(Path):
    # Define paths
    DATASET_FOLDER = Path("./dataset/titles_with_excerpts_2/")
    OUT_FOLDER = Path("./out") / "sentence_transformers" / "all_mini_lm_l6_v2"
    EMBEDDING_FOLDER =  OUT_FOLDER / "embeddings"
    BERTOPIC_FOLDER = OUT_FOLDER / "bertopic"
    IMGS_FOLDER = OUT_FOLDER / "imgs"

    # Define other constants
    IMGS_FOLDER.exists()
    return BERTOPIC_FOLDER, DATASET_FOLDER, EMBEDDING_FOLDER, IMGS_FOLDER


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset_topic.csv")
    docs = df.doc.to_list()
    df.sample(5)
    return df, docs


@app.cell
def _(EMBEDDING_FOLDER, OpenAI, OpenAIBackend, getenv):
    # Get embedding_model_name
    with (EMBEDDING_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()

    # Init embedding model
    client = OpenAI(api_key=getenv("OPENAI_APIKEY"))
    embedding_model = OpenAIBackend(client=client, embedding_model="text-embedding-3-small")
    return (embedding_model,)


@app.cell
def _(BERTOPIC_FOLDER, BERTopic, docs, embedding_model, np, pd):
    # Load BERTopic related files
    topic_model = BERTopic.load(BERTOPIC_FOLDER, embedding_model=embedding_model)
    probs = np.load(file=BERTOPIC_FOLDER / "probs.npy")
    topic_model.update_topics(docs)
    topics = topic_model.topics_
    topics_info = pd.read_csv(BERTOPIC_FOLDER / "topic_info.csv")
    topics_info.sort_values(by="Topic")
    return (topics_info,)


@app.cell
def _(df):
    df[df.doc.str.contains("sui")].topic.value_counts()
    return


@app.cell
def _(df):
    df[df.topic.eq(49)]
    return


@app.cell
def _(IMGS_FOLDER, colors, plt, topics_info):
    def plot2():
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Colorize features
        ax.tick_params(color=colors["base"], labelcolor=colors["base"])
        ax.spines[:].set_color(colors["base"])
        ax.xaxis.label.set_color(colors["base"])
        ax.yaxis.label.set_color(colors["base"])

        topics_info.Count[1:].plot(kind="hist", color=colors["base"], label="Frequenza")
        ax.set_ylabel("Frequenza")
        ax.set_xlabel("Cluster")
        fig.savefig(IMGS_FOLDER / "img_2.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
        plt.show()

    plot2()
    return


@app.cell
def _(IMGS_FOLDER, KneeLocator, colors, plt, topics_info):
    def plot3():
        y = topics_info.Count[1:]
        x = range(1, len(y)+1)
        kneedle = KneeLocator(x, y, S=1, curve="convex", direction="decreasing")
        elbow = round(kneedle.elbow, 1)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Colorize features
        ax.tick_params(color=colors["base"], labelcolor=colors["base"])
        ax.spines[:].set_color(colors["base"])
        ax.xaxis.label.set_color(colors["base"])
        ax.yaxis.label.set_color(colors["base"])

        ax.axvline(elbow, linestyle="--", label="gomito")
        ax.plot(x,y, label="dim cluster", color="orange")
        ax.annotate(
            text=f"cluster {elbow}, dim {y[elbow]}", 
            color=colors["base"],
            xy=(elbow +1 , y[elbow]), 
            xytext=(elbow+5*5, y[elbow] + 5),
            arrowprops=dict(facecolor=colors["base"], edgecolor=colors["base"], arrowstyle='->,head_width=.15')
        )
        ax.legend(frameon=False)
        ax.set_ylabel("Frequenza")
        ax.set_xlabel("Cluster")
        fig.savefig(IMGS_FOLDER / "img_3.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
        plt.show()
        return elbow

    elbow = plot3()
    return (elbow,)


@app.cell
def _(elbow, topics_info):
    topics_info.nlargest(elbow +1, "Count").loc[:, ["Topic", "Count", "Representation"]]
    return


@app.cell
def _(IMGS_FOLDER, colors, plt):
    def plot4():
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Colorize features
        ax.tick_params(color=colors["base"], labelcolor=colors["base"])
        ax.spines[:].set_color(colors["base"])
        ax.xaxis.label.set_color(colors["base"])
        ax.yaxis.label.set_color(colors["base"])


        fig.savefig(IMGS_FOLDER / "img_4.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
        plt.show()

    plot4()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
