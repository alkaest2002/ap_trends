import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import marimo as mo
    from pathlib import Path
    from bertopic import BERTopic
    from dotenv import load_dotenv
    from kneed import KneeLocator
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from lib.utils_pandas import get_topics_in_period
    from lib.utils_base import configure_matplotlib_environment

    # Get configured plt env
    plt, colors = configure_matplotlib_environment()
    return BERTopic, KneeLocator, Path, colors, np, pd, plt


@app.cell
def _(Path):
    # Define paths
    DATASET_FOLDER = Path("./dataset/titles_with_excerpts_2/")
    OUT_FOLDER = Path("./out") / "sentence_transformers" / "all_MiniLM_L6_v2"
    EMBEDDING_FOLDER =  OUT_FOLDER / "embeddings"
    BERTOPIC_FOLDER = OUT_FOLDER / "bertopic"
    IMGS_FOLDER = OUT_FOLDER / "imgs"
    OTHER_FOLDER = OUT_FOLDER / "other"

    for folder in {IMGS_FOLDER, OTHER_FOLDER}:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
    return (
        BERTOPIC_FOLDER,
        DATASET_FOLDER,
        EMBEDDING_FOLDER,
        IMGS_FOLDER,
        OTHER_FOLDER,
    )


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset_topic.csv")
    docs = df.doc.to_list()
    df.sample(5)
    return df, docs


@app.cell
def _(EMBEDDING_FOLDER):
    # Get embedding_model_name
    with (EMBEDDING_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()
    embedding_model_name
    return (embedding_model_name,)


@app.cell
def _(BERTOPIC_FOLDER, BERTopic, docs, embedding_model_name, np, pd):
    # Load BERTopic related files
    topic_model = BERTopic.load(BERTOPIC_FOLDER, embedding_model=embedding_model_name)
    probs = np.load(file=BERTOPIC_FOLDER / "probs.npy")
    topic_model.update_topics(docs)
    topics = topic_model.topics_
    topics_info = pd.read_csv(BERTOPIC_FOLDER / "topic_info.csv")
    topics_info.sort_values(by="Topic")
    return (topics_info,)


@app.cell
def _(df):
    # Explore topics
    df[df.topic.eq(9)]
    return


@app.cell
def _(IMGS_FOLDER, KneeLocator, colors, plt, topics_info):
    def plot_elbow():

        # Create figure and axex
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Colorize features
        ax.tick_params(color=colors["base"], labelcolor=colors["base"])
        ax.spines[:].set_color(colors["base"])
        ax.xaxis.label.set_color(colors["base"])
        ax.yaxis.label.set_color(colors["base"])

        # Compute data
        y = topics_info.Count[1:]
        x = range(1, len(y)+1)
        kneedle = KneeLocator(x, y, S=2, curve="convex", direction="decreasing")
        elbow = round(kneedle.elbow, 0)

        # Ploat data
        ax.axvline(elbow, linestyle="--")
        ax.plot(x,y, color="orange")

        # Customize plot
        ax.annotate(
            text=f"ID Cluster: {elbow}, Nr. Pubblicazioni: {y[elbow]}", 
            color=colors["base"],
            xy=(elbow +1 , y[elbow]+1.5), 
        )
        ax.legend(frameon=False)
        ax.set_ylabel("Nr Pubblicazioni", labelpad=10)
        ax.set_xlabel("ID Cluster")
        fig.savefig(IMGS_FOLDER / "img_elbow.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
        plt.show()
    
        return y[elbow]

    min_cluster_size = plot_elbow()
    min_cluster_size
    return (min_cluster_size,)


@app.cell
def _(IMGS_FOLDER, colors, df, plt):
    def plot_topic_trajectories():

        # Create figure and axes
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Colorize features
        ax.tick_params(color=colors["base"], labelcolor=colors["base"])
        ax.spines[:].set_color(colors["base"])
        ax.xaxis.label.set_color(colors["base"])
        ax.yaxis.label.set_color(colors["base"])

        # Compute data
        data = (
            df
                .groupby(["topic","year"])
                .size()
                .drop(-1, level=0, axis=0)
                .reindex(level=0, axis=0)
        )

        # Plot data
        for g_label, g_data in data.groupby(axis=0, level=0):
            ax.plot(g_data.index.get_level_values(-1), g_data.index.get_level_values(0), color=colors["color_1"])
    

        # Customize plot
        ax.set_ylabel("ID Cluster", labelpad=0)
        ax.set_xlabel("Anni")

        # Persist
        fig.savefig(IMGS_FOLDER / "img_topic_trajectories.svg", format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)
        plt.show()

    plot_topic_trajectories()
    return


@app.cell
def _(OTHER_FOLDER, min_cluster_size, topics_info):
    # Create list of most important clusters
    def get_list_of_most_important_cluster():

        # Get most importante clusters
        # i.e., with the minimun cluster size
        topics = (topics_info
            .loc[topics_info.Count.gt(min_cluster_size), :]
            .iloc[1:, :]
            .loc[:, ["Topic","Representation"]]
            .apply(lambda x: f"{x.Topic} - {x.Representation[2:-2]}", axis=1)
            .to_list()
        )
    
        # Persist
        with (OTHER_FOLDER / "consolidated_topics.txt").open("w") as fout:
            fout.write("\n".join(topics))

    get_list_of_most_important_cluster()
    return


@app.cell
def _(OTHER_FOLDER, df, topics_info):
    # Create list of emerging clusters
    def get_list_of_emergent_cluster():

        year_to_split = 2000
        most_recent_topics = df.loc[df.year.ge(year_to_split), "topic"].unique()
        least_recent_topics = df.loc[df.year.lt(year_to_split), "topic"].unique()
        emerging_topics = sorted([t for t in most_recent_topics if t not in least_recent_topics])

        # Get emerging clusters
        topics = (
            topics_info
                .loc[topics_info.Topic.isin(emerging_topics),  ["Topic","Representation"]]
                .apply(lambda x: f"{x.Topic} - {x.Representation[2:-2]}", axis=1)
                .to_list()
        )
    
        # Persist list
        with (OTHER_FOLDER / "emerging_topics.txt").open("w") as fout:
            fout.write("\n".join(topics))

    get_list_of_emergent_cluster()
    return


@app.cell
def _(OTHER_FOLDER, df, topics_info):
    # Get Topics per selected countries in 2025
    def get_topics_per_country():
        for country in ["EU", "United States", "China"]:
            with (OTHER_FOLDER / f"{country}_topics.txt").open("w") as fout:
                topics_list = (
                    (df.loc[df.year.between(2025, 2025) & df.country.str.contains(country)]
                        .merge(topics_info, left_on="topic", right_on="Topic")
                        ["Representation"].unique())
                )
                fout.write("\n".join([ t[2:-2] for t in topics_list]))

    get_topics_per_country()
    return


@app.cell
def _(OTHER_FOLDER, df, topics_info):
    # Get Common Topics per selected countries in 2025
    def get_common_topics_per_country():
        for country, other_countries in [
            ("EU", ("United States", "China")),
            ("United States", ("EU", "China")),
            ("China", ("United States", "EU")),
        ]:
            with (OTHER_FOLDER / f"{country}_common_topics.txt").open("w") as fout:
                df_year = df.loc[df.year.between(2025, 2025)]
                other_countries_topics = (
                    df_year.loc[df_year.country.str.contains(f"{other_countries[0]}|{other_countries[1]}", regex=True, na=False), "topic"].
                        to_list()
                )
                topics_list = (
                    (df_year.loc[df_year.country.str.contains(country) & (df_year.topic.isin(list(set(other_countries_topics))))]
                        .merge(topics_info, left_on="topic", right_on="Topic")
                        ["Representation"].unique())
                )
            
                fout.write("\n".join([ t[2:-2] for t in topics_list]))

    get_common_topics_per_country()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
