import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    from bertopic import BERTopic

    from lib.bertopic.model_dataset_1 import embedding_model
    return BERTopic, Path, embedding_model


@app.cell
def _(BERTopic, Path, embedding_model):
    DATASET_FOLDER = Path("./datasets/dataset_1/")

    bertopic_model = BERTopic.load(DATASET_FOLDER / "bertopic/", embedding_model=embedding_model)
    return (bertopic_model,)


@app.cell
def _(bertopic_model):
    bertopic_model.get_topic_tree()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
