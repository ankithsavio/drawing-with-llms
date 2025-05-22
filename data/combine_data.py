import json

import pandas as pd


def main():
    train_df = pd.read_csv("data/train.csv")
    train_question_df = pd.read_parquet("data/questions.parquet")

    train_df = pd.merge(train_df, train_question_df, how="left", on="id")
    train_df = train_df.groupby("id").apply(lambda df: df.to_dict(orient="list"))

    train_df = train_df.reset_index(name="qa")
    train_df["description"] = train_df.qa.apply(lambda qa: qa["description"][0])
    train_df["question"] = train_df.qa.apply(
        lambda qa: str(json.dumps(qa["question"], ensure_ascii=False))
    )
    train_df["answer"] = train_df.qa.apply(
        lambda qa: str(json.dumps(qa["answer"], ensure_ascii=False))
    )
    train_df["choices"] = train_df.qa.apply(
        lambda qa: str(
            json.dumps([x.tolist() for x in qa["choices"]], ensure_ascii=False)
        )
    )

    train_df = train_df.drop("qa", axis=1)

    train_df.to_csv("data/combined_train.csv")


if __name__ == "__main__":
    main()
