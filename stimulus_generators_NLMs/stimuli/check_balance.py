import argparse
import pandas as pd
from plotnine import ggplot, aes, facet_wrap, geom_bar, theme, geom_text

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--features",
    nargs="+",
    type=str,
    default="all",
    help="List of features to check.",
)
parser.add_argument(
    "--csv", help=".csv file containing the stimuli to check.", required=True
)
args = parser.parse_args()

df = pd.read_csv(args.csv, index_col=[0])
df = df.fillna("NA")

if args.features == "all":
    args.features = df.columns[df.apply(lambda x: 2 <= x.nunique() <= 20)]
else:
    args.features = [col for col in df.columns if col in args.features]

df_filtered = df[args.features]

melted_df = (
    df_filtered.melt().groupby(["variable", "value"]).size().reset_index(name="counts")
)

melted_df["value"] = melted_df["value"].replace({1: "1", 0: "0"})
melted_df["value"] = melted_df["value"].astype(str)

plot = (
    ggplot(melted_df, aes(x="value", y="counts", fill="factor(value)"))
    + geom_bar(stat="identity")
    + geom_text(aes(label="counts"), position="identity", va="bottom", size=8)
    + facet_wrap("~variable", scales="free_x")
    + theme(legend_position="none")
)

print(plot)
