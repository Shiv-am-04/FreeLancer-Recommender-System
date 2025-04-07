import pandas as pd
import random
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import lightgbm as lgb
import pickle


freelancers_df = pd.read_csv(r'dataset/freelancers.csv')

jobs_df = pd.read_csv(r'dataset/jobs.csv')


freelancers_df["Skills"] = freelancers_df["Skills"].apply(ast.literal_eval)
jobs_df["Required_Skills"] = jobs_df["Required_Skills"].apply(ast.literal_eval)


# converting text data into binary

mlb = MultiLabelBinarizer()

freelancer_skills = mlb.fit_transform(freelancers_df["Skills"])

job_skills = mlb.transform(jobs_df["Required_Skills"])

mlb.classes_


"""*Combining the encoded features back to the dataframe*"""

freelancer_features = pd.DataFrame(freelancer_skills, columns=[f"FSkill_{s}" for s in mlb.classes_])
freelancer_features["Hourly_Rate"] = freelancers_df["Hourly_Rate"]
freelancer_features["Rating"] = freelancers_df["Rating"]
freelancer_features["Completed_Projects"] = freelancers_df["Completed_Projects"]
freelancer_features["Freelancer_ID"] = freelancers_df["Freelancer_ID"]

job_features = pd.DataFrame(job_skills, columns=[f"JSkill_{s}" for s in mlb.classes_])
job_features["Budget"] = jobs_df["Budget"]
job_features["Duration_Days"] = jobs_df["Duration_Days"]
job_features["Job_ID"] = jobs_df["Job_ID"]


# Job-Freelancer Interactions dataframe

interactions = []

for job in jobs_df.itertuples():
    selected_freelancers = random.sample(list(freelancers_df.Freelancer_ID), 20)
    hired = random.choice(selected_freelancers)
    for f in selected_freelancers:
        interactions.append({
            "Job_ID": job.Job_ID,
            "Freelancer_ID": f,
            "Is_Hired": int(f == hired)
        })

interactions_df = pd.DataFrame(interactions)


# Merge and build training set

merged_df = interactions_df.merge(job_features, on="Job_ID").merge(freelancer_features, on="Freelancer_ID")


X = merged_df.drop(columns=["Job_ID","Freelancer_ID","Is_Hired"])
y = merged_df["Is_Hired"]


# Grouping number of freelancers to job id.

job_group = merged_df.groupby("Job_ID").size().to_list()


# Scale numeric features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Model training

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

ranker.fit(X_scaled, y, group=job_group)


# Save files
with open("freelancers.pkl", "wb") as f:
    pickle.dump(freelancers_df, f)
with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("ranker_model.pkl", "wb") as f:
    pickle.dump(ranker, f)

print("Model training and data saving completed.")
