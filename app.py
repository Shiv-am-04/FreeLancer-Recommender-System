from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load artifacts
with open("freelancers.pkl", "rb") as f:
    freelancers_df = pickle.load(f)
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("ranker_model.pkl", "rb") as f:
    ranker = pickle.load(f)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    required_skills = data.get("Required_Skills", [])
    budget = data.get("Budget", 1000)
    duration = data.get("Duration_Days", 10)

    job_skill_vec = mlb.transform([required_skills])[0]
    job_vector = list(job_skill_vec) + [budget, duration]

    scores = []
    for _, freelancer in freelancers_df.iterrows():
        fskill_vec = mlb.transform([freelancer["Skills"]])[0]
        fvec = list(fskill_vec) + [freelancer["Hourly_Rate"], freelancer["Rating"], freelancer["Completed_Projects"]]
        combined = job_vector + fvec
        scaled = scaler.transform([combined])
        score = ranker.predict(scaled)[0]
        scores.append((freelancer["Freelancer_ID"], score))

    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    return jsonify({"recommendations": [{"freelancer_id": fid, "score": round(score, 4)} for fid, score in top_k]})


if __name__ == '__main__':
    app.run(debug=True)