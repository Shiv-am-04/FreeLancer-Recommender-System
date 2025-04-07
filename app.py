from flask import Flask, request,render_template_string
import pickle
import warnings

warnings.filterwarnings(action='ignore')

app = Flask(__name__)

# Load artifacts
with open(r"resources/freelancers.pkl", "rb") as f:
    freelancers_df = pickle.load(f)
with open(r"resources/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open(r"resources/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(r"resources/ranker_model.pkl", "rb") as f:
    ranker = pickle.load(f)

template = """
<!DOCTYPE html>
<html>
<head>
    <title>Freelancer Recommender</title>
</head>
<body style="font-family: Arial; padding: 40px;">
    <h2>Freelancer Recommender</h2>
    <form method="POST">
        <label>Required Skills (comma-separated):</label><br>
        <input type="text" name="skills" required><br><br>

        <label>Budget:</label><br>
        <input type="number" name="budget" required><br><br>

        <label>Duration (in days):</label><br>
        <input type="number" name="duration" required><br><br>

        <input type="submit" value="Get Recommendations">
    </form>

    {% if top_freelancers %}
        <h3>Top 5 Freelancers:</h3>
        <table border="1" cellpadding="8">
            <tr>
                <th>ID</th>
                <th>Score</th>
            </tr>
            {% for fid, score in top_freelancers %}
            <tr>
                <td>{{ fid }}</td>
                <td>{{ score }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def recommend_freelancers_for_job():
    top_freelancers = None

    if request.method == "POST":
        try:
            skills = [s.strip() for s in request.form["skills"].split(",")]
            budget = float(request.form["budget"])
            duration = float(request.form["duration"])

            job_skills_vec = mlb.transform(skills)[0]
            job_vector = list(job_skills_vec) + [budget, duration]

            scores = []
            for _, freelancer in freelancers_df.iterrows():
                freelancer_skills_vec = mlb.transform([freelancer["Skills"]])[0]
                freelancer_vector = list(freelancer_skills_vec) + [
                    freelancer["Hourly_Rate"],
                    freelancer["Rating"],
                    freelancer["Completed_Projects"]
                ]

                input_vec = job_vector + freelancer_vector
                input_scaled = scaler.transform([input_vec])
                score = ranker.predict(input_scaled)[0]
                scores.append((freelancer["Freelancer_ID"], score))

            top_freelancers = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3>"

    return render_template_string(template, top_freelancers=top_freelancers)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
