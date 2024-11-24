import json
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
def load_dataset():
    with open('data/dataset.json', 'r') as f:
        return json.load(f)

# Prepare the dataset for training (using one-hot encoding for simplicity)
def preprocess_data():
    data = load_dataset()
    questions = []
    answers = []
    courses = []
    modules = []
    submodules = []
    
    for entry in data:
        questions.append(entry["question"])
        answers.append(entry["answer"])
        courses.append(entry["course"])
        modules.append(entry["module"])
        submodules.append(entry["submodule"])
    
    # We would need to encode answers and convert to numeric features
    # Here, for simplicity, we assume the model could be trained on the answers directly
    return questions, answers, courses, modules, submodules

# Train the Random Forest Model (simplified)
def train_model():
    questions, answers, courses, modules, submodules = preprocess_data()
    
    # In practice, we would preprocess text data here and train a model
    # For simplicity, let's assume we encode answers directly into numeric labels
    answer_labels = list(set(answers))
    answer_dict = {answer: idx for idx, answer in enumerate(answer_labels)}
    
    # Prepare the training data
    X = [[answer_dict[ans]] for ans in answers]  # Features (encoded answers)
    y = [idx for idx in range(len(answers))]  # Labels (indices for course, module, submodule)
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save the model to a file
    joblib.dump(model, 'model.pkl')
    joblib.dump(answer_dict, 'answer_dict.pkl')

# Load the trained model
def load_model():
    model = joblib.load('model.pkl')
    answer_dict = joblib.load('answer_dict.pkl')
    return model, answer_dict


def evaluate_answers(user_answers):
    model, answer_dict = load_model()  # Load the model and answer dictionary
    correct_count = 0
    feedback = []

    # Iterate through user answers and compare with the true answers
    for idx, user_answer in enumerate(user_answers):
        true_answer = load_dataset()[idx]["answer"]
        question = load_dataset()[idx]["question"]  # Get the question for this index

        # If the answer is correct
        if user_answer == true_answer:
            correct_count += 1
        else:
            # If the answer is incorrect, add both question and feedback info
            entry = load_dataset()[idx]
            feedback.append({
                "question": question,  # Include the question in the feedback
                "user_answer": user_answer,  # User's answer
                "correct_answer": true_answer,  # Correct answer
                "course": entry["course"],
                "module": entry["module"],
                "submodule": entry["submodule"]
            })

    # Return result based on the correctness of all answers
    if correct_count == len(user_answers):
        return {"result": "Congratulations! All answers are correct."}
    else:
        return {"result": "Some answers are incorrect.", "feedback": feedback}
