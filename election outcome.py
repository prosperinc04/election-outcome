import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

file_path = "/content/pseudo_facebook.csv"

# Columns to extract
columns_to_extract = ['userid', 'age', 'gender', 'likes', 'friendships_initiated']

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Filter data for individuals aged 18 and above
filtered_data = data[data['age'] >= 18][columns_to_extract]

# Generate a synthetic target variable 'voting_likelihood'
# Let's assume it's a random binary classification (0 or 1)
filtered_data['voting_likelihood'] = np.random.randint(0, 2, size=len(filtered_data))

# Define features and target
features = ['likes', 'friendships_initiated']
target = 'voting_likelihood'

# Split data into features and target variable
X = filtered_data[features]
y = filtered_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Assuming 'filtered_data' is your dataset containing 'likes' and 'friendships_initiated' columns
# Calculate mean values for 'likes' and 'friendships_initiated'
mean_likes = filtered_data['likes'].mean()
mean_friendships = filtered_data['friendships_initiated'].mean()

# Create new data using mean values as defaults for a new user
new_data = [[mean_likes, mean_friendships]]

# Predict the voting likelihood for the new user
predicted_likelihoods = clf.predict_proba(new_data)

# Extract predicted probabilities for each candidate
likelihood_candidate_1 = predicted_likelihoods[0][0]  # Probability for candidate 1
likelihood_candidate_2 = predicted_likelihoods[0][1]  # Probability for candidate 2

# Probabilities to tailor the campaign or messaging for the user:
if likelihood_candidate_1 > likelihood_candidate_2:
    # Target messaging for candidate 1 supporters
    print("Encourage user to vote for Candidate 1.")
elif likelihood_candidate_2 > likelihood_candidate_1:
    # Target messaging for candidate 2 supporters
    print("Encourage user to vote for Candidate 2.")
else:
    # No clear preference, a neutral or persuasive approach can be used
    print("Engage user with neutral content or persuasive messaging.")