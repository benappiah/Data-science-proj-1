# Import the packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1) (5 points) Read the vertebrate.csv data
data = pd.read_csv("C:/Users/Owner/Desktop/844/vertebrate(2).csv")

# 2) (15 points) The number of records is limited. Convert the data into a binary classification: mammals versus non-mammals
# Hint: ['fishes','birds','amphibians','reptiles'] are considered 'non-mammals'
data['Class'] = data['Class'].apply(lambda x: 'mammals' if x == 'mammals' else 'non-mammals')
unique_attributes = data['Class'].unique()
print(unique_attributes)
# 3) (15 points) We want to classify animals based on the attributes: Warm-blooded,Gives Birth,Aquatic Creature,Aerial Creature,Has Legs,Hibernates
# For training, keep only the attributes of interest, and seperate the target class from the class attributes
X = data.iloc[:, 1:7]
y = data['Class']

# 4) (10 points) Create a decision tree classifier object. The impurity measure should be based on entropy. Constrain the generated tree with a maximum depth of 3
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# 5) (10 points) Train the classifier
clf.fit(X, y)
# 6) (25 points) Suppose we have the following data
testData = [['lizard',0,0,0,0,1,1,'non-mammals'],
           ['monotreme',1,0,0,0,1,1,'mammals'],
           ['dove',1,0,0,1,1,0,'non-mammals'],
           ['whale',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)
print(testData)
# Prepare the test data and apply the decision tree to classify the test records.
# Extract the class attributes and target class from 'testData'
Xtest = testData.iloc[:, 1:7]
ytest = testData['Class']
# Hint: The classifier should correctly label the vertabrae of 'testData' except for the monotreme

# 7) (10 points) Compute and print out the accuracy of the classifier on 'testData'
accuracy = clf.score(Xtest, ytest)
print("Accuracy:", accuracy)

# 8) (10 points) Plot your decision tree
plt.figure(figsize=(6, 6))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
