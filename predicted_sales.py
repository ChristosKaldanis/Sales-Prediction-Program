import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 


# Φόρτωση του dataset
url = "your_file.csv"
names = ['stars', 'views']
dataset = pd.read_csv(url, sep=";", names=names)

# Αντικατάσταση του ',' με το '' σε όποια τιμή της στήλης views υπάρχει 
if dataset['views'].str.contains(',').any():
    dataset['views'] = dataset['views'].str.replace(',', '')

    dataset.to_csv('cleaned.csv')

# Μετατροπή δεδομένων σε αριθμητικά 
dataset['stars'] = pd.to_numeric(dataset['stars'], errors='coerce') 
dataset['views'] = pd.to_numeric(dataset['views'], errors='coerce')

# Στρογγυλοποίηση των τιμών της στήλης views στο ένα δεκαδικό ψηφίο
dataset['views'] = dataset['views'].round(1)

# Αφαίρεση γραμμών με ελλιπή δεδομένα
dataset.dropna(inplace=True)

# Προσθήκη τυχαίων κρατήσεων
np.random.seed(42)
dataset['sales'] = np.random.randint(0, 100, size=len(dataset))

# Εκτύπωση βασικών πληροφοριών
print(dataset.shape)
print(dataset.head(100)) # Εκτύπωση των πρώτων 100 σειρών
print(dataset.describe())

# Έλεγχος αν υπάρχουν αρκετά δείγματα για διαχωρισμό
if len(dataset) > 1:
    # Διαχωρισμός χαρακτηριστικών (X) και στόχου (y)
    X = dataset[['views', 'stars']]
    y = dataset['sales']

    # Διαχωρισμός του dataset σε σύνολα εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Εκπαίδευση του μοντέλου Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Πρόβλεψη στο σύνολο δοκιμής
    y_pred = model.predict(X_test)

    # Αξιολόγηση του μοντέλου
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Οπτικοποίηση των προβλέψεων
    plt.scatter(y_test, y_pred)
    plt.xlabel('Πραγματικές κρατήσεις')
    plt.ylabel('Προβλεπόμενες κρατήσεις')
    plt.title('Πραγματικές vs Προβλεπόμενες κρατήσεις')
    plt.savefig('predictions_plot.png')
    plt.show()

    # Ιστογράμματα των δεδομένων
    dataset.hist(bins=20, figsize=(10, 5), layout=(1, 3))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.show()

    # Scatter matrix των δεδομένων
    pd.plotting.scatter_matrix(dataset, figsize=(10, 10))
    plt.savefig('scatter_matrix.png')
    plt.show()
else:
    print("Το dataset δεν περιέχει αρκετά δείγματα για διαχωρισμό.")