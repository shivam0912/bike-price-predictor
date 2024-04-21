from flask import Flask, render_template, request,abort
import pickle
import pandas as pd 

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')



# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from HTML form
        try:
            year = int(request.form['year'])
            seller_type = (request.form['seller_type'])
            owner = int(request.form['owner'])
            km_driven = int(request.form['km_driven'])
            brand=(request.form['brand'])
            model = (request.form['model'])
            ex_showroom = int(request.form['ex-showroom'])
        except ValueError:
            # Handle invalid input
            abort(400, "Invalid input: Year, Seller Type, Owner, KM Driven, and Ex Showroom Price must be valid numbers.")

        # Create a DataFrame with user input
        user_input = pd.DataFrame({
            'Brand': [brand],
            'Model': [model],
            'Year': [year],
            'Seller_Type': [1 if seller_type == 'Individual' else 0],
            'Owner': [owner],
            'KM_Driven': [km_driven],
            'Ex_Showroom_Price': [ex_showroom]
        })

        # Make prediction
        try:
            predicted_price = lr_model.predict(user_input.drop(['Brand', 'Model'], axis=1))
            print(predicted_price)
        except Exception as e:
            # Handle prediction error
            abort(500, f"Prediction failed: {str(e)}")
        
        return render_template('index.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
