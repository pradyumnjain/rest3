import pandas as pd
from flask import Flask,jsonify,request
import pickle

#load model
model = pickle.load(open('model.pkl','rb'))

#app
app = Flask(__name__)
#Flask constructor takes 
#the name of current module (__name__) as argument.

#routes
@app.route('/predict', methods =['POST'])
def predict():

	data = request.get_json(force=True)

	data.update((x,[y]) for x,y in data.items())
	data_df = pd.DataFrame.from_dict(data)
	#because dataframe type data works with logistic regression model


	result = model.predict(data_df)

	output = {'results': int(result[0])}

	return jsonify(output)

if __name__ == '__main__':
	app.run(debug = True)







