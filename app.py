from flask import Flask, request, jsonify  #Falsk is for object initae
from src.pipelines.prediction_pipeline import CustomClass, PredicitonPipeline

app = Flask(__name__) # Flass apppplication instiate done

@app.route("/predict", methods=["POST"])  # api flass server a request sent, here post request (not http requst)

#normamy fast appi -- http request send

def predict():
    try:
        #get json data format request
        json_data = request.get_json()

        #create a custom class instancw with json data
        data = CustomClass(
            age=int(json_data.get("age")),
            workclass=int(json_data.get("workclass")),
            education_num=int(json_data.get("education_num")),
            marital_status=int(json_data.get("marital_status")),
            occupation=int(json_data.get("occupation")),
            relationship=int(json_data.get("relationship")),
            race= int(json_data.get("race")),
            sex= int(json_data.get("sex")),
            capital_gain=int(json_data.get("capital_gain")),
            capital_loss=int(json_data.get("capital_loss")),
            hours_per_week=int(json_data.get("hours_per_week")),
            native_country=int(json_data.get("native_country"))
        )
        #get prediction using predictionpiple class, in return you get pred result
        final_data = data.get_data_DataFrame()
        pipeline_prediction = PredicitonPipeline()
        pred = pipeline_prediction.predict(final_data)

        #Return prediction result as a json object
        return jsonify({
            "status": "success",
            "prediction": int(pred[0]),
            "income_category": "<=50K" if pred[0] == 0 else ">50K"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)  # this command for app instiate



