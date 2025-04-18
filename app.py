from flask import Flask, render_template
from routes.predict import predict_bp
from routes.explain import explain_bp


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(explain_bp)

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, port=8080, use_reloader=False)
