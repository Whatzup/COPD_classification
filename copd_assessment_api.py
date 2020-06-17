import copd_predict as cp

try:
    from flask import Flask, request, jsonify
    from flask_restful import Resource, Api

    from flask_restful import reqparse

# from flask_limiter.util import get_remote_address
# from flask_limiter import Limiter

except Exception as e:
    print('Modules missing : {}'.format(e))
app = Flask(__name__)
api = Api(app)

# Limiter = Limiter(app, key_func=get_remote_address)
# Limiter.init_app(app)

parser = reqparse.RequestParser()
parser.add_argument('patient_input', type=str, required=True, help="please enter the COPD data")

# patient_input = {
#     'Breathing today': 'Very Bad',
#     'Breathing affecting daily activity': 'Much',
#     'Breathing affecting physical activity': 'Very Much',
#     'Cough today': 'Worse than usual',
#     'Daily Sputum production': 'Up to 15mls (1 tablespoon)',
#     'Colour of Sputum': 'Yellowish',
#     'How much tired today?': 'Worse than usual',
#     'Ankle swelling today?': 'NO',
#     'Chest Pain today?': 'YES',
#     'Heartburn today?': 'NO',
#     'Exacerbation of COPD': 'NO',
#     'Receiving treatment for COPD flare-up': 'NO'
# }


class MyApi(Resource):
    def __init__(self):
        self.patient_input = parser.parse_args().get('patient_input', None)

    def get(self):
        return jsonify(cp.predict_helath(self.patient_input))


api.add_resource(MyApi, '/')

if __name__ == "__main__":
    app.run(debug=True)
