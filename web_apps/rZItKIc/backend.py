from flask import jsonify

@app.route('/ping')
def ping():
    return jsonify({'ok': True})
