from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# ----------------------------------------------------
# モデルの読み込み
# ----------------------------------------------------
try:
    model = joblib.load('activity_model.joblib')
    print("モデルの読み込みに成功しました。")
except Exception as e:
    model = None
    print(f"モデル読み込みエラー: {e}")

# ----------------------------------------------------
# 1. Webページ (index.html) を表示するルート
# ----------------------------------------------------
@app.route('/')
def home():
    # 'templates' フォルダ内の 'index.html' をブラウザに返す
    return render_template('index.html')

# ----------------------------------------------------
# 2. 予測API (/predict) のルート
# ----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_activity():
    if model is None:
        return jsonify({'error': 'モデルが読み込まれていません'}), 500

    try:
        # ブラウザから送信されたJSONデータを取得
        data = request.get_json(force=True)
        
        # 訓練時と同じ5つの特徴量
        features = [
            data['mean_acc'],
            data['std_acc'],
            data['max_acc'],
            data['min_acc'],
            data['energy']
        ]
        
        # モデルで予測
        final_features = [np.array(features)]
        prediction_id = model.predict(final_features)
        
        # 予測結果を分かりやすい形式に
        output_id = int(prediction_id[0])
        label_map = {0: '静止 (Stay)', 1: '歩行 (Walk)', 2: '走行 (Jog)'} # CSVの凡例に合わせてください
        output_label = label_map.get(output_id, '不明')

        # 予測結果をJSONでブラウザに返す
        return jsonify({
            'predicted_id': output_id,
            'predicted_label': output_label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ----------------------------------------------------
# (ローカルテスト用) 
# ----------------------------------------------------
if __name__ == '__main__':
    # python app.py で直接実行した時用
    app.run(debug=True, host='0.0.0.0', port=5000)