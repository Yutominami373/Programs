from flask import Flask, render_template
from flask_socketio import SocketIO, send  # send関数をインポート
from flask_cors import CORS  # CORSのためのインポート

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # シークレットキーを設定
CORS(app)  # CORSをアプリ全体に適用
socketio = SocketIO(app, cors_allowed_origins="*")  # すべてのオリジンからの接続を許可

@app.route('/')
def index():
    return render_template('index.html')  # index.htmlをクライアントに提供

@socketio.on('message')
def handleMessage(msg):
    print('Received message:', msg)  # コンソールに受け取ったメッセージを表示
    send(msg, broadcast=True)  # 受け取ったメッセージを全クライアントにブロードキャスト

if __name__ == '__main__':
    socketio.run(app, debug=True)  # デバッグモードでアプリを実行
