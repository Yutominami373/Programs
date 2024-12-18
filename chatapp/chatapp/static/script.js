var socket = io('http://localhost:5000');

// メッセージを送信する関数
function sendMessage() {
    var username = document.getElementById('username').value.trim();
    var message = document.getElementById('myMessage').value.trim();
    if (username === '') {
        alert('Please enter your name.');
        return;
    }
    if (message === '') {
        alert('Please enter a message.');
        return;
    }
    var timestamp = new Date().toLocaleTimeString();  // 現在の時間を取得
    var formattedMessage = username + ': ' + message + ' at ' + timestamp;  // フォーマットを整える
    socket.emit('message', formattedMessage);  // サーバーにメッセージを送信
    document.getElementById('myMessage').value = '';  // 入力フィールドをクリア
}

// サーバーからメッセージを受信したときの処理
socket.on('message', function(msg) {
    var item = document.createElement('li');  // 新しいリストアイテムを作成
    item.textContent = msg;  // リストアイテムのテキストにメッセージを設定
    document.getElementById('messages').appendChild(item);  // メッセージリストにリストアイテムを追加
    window.scrollTo(0, document.body.scrollHeight);  // ビューを最下部にスクロール
});
