from flask import Flask , render_template , request ,redirect  , render_template_string
from flask_sqlalchemy import SQLAlchemy
from llama_reader import AI_Assistant

app= Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db= SQLAlchemy(app)
app.app_context().push()
chat_bot= AI_Assistant()

class Log_SQL(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    def __repr__(self) -> str:
        return '<log %r>' % self.id

user_name = Log_SQL(content="user :")
bot_name = Log_SQL(content="not llama3 :")
space = Log_SQL(content=" ")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat_room')
def chat_test():
    return render_template('chat_room.html')

@app.route('/sui')
def sui():
    return render_template('sui_error.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        log_content = request.form['content']
        new_log = Log_SQL(content=log_content)


        try:
            user_name = Log_SQL(content="user :")
            db.session.add(user_name)
            db.session.commit()
            db.session.add(new_log)
            db.session.commit()
            response = chat_bot.receive_user_input(log_content)
            bot_name = Log_SQL(content="not llama3 :")
            db.session.add(bot_name)
            db.session.commit()
            bot_log = Log_SQL(content=response)
            db.session.add(bot_log)
            db.session.commit()
            return redirect('/chat')
        except:
            return render_template('sui_error.html')
    else:
        logs = Log_SQL.query.order_by(Log_SQL.id).all()
        return render_template('chat.html', logs=logs)

if __name__ == "__main__":
    app.run(debug=True)
    
    