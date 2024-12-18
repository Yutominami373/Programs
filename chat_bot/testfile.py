from flask import Flask , render_template , request ,redirect  , render_template_string
from flask_sqlalchemy import SQLAlchemy


app= Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db= SQLAlchemy(app)
app.app_context().push()



@app.route('/')

def main():
    return render_template ('sui_error.html')

if __name__ == "__main__":
    #main()
    app.run(debug=True)