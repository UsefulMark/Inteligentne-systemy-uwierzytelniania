from flask import Flask, render_template, request, redirect, url_for, session
import hashlib

app = Flask(__name__)
app.secret_key = '1234'  # Ważne: Użyj silnego klucza w produkcji!

@app.route('/')
def home():
    user = None
    if 'username' in session:
        user = session['username']
    return render_template('home.html', user=user)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hashowanie hasła
        with open('users.txt', 'a') as f:
            f.write(f'{username}:{hashed_password}\n')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        with open('users.txt', 'r') as f:
            users = f.readlines()
        for user in users:
            u, p = user.strip().split(':')
            if u == username and p == hashed_password:
                session['username'] = username
                return redirect(url_for('home'))
        return 'Invalid username or password'
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
