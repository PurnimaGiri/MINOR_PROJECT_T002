import pymysql
try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root123', # Use your shell password here
        db='hoams_db'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT name, role FROM users WHERE email = 'admin@cityhospital.com'")
    result = cursor.fetchone()
    print(f"SUCCESS! Python sees: {result}")
except Exception as e:
    print(f"FAILURE: {e}")