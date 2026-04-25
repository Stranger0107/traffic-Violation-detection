import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0107@Bbs",
    database="traffic_system"
)

print("Connected to MySQL!")