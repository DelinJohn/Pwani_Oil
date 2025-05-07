import sqlite3
import streamlit as st

conn=sqlite3.connect('Output.db')
cursor=conn.cursor()


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table in tables:
    print(table[0])


cursor.execute('Select count(*) from image_and_text_output;')
table=cursor.fetchall()
print(table)

cursor.execute("PRAGMA table_info(image_and_text_output);")
columns = cursor.fetchall()
print(columns)
