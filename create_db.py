import sqlite3
import pandas as pd

conn = sqlite3.connect("database.db")
df = pd.read_csv("data/fact_consolidated.csv")
df.to_sql("fact_consolidated", conn, if_exists="replace", index=False)
conn.close()
print("Done! database.db created.")