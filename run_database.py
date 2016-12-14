import sqlite3 as lite
import sys
con = lite.connect('run.db')

with con:
    cur = con.cursor()
    cur.execute("CREATE TABLE Dataset(Id INT PRIMARY_, Folder TEXT, Filename INT, Timestamp INT, DigicamTriggerConfig TEXT, DigicamTriggerFreq INT, DigicamTr)")
    cur.execute("CREATE TABLE Dataset(Id INT, Folder TEXT, Filename INT, Timestamp INT, DigicamTriggerConfig TEXT, DigicamTriggerFreq INT, DigicamTr)")
    cur.execute("CREATE TABLE Dataset(Id INT, Folder TEXT, Filename INT, Timestamp INT, DigicamTriggerConfig TEXT, DigicamTriggerFreq INT, DigicamTr)")