import sqlite3 as lite
import sys
con = lite.connect('sst1m.db')


cts = cts.CTS('/data/software/CTS/config/cts_config_%d.cfg'%(0),
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)