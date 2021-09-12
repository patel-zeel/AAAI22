import pandas as pd
import sys
path_1 = str(sys.argv[1]) # Path to time-series 1
path_2 = str(sys.argv[2]) # Path to time-series 2
TS1 = pd.read_csv(path_1)
TS2 = pd.read_csv(path_2)

TS1["abs_diff"] = (TS1.PM25_Concentration - TS1.pred_PM25).abs()
TS2["abs_diff"] = (TS2.PM25_Concentration - TS2.pred_PM25).abs()
win1 = (TS2["abs_diff"] > TS1["abs_diff"]).sum()
win2 = (TS1["abs_diff"] > TS2["abs_diff"]).sum()

print("% win Series 1:", 100*win1/(win1+win2))
print("% win Series 2:", 100*win2/(win1+win2))