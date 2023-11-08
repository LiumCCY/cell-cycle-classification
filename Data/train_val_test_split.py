import splitfolders

input_folder1 = "/home/ccy/cellcycle/Data/FUCCI_SR"
output_folder1= "/home/ccy/cellcycle/Split"
splitfolders.ratio(input_folder1, output=output_folder1, seed=42, ratio=(0.7, 0.2, 0.1))