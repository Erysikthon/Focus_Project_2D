import matplotlib.pyplot as plt
from ModelWrapper import ModelWrapper

print("""\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

ooo        ooooo   .oooooo.   oooooooooo.   oooooooooooo ooooo             ooooooooooooo oooooooooooo  .oooooo..o ooooooooooooo oooooooooooo ooooooooo.   
`88.       .888'  d8P'  `Y8b  `888'   `Y8b  `888'     `8 `888'             8'   888   `8 `888'     `8 d8P'    `Y8 8'   888   `8 `888'     `8 `888   `Y88. 
 888b     d'888  888      888  888      888  888          888                   888       888         Y88bo.           888       888          888   .d88' 
 8 Y88. .P  888  888      888  888      888  888oooo8     888                   888       888oooo8     `"Y8888o.       888       888oooo8     888ooo88P'  
 8  `888'   888  888      888  888      888  888    "     888                   888       888    "         `"Y88b      888       888    "     888`88b.    
 8    Y     888  `88b    d88'  888     d88'  888       o  888       o           888       888       o oo     .d8P      888       888       o  888  `88b.  
o8o        o888o  `Y8bood8P'  o888bood8P'   o888ooooood8 o888ooooood8          o888o     o888ooooood8 8""88888P'      o888o     o888ooooood8 o888o  o888o 
      
            \n\n\n\n\n""")

TEST_VIDEO_IDS  = ['20231123_10min_OFT-BL_4025', '3279_21min_behaviour_2023-01-19T12_57_29', 'BehavioralCamera2023-02-23T10_23_42_shorter', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'BehavioralCamera2023-03-09T12_08_14', 'MBT1-M7', 'T11', 'T15', 'T4', 'T6']
TRUE_PATH = "./true"
PREDICTIONS_PATH_HGB = "./predictions/HGB"
OUTPUT_PATH_HGB = "./output/HGB"


HGB = ModelWrapper(name = "HGB", test_set = TEST_VIDEO_IDS, predictions_path = PREDICTIONS_PATH_HGB, true_path = TRUE_PATH, output_path = OUTPUT_PATH_HGB)
HGB.plot_confusion_matrix()
