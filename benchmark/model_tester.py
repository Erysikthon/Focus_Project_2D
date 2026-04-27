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
TRUE_FOLDER = "./true"
COLUMN_NAMES = {0 : "background", 1 : "Supportedrearing", 2 : "Unsupportedrearing", 3 : "Grooming", 4 : "Digging"}
OUTPUT_FOLDER = "./output"
SMOOTHING = "gap"
SMOOTHING_WINDOW = 5
CONFUSION_MATRIX_NORMALIZE = True
PREDICTIONS_FOLDER_HGB = "./predictions/HGB"
PREDICTIONS_FOLDER_CNN_TRANSFORMER = "./predictions/CNN_Transformer"
PREDICTIONS_FOLDER_OLD_MODEL = "./predictions/old_model"
PREDICTIONS_FOLDER_TCNN = "./predictions/TCNN"

old_model = ModelWrapper(name = "Old Model", test_set = TEST_VIDEO_IDS, predictions_folder = PREDICTIONS_FOLDER_OLD_MODEL, true_folder = TRUE_FOLDER, output_folder = OUTPUT_FOLDER, column_names = COLUMN_NAMES, smoothing = SMOOTHING, smoothing_window = SMOOTHING_WINDOW)
old_model.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)

hgb = ModelWrapper(name = "HGB", test_set = TEST_VIDEO_IDS, predictions_folder = PREDICTIONS_FOLDER_HGB, true_folder = TRUE_FOLDER, output_folder = OUTPUT_FOLDER, column_names = COLUMN_NAMES, smoothing = SMOOTHING, smoothing_window = SMOOTHING_WINDOW)
hgb.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)

CNN_transformer = ModelWrapper(name = "CNN Transformer", test_set = TEST_VIDEO_IDS, predictions_folder = PREDICTIONS_FOLDER_CNN_TRANSFORMER, true_folder = TRUE_FOLDER,column_names = COLUMN_NAMES, output_folder = OUTPUT_FOLDER, smoothing = SMOOTHING, smoothing_window = SMOOTHING_WINDOW)
CNN_transformer.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)

TCNN = ModelWrapper(name = "TCNN", test_set = TEST_VIDEO_IDS, predictions_folder = PREDICTIONS_FOLDER_TCNN, true_folder = TRUE_FOLDER,column_names = COLUMN_NAMES, output_folder = OUTPUT_FOLDER, smoothing = SMOOTHING, smoothing_window = SMOOTHING_WINDOW)
TCNN.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)

# How to see total instance count
pred_behaviors = [0, 0, 0, 0, 0]
true_behaviors = [0, 0, 0, 0, 0]

for dictionary in CNN_transformer.pred_behavior_count:
    for behavior in range(0, len(CNN_transformer.column_names)):
        pred_behaviors[behavior] +=  dictionary[behavior]
for dictionary in CNN_transformer.true_behavior_count:
    for behavior in range(0, len(CNN_transformer.column_names)):
        true_behaviors[behavior] +=  dictionary[behavior]

print(pred_behaviors)
print(true_behaviors)

# Scatterplot

from plots import plot_instance_count_scatter

plot_instance_count_scatter(
    model_wrappers=[old_model,hgb , CNN_transformer],
    output_path="./output/instance_count_scatter.png"
)

# F1 plot

from plots import plot_f1_scores

plot_f1_scores(model_wrappers = [old_model, hgb, CNN_transformer], output_path="output/f1_scores.png")

# Computing time

from plots import plot_computing_times

plot_computing_times(output_path="./output/computing_time.png")