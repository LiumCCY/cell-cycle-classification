import os


DEVICE = "cuda" 

PIN_MEMORY = True if DEVICE == "cuda" else False

INIT_LR = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 4
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_LENGTH = 1024

BASE_OUTPUT = "/home/ccy/cellcycle/Predict_GFP_RFP/output_GFP/SRGAN"
BASE_MODEL = "/home/ccy/cellcycle/Predict_GFP_RFP/save_model_GFP/SRGAN"

MODEL_PATH = os.path.join(BASE_MODEL, "unet_BCE_dice_BS4*8.pth")
RECORD_PATH = os.path.join(BASE_OUTPUT,"unet_BCE_dice_BS4*8_record.csv")
PLOT_PATH = os.path.join(BASE_OUTPUT, "unet_BCE_dice_BS4*8.png")
PREDICT_PATHS = os.path.join(BASE_OUTPUT, "unet_BCE_dice_BS4*8_pd.png")
CHECKPOINT_PATHS = os.path.join(BASE_MODEL, "checkpoint_unet_BCE_dice_BS4*8.ckpt")


