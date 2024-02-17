import os

# determine the device to be used for training and evaluation
#DEVICE = "cuda" # if torch.cuda.is_available() else "cpu"

DEVICE = None
# determine if we will be pinning memory during data loadingy
PIN_MEMORY = True if DEVICE == "cuda" else False

INIT_LR = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 4
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_LENGTH = 1024

BASE_MODEL = "/home/ccy/cellcycle/GFP&RFP_prediction/save_model"
BASE_OUTPUT = "/home/ccy/cellcycle/GFP&RFP_prediction/output"

MODEL_PATH = os.path.join(BASE_MODEL, "/model_path/EUNet.pth")
CHECKPOINT_PATHS = os.path.join(BASE_MODEL, "/model_path/EUnet_checkpoint.ckpt")
RECORD_PATH = os.path.join(BASE_MODEL,"/record/EUNet_record.csv")


PLOT_PATH = os.path.join(BASE_OUTPUT, "/processEUNet.png")
PREDICT_PATHS = os.path.join(BASE_OUTPUT, "/prediction/EUNet.png")



