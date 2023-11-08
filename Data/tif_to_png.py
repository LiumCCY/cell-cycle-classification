import os 
import cv2

bright_field = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP/brightfield"

def tif_to_png(input_path,output_path_root,save_file_name):
    
    file = os.listdir(input_path)
    print(len(file))
    for i in range(len(file)):
        try:
            image = cv2.imread(os.path.join(input_path, file[i]))
            if image is None:
                raise Exception("Failed to read image")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #blurred = cv2.blur(image, (3, 3))
            #corrected_image = blurred / 255.0

            # Gamma
            #corrected_image = np.power(normalized_image, 0.5)

            # [0, 255]
            #corrected_image = (corrected_image * 255).astype(np.uint8)
   
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
            os.chdir(output_path_root)
            output_path = os.path.join(output_path_root, save_file_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cv2.imwrite(os.path.join(output_path, "{}.png".format(os.path.splitext(file[i])[0])),image)
            
        except Exception as e:
            print("Failed to process image {}: {}".format(file[i], e))
        finally:
            if image is not None:
                del image

tif_to_png(bright_field, "/home/ccy/cellcycle/data/FUCCI_RFP_GFP", "brightfield_png")
