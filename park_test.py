from __future__ import division
import argparse
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from PIL import Image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from Parking import Parking
import pickle
cwd = os.getcwd()

def img_process(test_images,park):
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)
    
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)
    
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)
    
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)
    
    list_of_lines = list(map(park.hough_lines, roi_images))
    
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines)) 
    park.show_images(line_images)
    
    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)
        
    park.show_images(rect_images)
    
    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)
        
    park.show_images(delineated)
    final_spot_dict = spot_pos[0]
    print(len(final_spot_dict))

    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    park.save_images_for_cnn(test_images[0],final_spot_dict)
    
    return final_spot_dict
def keras_model(weights_path):    
    model = load_model(weights_path)
    return model
def img_test(test_images,final_spot_dict,model,class_dictionary,model_name='VGG16'):
    for i in range (len(test_images)):
        predicted_images = park.predict_on_image(test_images[i],final_spot_dict,model,class_dictionary,model_name=model_name)
def video_test(video_name,final_spot_dict,model,class_dictionary,model_name='VGG16'):
    name = video_name
    cap = cv2.VideoCapture(name)
    park.predict_on_video(name,final_spot_dict,model,class_dictionary,model_name=model_name,ret=True)
    
def compare_models_on_image(test_image, spot_dict, model1, model2, class_dict):
    """Display detection results of two models side by side for comparison"""
    park = Parking()
    
    # Detect with both models
    result1 = park.predict_on_image(
        test_image, spot_dict, model1, class_dict, 
        model_name="VGG16", save_result=True, filename="vgg_result.jpg"
    )
    
    result2 = park.predict_on_image(
        test_image, spot_dict, model2, class_dict, 
        model_name="Simple CNN", save_result=True, filename="cnn_result.jpg"
    )
    
    # Display results side by side
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    plt.title('VGG16 Detection Results', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    plt.title('Simple CNN Detection Results', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison results saved as model_comparison.png")

def compare_models_on_video(video_name, spot_dict, model1, model2, class_dict):
    """Process video with both models and save results"""
    park = Parking()
    
    # Process video with VGG16
    park.predict_on_video(
        video_name, spot_dict, model1, class_dict,
        model_name="VGG16", save_video=True, ret=True
    )
    
    # Process video with Simple CNN
    park.predict_on_video(
        video_name, spot_dict, model2, class_dict,
        model_name="CNN", save_video=True, ret=True
    )
    
    print(" output_VGG16.mp4 (VGG16 detection results)")
    print(" output_CNN.mp4 (Simple CNN detection results)")    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parking Spot Detection - Model Comparison Tool')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['vgg', 'cnn', 'compare'],
                        help='Run mode: vgg(only use VGG16), cnn(only use CNN), compare(compare both models)')
    parser.add_argument('--test_type', type=str, default='both',
                        choices=['image', 'video', 'both'],
                        help='Test type: image(only test images), video(only test videos), both(test both)')
    
    args = parser.parse_args()
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    video_name = 'parking_video.mp4'
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'
    park = Parking()
    park.show_images(test_images)
    final_spot_dict = img_process(test_images,park)
    
      # Load different models based on run mode
    if args.mode == 'vgg':
        print("Loading VGG16 model")
        model = keras_model('VGG16_model.keras')
        model_name = "VGG16"
        if args.test_type in ['image', 'both']:
            img_test(test_images, final_spot_dict, model, class_dictionary, model_name)
        if args.test_type in ['video', 'both'] and os.path.exists(video_name):
            video_test(video_name, final_spot_dict, model, class_dictionary, model_name)
            
    elif args.mode == 'cnn':
        print("Loading Simple CNN model")
        model = keras_model('cnn_model.keras')
        model_name = "Simple CNN"
        if args.test_type in ['image', 'both']:
            img_test(test_images, final_spot_dict, model, class_dictionary, model_name)
        if args.test_type in ['video', 'both'] and os.path.exists(video_name):
            video_test(video_name, final_spot_dict, model, class_dictionary, model_name)
    else:  # compare mode
        print("Loading both models for comparison...")
        
        # Check if both model files exist
        if not os.path.exists('VGG16_model.keras'):
            print("Error: Cannot find VGG16_model.keras (VGG16 model)")
            exit(1)
        if not os.path.exists('cnn_model.keras'):
            print("Error: Cannot find cnn_model.keras (Simple CNN model)")
            exit(1)
        
        # Load both models
        vgg_model = keras_model('VGG16_model.keras')
        cnn_model = keras_model('cnn_model.keras')
        
        print(" Both models loaded successfully!")
        print(" VGG16: VGG16_model.keras")
        print(" Simple CNN: cnn_model.keras")
 
 # Image comparison
        if args.test_type in ['image', 'both'] and len(test_images) > 0:
            print("Performing image detection comparison")
            compare_models_on_image(
                test_images[0], 
                final_spot_dict, 
                vgg_model, 
                cnn_model, 
                class_dictionary
            )
        
        # Video comparison
        if args.test_type in ['video', 'both'] and os.path.exists(video_name):
            print("Performing video detection comparison")
            compare_models_on_video(
                video_name,
                final_spot_dict,
                vgg_model,
                cnn_model,
                class_dictionary
            )
    
    print("All tasks completed!")
