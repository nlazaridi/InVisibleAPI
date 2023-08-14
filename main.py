from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torchvision.models as models
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from llama_cpp import Llama
import os
import time

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.use_static_for_external = True
features_blobs = []
# Helper function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Welcome page with image upload form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No file selected.'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
            return redirect(url_for('waiting_page', filename=filename))
    return render_template('welcome.html')

# Waiting page
@app.route('/waiting/<filename>')
def waiting_page(filename):
    return render_template('waiting.html', filename=filename)

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

# API endpoint to check the inference status
@app.route('/check_status/<filename>')
def check_status(filename):
    # Simulate inference time
    time.sleep(5)

    # Perform classification on the uploaded image here
    # load the labels
    image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the model
    model = load_model()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    img=Image.open(image_path)

    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        type_of_env = 'indoor'
    #    print('--TYPE OF ENVIRONMENT: indoor')
    else:
        type_of_env = 'outdoor'
    #    print('--TYPE OF ENVIRONMENT: outdoor')

    new_classes = ['altar', 'apse', 'bell_tower', 'column', 'dome(inner)', 'dome(outer)', 'flying_buttress', 'gargoyle', 'stained_glass', 'vault']
    scene_cat = [classes[idx[0]]]
    if classes[idx[0]] in new_classes:
        scene_cat.append(classes[idx[1]])
    else:    
        for i in range(len(new_classes)):
            for j in range(1,5):
                if new_classes[i] == classes[idx[j]]:
                    scene_cat.append(new_classes[i])

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)


    attributes_list = []
    found_light=False
    for i in range(-1,-10,-1):
        if 'light' in labels_attribute[idx_a[i]] and not found_light:
            lighting_cond = labels_attribute[idx_a[i]]
            found_light=True
        elif 'horizon' in labels_attribute[idx_a[i]] or 'cloth' in labels_attribute[idx_a[i]]:
            continue
        else:  
            attributes_list.append(labels_attribute[idx_a[i]])

    
    if len(scene_cat)==1:
        input_text = '### Instruction: Describe without adding too much random information an ' + type_of_env + ' image of a ' + scene_cat[0] +'. The lighting conditions are ' + lighting_cond + '. Some more relevant words that you can use are: ' + attributes_list[0] +', ' + attributes_list[1] + '. \n### Response:'
    else:
        input_text = '### Instruction: Describe without adding too much random information an ' + type_of_env + ' image of a ' + scene_cat[0] + ' and a '+ scene_cat[1] +'. The lighting conditions are ' + lighting_cond + '. Some more relevant words that you can use are: ' + attributes_list[0] +', ' + attributes_list[1] + '. \n### Response:'
    llm = Llama(model_path='/data2/nlazaridis/places_finetunning/models/stable-vicuna-13B.ggmlv3.q4_K_S.bin')
    output = llm(input_text,
                max_tokens=200,
                temperature=0.4,
                stop=['\n'],
                echo=True)
    print(output['choices'][0]['text'])

    response_index = output['choices'][0]['text'].find("### Response:")
    if response_index != -1:
        text_message = output['choices'][0]['text'][response_index + len("### Response:"):]
        text_message = text_message.strip()  # Remove leading/trailing whitespaces
    else:
        text_message = None    

    file_url = url_for('get_file', filename=filename)

    # Replace the following code with your actual classification model logic
    #classification_output = "Example classification output"
    classification_output = text_message
    return jsonify({'classification': classification_output}), 200

# Final page with image and classification output
@app.route('/result/<filename>')
def result_page(filename):
    file_url = url_for('get_file', filename=filename)
    classification_output = request.args.get('classification', 'No classification available')
    return render_template('result.html', filename=filename, classification=classification_output, img_url = file_url)

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365_enhanced.txt'
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365_enhanced.txt'
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]

    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_model():
    # this model has a last conv feature map as 14x14
    # th architecture to use
    arch = 'resnet18'
    model_file = 'resnet18_bestBEST.pth.tar'
    model = models.__dict__[arch](num_classes=375)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

if __name__ == '__main__':
    app.run(debug=True)
